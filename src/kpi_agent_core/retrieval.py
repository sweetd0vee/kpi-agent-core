"""
Поиск по эмбеддингам: по запросу (например текст целей) подобрать релевантные ключевые документы,
чтобы подставить их полный текст в промпт каскада. LLM получает текст, не векторы.
"""
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from kpi_agent_core.embeddings import get_embedding_ollama


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Косинусное сходство между двумя векторами (эмбеддинги Ollama уже нормализованы)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def load_embedding_index(embeddings_dir: Union[str, Path]) -> List[dict]:
    """
    Загрузить индекс эмбеддингов из папки: все JSON с embedding, document_id, text_full/text_preview, source_path.

    :param embeddings_dir: путь к папке (например out/)
    :return: список записей {embedding, document_id, document_type, source_path?, text_full?, text_preview, ...}
    """
    dir_path = Path(embeddings_dir)
    if not dir_path.is_dir():
        return []
    index = []
    for path in dir_path.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            emb = data.get("embedding")
            if emb and isinstance(emb, list):
                index.append({
                    "embedding": emb,
                    "document_id": data.get("document_id") or path.stem,
                    "document_type": data.get("document_type", ""),
                    "source_path": data.get("source_path"),
                    "text_full": data.get("text_full", ""),
                    "text_preview": data.get("text_preview", ""),
                    "path": str(path),
                })
        except Exception:
            continue
    return index


def _get_document_full_text(entry: dict, base_dir: Optional[Path] = None) -> str:
    """Получить полный текст документа: source_path → text_full (весь сохранённый текст) → text_preview."""
    src = entry.get("source_path")
    if src and base_dir is not None:
        path = (base_dir / src) if not Path(src).is_absolute() else Path(src)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
    if src:
        try:
            p = Path(src)
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
    # Полный текст из JSON (тот же, по которому строился эмбеддинг)
    if entry.get("text_full"):
        return entry["text_full"]
    return entry.get("text_preview") or ""


def retrieve_relevant_documents(
    query_text: str,
    index: List[dict],
    top_k: int = 5,
    get_embedding: Optional[Callable[[str], List[float]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str]]:
    """
    По запросу (например текст целей) вернуть топ‑k наиболее релевантных документов с полным текстом.
    Результат можно склеить и передать в граф каскада как checklists_text.

    :param query_text: запрос (цели руководства или краткое описание)
    :param index: индекс из load_embedding_index()
    :param top_k: сколько документов вернуть
    :param get_embedding: функция (text) -> embedding; по умолчанию get_embedding_ollama
    :param base_dir: базовая папка для разрешения относительных source_path (например корень kpi-agent-core)
    :return: список пар (document_id, full_text)
    """
    if not index:
        return []
    get_emb = get_embedding or (lambda t: get_embedding_ollama(t))
    query_emb = get_emb(query_text[:32000])
    base = Path(base_dir) if base_dir else None
    scored = []
    for entry in index:
        sim = _cosine_similarity(query_emb, entry["embedding"])
        scored.append((sim, entry))
    scored.sort(key=lambda x: -x[0])
    result = []
    for _, entry in scored[:top_k]:
        full_text = _get_document_full_text(entry, base)
        if full_text:
            result.append((entry["document_id"], full_text))
    return result


def retrieved_texts_to_checklists_string(docs: List[Tuple[str, str]], separator: str = "\n\n---\n\n") -> str:
    """Склеить тексты выбранных документов в одну строку для поля checklists_text в графе."""
    return separator.join(t for _, t in docs)
