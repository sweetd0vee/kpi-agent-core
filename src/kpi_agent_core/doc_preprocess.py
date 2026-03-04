"""
Интеллектуальная предобработка больших документов для каскада целей.

Вместо передачи сырого текста цели: разбиение на чанки → отбор по релевантности к целям
руководителя (эмбеддинги) → в промпт попадают только релевантные фрагменты. Это повышает
качество при слабых локальных моделях и укладывается в лимит контекста.
"""
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Размер чанка и перекрытие (символы)
DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 200
# Макс. символов на документ после отбора релевантных чанков (компактный контекст для слабой модели)
DEFAULT_MAX_CHARS_PER_DOC_AFTER_RETRIEVAL = 6000
# Топ‑k чанков на документ по сходству с запросом
DEFAULT_TOP_K_CHUNKS_PER_DOC = 5


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Разбить текст на чанки фиксированного размера с перекрытием.
    Границы по границам строк, чтобы не резать середину слова.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            last_newline = text.rfind("\n", start, end + 1)
            if last_newline > start:
                end = last_newline + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def chunk_by_sections(
    text: str,
    max_chars: int = 3500,
    section_pattern: Optional[str] = None,
) -> List[str]:
    """
    Разбить текст по секциям (заголовки вида ##, 1., 2., или «Раздел 1»).
    Секции больше max_chars дополнительно режутся через chunk_text.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if section_pattern is None:
        section_pattern = r"(?:\n\s*#{1,3}\s+|\n\s*\d+[.)]\s+|\n\s*[А-Яа-яA-Za-z]+\s+—\s+)"
    parts = re.split(section_pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text] if len(text) <= max_chars else chunk_text(text, max_chars)
    result = []
    for p in parts:
        if len(p) <= max_chars:
            result.append(p)
        else:
            result.extend(chunk_text(p, max_chars))
    return result


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def build_chunk_index(
    documents: Dict[str, str],
    get_embedding: Callable[[str], List[float]],
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    use_sections: bool = True,
) -> List[Dict[str, Any]]:
    """
    По словарю документов (doc_type -> full text) построить индекс чанков с эмбеддингами.
    Результат можно сохранить в JSON и переиспользовать, чтобы не пересчитывать эмбеддинги.

    :return: список записей {doc_type, chunk_index, text, embedding}
    """
    index = []
    for doc_type, content in documents.items():
        if not content or content.strip() == "":
            continue
        if use_sections:
            chunks = chunk_by_sections(content, max_chars=chunk_size)
        else:
            chunks = chunk_text(content, max_chars=chunk_size, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            try:
                emb = get_embedding(chunk)
                index.append({
                    "doc_type": doc_type,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": emb,
                })
            except Exception:
                continue
    return index


def retrieve_relevant_chunks(
    chunk_index: List[Dict[str, Any]],
    query_text: str,
    get_embedding: Callable[[str], List[float]],
    top_k_per_doc: int = DEFAULT_TOP_K_CHUNKS_PER_DOC,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC_AFTER_RETRIEVAL,
) -> Dict[str, str]:
    """
    По запросу (например цели руководителя) отобрать для каждого типа документа
    наиболее релевантные чанки и склеить их в один текст. Возвращает словарь
    doc_type -> релевантный фрагмент (до max_chars_per_doc на документ).
    """
    if not chunk_index or not query_text.strip():
        return {}
    try:
        query_emb = get_embedding(query_text[:32000])
    except Exception:
        return {}

    by_doc_type: Dict[str, List[Tuple[float, str]]] = {}
    for entry in chunk_index:
        doc_type = entry.get("doc_type", "")
        text = entry.get("text", "")
        emb = entry.get("embedding")
        if not text or not emb:
            continue
        sim = _cosine_similarity(query_emb, emb)
        if doc_type not in by_doc_type:
            by_doc_type[doc_type] = []
        by_doc_type[doc_type].append((sim, text))

    result = {}
    for doc_type, scored in by_doc_type.items():
        scored.sort(key=lambda x: -x[0])
        selected = scored[:top_k_per_doc]
        parts = []
        total = 0
        for _, text in selected:
            if total + len(text) > max_chars_per_doc:
                take = max_chars_per_doc - total
                if take > 100:
                    parts.append(text[:take] + "\n[...]")
                break
            parts.append(text)
            total += len(text)
        if parts:
            result[doc_type] = "\n\n".join(parts)
    return result


def preprocess_documents_simple(
    documents: Dict[str, str],
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC_AFTER_RETRIEVAL,
    use_sections: bool = True,
    placeholder_empty: str = "(документ не приложен)",
) -> Dict[str, str]:
    """
    Предобработка без эмбеддингов: каждый документ обрезается до max_chars_per_doc.
    Подходит, когда документов мало (например 5 фиксированных), а нужно лишь уместить
    контекст в лимит модели. Режет по началу текста или по секциям (берёт секции с начала).

    :param documents: словарь doc_type -> полный текст
    :param max_chars_per_doc: макс. символов на документ в результате
    :param use_sections: если True, резать по секциям и брать секции с начала до лимита
    :return: словарь doc_type -> обрезанный текст
    """
    result = {}
    for doc_type, content in documents.items():
        if not content or not content.strip():
            result[doc_type] = placeholder_empty
            continue
        text = content.strip()
        if len(text) <= max_chars_per_doc:
            result[doc_type] = text
            continue
        if use_sections:
            chunks = chunk_by_sections(text, max_chars=max_chars_per_doc * 2)
            parts = []
            total = 0
            for c in chunks:
                if total + len(c) > max_chars_per_doc:
                    take = max_chars_per_doc - total
                    if take > 100:
                        parts.append(c[:take] + "\n\n[...]")
                    break
                parts.append(c)
                total += len(c)
            result[doc_type] = "\n\n".join(parts) if parts else text[:max_chars_per_doc] + "\n\n[...]"
        else:
            result[doc_type] = text[:max_chars_per_doc] + "\n\n[... обрезано ...]"
    return result


def preprocess_documents_for_cascade(
    documents: Dict[str, str],
    leader_goals_text: str,
    get_embedding: Callable[[str], List[float]],
    top_k_per_doc: int = DEFAULT_TOP_K_CHUNKS_PER_DOC,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC_AFTER_RETRIEVAL,
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    use_sections: bool = True,
    placeholder_empty: str = "(документ не приложен)",
) -> Dict[str, str]:
    """
    Интеллектуальная предобработка: большие документы режутся на чанки, по целям
    руководителя отбираются самые релевантные фрагменты, в итоге словарь с короткими
    текстами для передачи в LLM (качество выше при слабых моделях и ограниченном контексте).

    :param documents: словарь doc_type -> полный текст (может быть очень большим)
    :param leader_goals_text: цели главного руководителя (строка) — запрос для отбора релевантности
    :param get_embedding: функция (text) -> embedding
    :param top_k_per_doc: сколько чанков брать на документ по сходству
    :param max_chars_per_doc: макс. символов на документ в результате
    :return: словарь doc_type -> релевантный фрагмент (или placeholder для пустых)
    """
    if not leader_goals_text or not leader_goals_text.strip():
        return documents

    index = build_chunk_index(
        documents,
        get_embedding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_sections=use_sections,
    )
    relevant = retrieve_relevant_chunks(
        index,
        leader_goals_text,
        get_embedding,
        top_k_per_doc=top_k_per_doc,
        max_chars_per_doc=max_chars_per_doc,
    )

    result = {}
    for doc_type, full_text in documents.items():
        if not full_text or not full_text.strip():
            result[doc_type] = placeholder_empty
            continue
        if doc_type in relevant and relevant[doc_type]:
            result[doc_type] = relevant[doc_type]
        else:
            if len(full_text) > max_chars_per_doc:
                result[doc_type] = full_text[:max_chars_per_doc] + "\n\n[... обрезано ...]"
            else:
                result[doc_type] = full_text
    return result


def save_chunk_index(index: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Сохранить индекс чанков в JSON (эмбеддинги — числа, текст — строки)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")


def load_chunk_index(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Загрузить индекс чанков из JSON."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []
