"""
Эмбеддинги документов через Ollama (например qwen3-embedding).

Вход — готовый текст (например пронумерованные чеклисты). Вызов API эмбеддингов, сохранение результата.
"""
import json
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from kpi_agent_core.prompts import EMBED_DOCUMENT_TEMPLATE

# Модель по умолчанию (должна быть загружена в Ollama: ollama pull qwen3-embedding)
DEFAULT_EMBED_MODEL = "qwen3-embedding"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def document_to_embedding_text(
    text: str,
    document_type: str = "business_plan_checklist",
) -> str:
    """
    Подготавливает готовый текст документа (например чеклист) для эмбеддинга:
    добавляет тип документа. Весь текст передаётся в модель без обрезки.
    """
    body = text.strip()
    return EMBED_DOCUMENT_TEMPLATE.format(
        document_type=document_type,
        content=body,
    ).strip()


def get_embedding_ollama(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> list[float]:
    """
    Получить вектор эмбеддинга для одного текста через Ollama API.

    :param text: строка для эмбеддинга
    :param model: имя модели (например qwen3-embedding)
    :param base_url: базовый URL Ollama (без слэша в конце)
    :return: список float — вектор эмбеддинга
    :raises httpx.HTTPError: при ошибке запроса
    :raises ValueError: если в ответе нет вектора
    """
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": text}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings or not isinstance(embeddings, list):
        raise ValueError("В ответе Ollama нет поля embeddings или оно пустое")
    # Один текст — один вектор
    vec = embeddings[0] if isinstance(embeddings[0], list) else embeddings
    return [float(x) for x in vec]


def embed_document(
    text: str,
    document_type: str = "business_plan_checklist",
    document_id: Optional[str] = None,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    save_path: Optional[Union[str, Path]] = None,
    source_path: Optional[Union[str, Path]] = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Создать эмбеддинг для готового текста (например пронумерованный чеклист),
    опционально сохранить результат. source_path сохраняется в JSON для последующего
    поиска по релевантности и подстановки полного текста в каскад.

    :param text: готовый текст документа (чеклист и т.д.)
    :param document_type: тип документа (business_plan_checklist, strategy_checklist и т.д.)
    :param document_id: опциональный id для метаданных
    :param model: модель Ollama для эмбеддингов
    :param base_url: базовый URL Ollama
    :param save_path: если задан — сохранить результат в JSON (embedding + метаданные)
    :param source_path: путь к исходному .txt — сохраняется в JSON для retrieval (загрузка полного текста)
    :param timeout: таймаут запроса (секунды)
    :return: embedding, text_preview, document_type, model, document_id
    """
    prepared = document_to_embedding_text(text, document_type=document_type)
    # Эмбеддинг создаётся для всего текста документа (без обрезки)
    embedding = get_embedding_ollama(prepared, model=model, base_url=base_url, timeout=timeout)
    text_preview = prepared[:2000] + ("..." if len(prepared) > 2000 else "")
    src_str = str(source_path) if source_path else None
    result = {
        "embedding": embedding,
        "text_preview": text_preview,
        "text_full": prepared,
        "document_type": document_type,
        "model": model,
        "document_id": document_id,
        "source_path": src_str,
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Сохраняем полный текст, чтобы при retrieval использовать весь документ
        to_save = {
            "embedding": result["embedding"],
            "text_preview": result["text_preview"],
            "text_full": prepared,
            "document_type": result["document_type"],
            "model": result["model"],
            "document_id": result["document_id"],
            "source_path": src_str,
            "text_length": len(prepared),
        }
        path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
