"""
kpi-agent-core — ядро приложения AI KPI.

Модели, промпты и граф каскадирования целей (LangGraph).
Бэкенд использует этот пакет для /api/chat/completions и /api/chat/cascade.
"""

from kpi_agent_core.models import (
    CascadeState,
    GoalItem,
    SubdivisionGoal,
)
from kpi_agent_core.prompts import (
    CASCADE_EXTRACT_GOALS,
    CASCADE_SPLIT_BY_SUBDIVISION,
    CASCADE_CHECK_CHECKLIST,
    CHAT_SYSTEM_PROMPT,
    EMBED_DOCUMENT_TEMPLATE,
)
from kpi_agent_core.embeddings import (
    document_to_embedding_text,
    get_embedding_ollama,
    embed_document,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)

# Ленивый импорт графа: не тянет langgraph/uuid_utils при использовании только эмбеддингов (скрипт run_embed)
def __getattr__(name):
    if name == "build_cascade_graph":
        from kpi_agent_core.graph import build_cascade_graph
        return build_cascade_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CascadeState",
    "GoalItem",
    "SubdivisionGoal",
    "build_cascade_graph",
    "CASCADE_EXTRACT_GOALS",
    "CASCADE_SPLIT_BY_SUBDIVISION",
    "CASCADE_CHECK_CHECKLIST",
    "CHAT_SYSTEM_PROMPT",
    "EMBED_DOCUMENT_TEMPLATE",
    "document_to_embedding_text",
    "get_embedding_ollama",
    "embed_document",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
]
