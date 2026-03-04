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
    CASCADE_LEADER_TO_TABLE_SYSTEM,
    CASCADE_LEADER_TO_TABLE_USER,
    CHAT_SYSTEM_PROMPT,
    EMBED_DOCUMENT_TEMPLATE,
    RETRIEVED_DOCUMENTS_HEADER,
)
from kpi_agent_core.embeddings import (
    document_to_embedding_text,
    get_embedding_ollama,
    embed_document,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)

# Ленивый импорт графа и retrieval: не тянет langgraph при использовании только эмбеддингов
def __getattr__(name):
    if name == "build_cascade_graph":
        from kpi_agent_core.graph import build_cascade_graph
        return build_cascade_graph
    if name in ("load_embedding_index", "retrieve_relevant_documents", "retrieved_texts_to_checklists_string"):
        from kpi_agent_core import retrieval
        return getattr(retrieval, name)
    if name in ("cascade_leader_goals_to_table", "cascade_leader_goals_to_table_with_preprocess",
                "cascade_leader_goals_to_table_simple",
                "build_documents_dict", "load_documents_from_paths", "format_leader_goals",
                "DOC_BUSINESS_PLAN", "DOC_STRATEGY", "DOC_REGULATION", "DOC_DEPARTMENT_REGULATION", "DOC_KNOWN_GOALS"):
        from kpi_agent_core import cascade_to_table
        return getattr(cascade_to_table, name)
    if name in ("preprocess_documents_for_cascade", "preprocess_documents_simple",
                "build_chunk_index", "retrieve_relevant_chunks",
                "chunk_text", "chunk_by_sections", "save_chunk_index", "load_chunk_index"):
        from kpi_agent_core import doc_preprocess
        return getattr(doc_preprocess, name)
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
    "RETRIEVED_DOCUMENTS_HEADER",
    "CASCADE_LEADER_TO_TABLE_SYSTEM",
    "CASCADE_LEADER_TO_TABLE_USER",
    "cascade_leader_goals_to_table",
    "cascade_leader_goals_to_table_with_preprocess",
    "cascade_leader_goals_to_table_simple",
    "build_documents_dict",
    "load_documents_from_paths",
    "format_leader_goals",
    "DOC_BUSINESS_PLAN",
    "DOC_STRATEGY",
    "DOC_REGULATION",
    "DOC_DEPARTMENT_REGULATION",
    "DOC_KNOWN_GOALS",
    "preprocess_documents_for_cascade",
    "preprocess_documents_simple",
    "build_chunk_index",
    "retrieve_relevant_chunks",
    "chunk_text",
    "chunk_by_sections",
    "save_chunk_index",
    "load_chunk_index",
    "document_to_embedding_text",
    "get_embedding_ollama",
    "embed_document",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "load_embedding_index",
    "retrieve_relevant_documents",
    "retrieved_texts_to_checklists_string",
]
