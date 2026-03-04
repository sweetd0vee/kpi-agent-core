"""
Каскад целей главного руководителя в таблицу для подчинённых.

Вход: список целей руководителя, приложенные документы (Бизнес-план, Стратегия, Регламент,
Положение о департаменте, известные цели), список подчинённых.
Выход: таблица (markdown), сформированная локальной LLM — релевантные цели и каскадированные формулировки.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from kpi_agent_core.prompts import CASCADE_LEADER_TO_TABLE_SYSTEM, CASCADE_LEADER_TO_TABLE_USER

# Ключи для структуры документов (как в промпте)
DOC_BUSINESS_PLAN = "business_plan"
DOC_STRATEGY = "strategy"
DOC_REGULATION = "regulation"
DOC_DEPARTMENT_REGULATION = "department_regulation"
DOC_KNOWN_GOALS = "known_goals"

DOC_KEYS = [
    DOC_BUSINESS_PLAN,
    DOC_STRATEGY,
    DOC_REGULATION,
    DOC_DEPARTMENT_REGULATION,
    DOC_KNOWN_GOALS,
]

# Максимум символов на документ в промпте (чтобы не превысить контекст LLM)
MAX_CHARS_PER_DOC = 40000
PLACEHOLDER_EMPTY = "(документ не приложен)"


def build_documents_dict(
    business_plan: Optional[str] = None,
    strategy: Optional[str] = None,
    regulation: Optional[str] = None,
    department_regulation: Optional[str] = None,
    known_goals: Optional[str] = None,
    max_chars_per_doc: int = MAX_CHARS_PER_DOC,
) -> Dict[str, str]:
    """
    Собрать словарь документов для промпта. Каждый текст обрезается до max_chars_per_doc.
    Пустые или None заменяются на плейсхолдер.
    """
    def trim(s: Optional[str]) -> str:
        if not s or not s.strip():
            return PLACEHOLDER_EMPTY
        t = s.strip()
        if len(t) > max_chars_per_doc:
            t = t[:max_chars_per_doc] + "\n\n[... документ обрезан ...]"
        return t

    return {
        DOC_BUSINESS_PLAN: trim(business_plan),
        DOC_STRATEGY: trim(strategy),
        DOC_REGULATION: trim(regulation),
        DOC_DEPARTMENT_REGULATION: trim(department_regulation),
        DOC_KNOWN_GOALS: trim(known_goals),
    }


def load_documents_from_paths(
    business_plan_path: Optional[Union[str, Path]] = None,
    strategy_path: Optional[Union[str, Path]] = None,
    regulation_path: Optional[Union[str, Path]] = None,
    department_regulation_path: Optional[Union[str, Path]] = None,
    known_goals_path: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
    max_chars_per_doc: int = MAX_CHARS_PER_DOC,
) -> Dict[str, str]:
    """
    Загрузить тексты документов из файлов и собрать словарь для промпта.
    Если путь не передан или файл не найден — в промпте будет плейсхолдер «документ не приложен».
    """
    def read_path(path: Optional[Union[str, Path]]) -> Optional[str]:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            return p.read_text(encoding=encoding)
        except Exception:
            return None

    return build_documents_dict(
        business_plan=read_path(business_plan_path),
        strategy=read_path(strategy_path),
        regulation=read_path(regulation_path),
        department_regulation=read_path(department_regulation_path),
        known_goals=read_path(known_goals_path),
        max_chars_per_doc=max_chars_per_doc,
    )


def format_leader_goals(leader_goals: Union[str, List[Any]]) -> str:
    """Привести список целей руководителя к тексту для промпта."""
    if isinstance(leader_goals, str):
        return leader_goals.strip() or "(цели не указаны)"
    if isinstance(leader_goals, list):
        lines = []
        for i, g in enumerate(leader_goals, 1):
            if isinstance(g, dict):
                title = g.get("title") or g.get("name") or ""
                desc = g.get("description") or g.get("desc") or ""
                kpi = g.get("kpi") or []
                kpi_str = ", ".join(kpi) if isinstance(kpi, list) else str(kpi)
                lines.append(f"{i}. {title}\n   {desc}\n   KPI: {kpi_str}")
            else:
                lines.append(f"{i}. {g}")
        return "\n".join(lines) if lines else "(цели не указаны)"
    return "(цели не указаны)"


def cascade_leader_goals_to_table(
    leader_goals: Union[str, List[Any]],
    documents: Dict[str, str],
    subordinates: List[str],
    invoke_llm: Callable[[List[Dict[str, str]]], str],
    *,
    subordinates_sep: str = ", ",
) -> str:
    """
    Сформировать промпт с целями руководителя и приложенными документами, вызвать LLM,
    получить таблицу каскадированных целей для подчинённых.

    :param leader_goals: цели главного руководителя (строка или список dict с title, description, kpi)
    :param documents: словарь от build_documents_dict или load_documents_from_paths
    :param subordinates: список имён/департаментов подчинённых
    :param invoke_llm: функция (messages) -> content (ответ LLM)
    :param subordinates_sep: разделитель в списке подчинённых в промпте
    :return: ответ LLM (таблица в markdown и/или текст)
    """
    leader_text = format_leader_goals(leader_goals)
    subs_text = subordinates_sep.join(subordinates) if subordinates else "не указаны — предложи типовые по документам"

    docs = {k: documents.get(k) or PLACEHOLDER_EMPTY for k in DOC_KEYS}

    user_content = CASCADE_LEADER_TO_TABLE_USER.format(
        leader_goals=leader_text,
        business_plan=docs[DOC_BUSINESS_PLAN],
        strategy=docs[DOC_STRATEGY],
        regulation=docs[DOC_REGULATION],
        department_regulation=docs[DOC_DEPARTMENT_REGULATION],
        known_goals=docs[DOC_KNOWN_GOALS],
        subordinates=subs_text,
    )

    messages = [
        {"role": "system", "content": CASCADE_LEADER_TO_TABLE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return invoke_llm(messages)


def cascade_leader_goals_to_table_with_preprocess(
    leader_goals: Union[str, List[Any]],
    documents: Dict[str, str],
    subordinates: List[str],
    invoke_llm: Callable[[List[Dict[str, str]]], str],
    get_embedding: Callable[[str], List[float]],
    *,
    top_k_per_doc: int = 5,
    max_chars_per_doc: int = 6000,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    subordinates_sep: str = ", ",
) -> str:
    """
    То же, что cascade_leader_goals_to_table, но с предобработкой через эмбеддинги:
    отбор релевантных чанков по целям руководителя. Имеет смысл при очень длинных документах.
    """
    from kpi_agent_core.doc_preprocess import preprocess_documents_for_cascade

    leader_text = format_leader_goals(leader_goals)
    preprocessed = preprocess_documents_for_cascade(
        documents,
        leader_text,
        get_embedding,
        top_k_per_doc=top_k_per_doc,
        max_chars_per_doc=max_chars_per_doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        placeholder_empty=PLACEHOLDER_EMPTY,
    )
    return cascade_leader_goals_to_table(
        leader_goals,
        preprocessed,
        subordinates,
        invoke_llm,
        subordinates_sep=subordinates_sep,
    )


def cascade_leader_goals_to_table_simple(
    leader_goals: Union[str, List[Any]],
    documents: Dict[str, str],
    subordinates: List[str],
    invoke_llm: Callable[[List[Dict[str, str]]], str],
    *,
    max_chars_per_doc: int = 6000,
    subordinates_sep: str = ", ",
) -> str:
    """
    Каскад в таблицу с простой предобработкой без эмбеддингов: каждый документ
    обрезается до max_chars_per_doc (по секциям с начала). Рекомендуется, когда
    документов мало (например 5 фиксированных) — эмбеддинги не нужны.
    """
    from kpi_agent_core.doc_preprocess import preprocess_documents_simple

    preprocessed = preprocess_documents_simple(
        documents,
        max_chars_per_doc=max_chars_per_doc,
        placeholder_empty=PLACEHOLDER_EMPTY,
    )
    return cascade_leader_goals_to_table(
        leader_goals,
        preprocessed,
        subordinates,
        invoke_llm,
        subordinates_sep=subordinates_sep,
    )
