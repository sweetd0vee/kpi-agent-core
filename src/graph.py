"""
Граф LangGraph: каскадирование целей от руководства к подразделениям.

Вход: текст целей, текст чеклистов, опциональный контекст.
Выход: состояние с extracted_goals и subdivision_goals для API.
"""
import json
import re
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import START, END, StateGraph

from kpi_agent_core.prompts import (
    CASCADE_EXTRACT_GOALS,
    CASCADE_SPLIT_BY_SUBDIVISION,
    CASCADE_CHECK_CHECKLIST,
)


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Извлечь JSON-массив из ответа LLM (с учётом ```json ... ```)."""
    cleaned = raw.strip()
    for pattern in (r"^```(?:json)?\s*", r"\s*```\s*$"):
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return []


# Состояние графа: тип для узлов (обновляют часть полей)
class CascadeGraphState(TypedDict, total=False):
    goals_text: str
    checklists_text: str
    context: dict[str, Any]
    extracted_goals: list[dict[str, Any]]
    subdivision_goals: list[dict[str, Any]]
    raw_output: Optional[str]
    error: Optional[str]


def build_cascade_graph(
    invoke_llm: Callable[[list[dict[str, str]]], str],
    *,
    model: str = "gpt-4o-mini",
) -> Any:
    """
    Собрать и скомпилировать граф каскадирования.

    :param invoke_llm: функция (messages) -> content; вызывается из узлов для запросов к LLM.
    :param model: имя модели (для логов; вызов к модели делает invoke_llm).
    :return: скомпилированный граф (invoke(state) -> state).
    """
    model = model or "gpt-4o-mini"

    def node_extract_goals(state: CascadeGraphState) -> dict[str, Any]:
        goals_text = state.get("goals_text") or ""
        if not goals_text.strip():
            return {"error": "Нет текста целей для извлечения", "extracted_goals": []}
        prompt = CASCADE_EXTRACT_GOALS.format(goals_text=goals_text)
        messages = [{"role": "user", "content": prompt}]
        try:
            content = invoke_llm(messages)
            if not content:
                return {"error": "Пустой ответ LLM при извлечении целей", "extracted_goals": []}
            items = _parse_json_array(content)
            goals = []
            for it in items:
                if isinstance(it, dict) and it.get("title"):
                    goals.append({
                        "title": str(it.get("title", "")),
                        "description": str(it.get("description", "")),
                        "source": str(it.get("source", "")),
                        "kpi": list(it.get("kpi", [])) if isinstance(it.get("kpi"), list) else [],
                    })
            return {"extracted_goals": goals, "error": None}
        except Exception as e:
            return {"error": str(e), "extracted_goals": []}

    def node_split_by_subdivision(state: CascadeGraphState) -> dict[str, Any]:
        extracted = state.get("extracted_goals") or []
        if not extracted:
            return {"subdivision_goals": [], "error": state.get("error") or "Нет извлечённых целей"}
        context = state.get("context") or {}
        subdivisions = context.get("subdivisions") or []
        subs_str = ", ".join(subdivisions) if subdivisions else "не указаны — предложи типовые"
        goals_json = json.dumps(extracted, ensure_ascii=False, indent=2)
        prompt = CASCADE_SPLIT_BY_SUBDIVISION.format(
            goals_json=goals_json,
            subdivisions=subs_str,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            content = invoke_llm(messages)
            if not content:
                return {"error": "Пустой ответ LLM при разбивке по подразделениям", "subdivision_goals": []}
            items = _parse_json_array(content)
            sub_goals = []
            for it in items:
                if isinstance(it, dict) and it.get("subdivision") and isinstance(it.get("goal"), dict):
                    g = it["goal"]
                    sub_goals.append({
                        "subdivision": str(it["subdivision"]),
                        "goal": {
                            "title": str(g.get("title", "")),
                            "description": str(g.get("description", "")),
                            "source": str(g.get("source", "")),
                            "kpi": list(g.get("kpi", [])) if isinstance(g.get("kpi"), list) else [],
                        },
                        "checklist_ok": True,
                        "comment": str(it.get("comment", "")),
                    })
            return {"subdivision_goals": sub_goals, "error": None}
        except Exception as e:
            return {"error": str(e), "subdivision_goals": []}

    def node_check_checklist(state: CascadeGraphState) -> dict[str, Any]:
        sub_goals = state.get("subdivision_goals") or []
        checklists_text = state.get("checklists_text") or ""
        if not sub_goals:
            return {}
        if not checklists_text.strip():
            return {"raw_output": json.dumps(sub_goals, ensure_ascii=False, indent=2)}
        goals_json = json.dumps(sub_goals, ensure_ascii=False, indent=2)
        prompt = CASCADE_CHECK_CHECKLIST.format(
            goals_json=goals_json,
            checklist_text=checklists_text,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            content = invoke_llm(messages)
            if content:
                items = _parse_json_array(content)
                if items:
                    sub_goals = []
                    for it in items:
                        if isinstance(it, dict):
                            g = it.get("goal") or {}
                            sub_goals.append({
                                "subdivision": str(it.get("subdivision", "")),
                                "goal": g if isinstance(g, dict) else {"title": "", "description": "", "source": "", "kpi": []},
                                "checklist_ok": bool(it.get("checklist_ok", True)),
                                "comment": str(it.get("comment", "")),
                            })
                    return {"subdivision_goals": sub_goals, "raw_output": content}
            return {"raw_output": content or ""}
        except Exception as e:
            return {"error": str(e), "raw_output": state.get("raw_output")}

    builder: StateGraph = StateGraph(CascadeGraphState)
    builder.add_node("extract_goals", node_extract_goals)
    builder.add_node("split_by_subdivision", node_split_by_subdivision)
    builder.add_node("check_checklist", node_check_checklist)

    builder.add_edge(START, "extract_goals")
    builder.add_edge("extract_goals", "split_by_subdivision")
    builder.add_edge("split_by_subdivision", "check_checklist")
    builder.add_edge("check_checklist", END)

    return builder.compile()
