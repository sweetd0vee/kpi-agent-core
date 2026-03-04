"""
Модели ядра: состояние каскада, цели, подразделения.

Типизация для графа LangGraph и ответов API.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class GoalItem(BaseModel):
    """Одна цель (из документа руководства или сгенерированная)."""
    id: Optional[str] = None
    title: str
    description: str = ""
    source: str = ""  # откуда извлечена (документ, раздел)
    kpi: list[str] = Field(default_factory=list)  # связанные KPI


class SubdivisionGoal(BaseModel):
    """Цель, привязанная к подразделению (результат каскада)."""
    subdivision: str
    goal: GoalItem
    checklist_ok: bool = True  # пройдена ли проверка по чеклисту
    comment: str = ""


class CascadeState(BaseModel):
    """
    Состояние графа каскадирования.

    Заполняется по шагам: извлечение целей → разбивка по подразделениям → проверка по чеклистам.
    """
    # Вход (заполняет бэкенд перед вызовом графа)
    goals_text: str = ""           # текст документа с целями руководства
    checklists_text: str = ""      # объединённый текст чеклистов (стратегия, регламент и т.д.)
    context: dict[str, Any] = Field(default_factory=dict)

    # Промежуточные результаты
    extracted_goals: list[GoalItem] = Field(default_factory=list)
    subdivision_goals: list[SubdivisionGoal] = Field(default_factory=list)

    # Выход для API
    raw_output: Optional[str] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
