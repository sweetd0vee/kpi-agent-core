# kpi-agent-core

Ядро приложения **AI KPI**: модели, промпты и граф каскадирования целей (LangGraph). Основная бизнес-логика, которую использует бэкенд (`kpi-agent-backend`) для API чата и каскада.

## Содержимое

- **models** — `CascadeState`, `GoalItem`, `SubdivisionGoal` (Pydantic) для типизации входа/выхода графа и API.
- **prompts** — промпты для шагов каскада (извлечение целей, разбивка по подразделениям, проверка по чеклистам) и системный промпт чата.
- **graph** — `build_cascade_graph(invoke_llm)` — собирает LangGraph из трёх узлов: извлечение целей → разбивка по подразделениям → проверка по чеклистам.
- **embeddings** — первый этап пайплайна: готовый текст (например пронумерованные чеклисты) оборачивается в шаблон `EMBED_DOCUMENT_TEMPLATE`, вызов Ollama (`qwen3-embedding`), сохранение результата.

Граф не вызывает LLM сам: принимает функцию `invoke_llm(messages) -> content`, которую передаёт бэкенд (Open Web UI / OpenAI). Эмбеддинги получаются через Ollama API `/api/embed`.

## Установка

Рекомендуется **обычная установка** (без `-e`), т.к. `pip install -e .` на старом pip под Windows часто падает (deprecated `develop`, "No module named pip" в build env):

```bash
cd kpi-agent-core
pip install .
```

Из корня репозитория: `pip install ./kpi-agent-core`

Зависимости: `pydantic`, `langgraph`, `langchain-core`, `httpx`. После правок в коде ядра переустановите: `pip install .`

## Использование из бэкенда

```python
from kpi_agent_core import build_cascade_graph
from src.services.llm import chat_completion

def invoke_llm(messages):
    return chat_completion(messages) or ""

graph = build_cascade_graph(invoke_llm)
result = graph.invoke({
    "goals_text": "Текст документа с целями...",
    "checklists_text": "Текст чеклистов...",
    "context": {"subdivisions": ["Кредитование", "Риски"]},
})
# result["subdivision_goals"], result["raw_output"], result.get("error")
```

## Эмбеддинги (Ollama qwen3-embedding)

Вход — готовый текст (например пронумерованный чеклист). Создание эмбеддинга и сохранение:

```python
from kpi_agent_core import embed_document

# text — готовый текст чеклиста (из .txt или строки)
result = embed_document(
    text=checklist_text,
    document_type="business_plan_checklist",
    document_id="eb241170-69f7-40a8-acaa-ab7d42892aea",
    model="qwen3-embedding",
    base_url="http://localhost:11434",
    save_path="output/embed_business_plan.json",
)
# result["embedding"] — список float, результат сохранён в output/embed_business_plan.json
```

Перед вызовом: `ollama pull qwen3-embedding` и запущенный Ollama.

### Как протестировать создание эмбеддинга

1. Запустите Ollama и загрузите модель:
   ```bash
   ollama pull qwen3-embedding
   ```

2. Установите ядро (один раз): `pip install .` из папки `kpi-agent-core`. Не используйте `pip install -e .` — на старом pip под Windows это может падать.

3. Запуск скрипта (из папки `kpi-agent-core`):
   ```bash
   cd kpi-agent-core
   python scripts/run_embed.py "путь\к\документу.txt"
   ```
   Результат по умолчанию: `out/embed_result.json`.

4. Свой выход и тип документа:
   ```bash
   python scripts/run_embed.py path/to/checklist.txt -o out/my_embed.json
   python scripts/run_embed.py path/to/doc.txt --type strategy_checklist -o out/strategy.json
   ```

## Запуск тестов

```bash
cd kpi-agent-core
pip install -e ".[dev]"
pytest
```
