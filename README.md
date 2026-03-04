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

## Пайплайн: эмбеддинги → каскад целей

**Идея:** Эмбеддинги нужны, чтобы по запросу (текст целей) подобрать релевантные ключевые документы (чеклисты). В LLM передаётся **текст** этих документов вместе с промптом каскада — не векторы. Модель получает цели + чеклисты и выдаёт разбивку по подразделениям.

### Шаг 1. Сформировать эмбеддинги для ключевых документов

Для каждого чеклиста (бизнес-план, стратегия, регламент и т.д.) создайте эмбеддинг и сохраните в `out/`. В JSON сохраняется путь к исходному файлу (`source_path`), чтобы при поиске по релевантности подставлять полный текст.

```bash
python scripts/run_embed.py raw/checklist_business_plan.txt -n business_plan_embed.json
python scripts/run_embed.py raw/checklist_strategy.txt -n strategy_embed.json --type strategy_checklist
```

### Шаг 2. Запуск каскада с подготовленным контекстом

**Вариант A — явно передать тексты (как сейчас в API):**  
Бэкенд получает `goal_file_ids` и `checklist_file_ids`, загружает текст документов и передаёт в граф: `goals_text` + `checklists_text` → граф → цели по подразделениям.

**Вариант B — подобрать чеклисты по релевантности к целям:**  
По тексту целей делается поиск по эмбеддингам, выбираются топ‑k чеклистов, их полный текст подставляется в `checklists_text`:

```python
from kpi_agent_core import (
    load_embedding_index,
    retrieve_relevant_documents,
    retrieved_texts_to_checklists_string,
    build_cascade_graph,
)

# Индекс из папки out/
index = load_embedding_index("out/")
# По тексту целей подобрать релевантные чеклисты (полный текст берётся из source_path в JSON)
docs = retrieve_relevant_documents(goals_text, index, top_k=3)
checklists_text = retrieved_texts_to_checklists_string(docs)

graph = build_cascade_graph(invoke_llm)
result = graph.invoke({
    "goals_text": goals_text,
    "checklists_text": checklists_text,
    "context": {"subdivisions": ["Кредитование", "Риски"]},
})
```

Итог: ключевые документы представлены эмбеддингами; при каскаде по запросу выбираются нужные чеклисты, в LLM уходит их **текст** + промпт каскада → структурированные цели по подразделениям.

**Важно:** в LLM не передают JSON с векторами. Модель работает только с текстом. Эмбеддинги нужны лишь для выбора релевантных документов; в промпт подставляется полный текст этих документов (из `text_full` или файла по `source_path`). При желании перед текстом можно добавить подпись из `RETRIEVED_DOCUMENTS_HEADER`.

## Каскад целей главного руководителя → подчинённые (таблица)

Задача: по целям главного руководителя и приложенным документам (Бизнес-план, Стратегия, Регламент, Положение о департаменте, известные цели подчинённых) получить таблицу каскадированных целей: для каждого подчинённого — релевантные цели руководителя и сформулированные цели подчинённого.

### Предобработка документов

1. **Формат:** каждый документ — текст (например .txt или извлечённый из PDF/DOCX). Желательно структурированный (разделы, пункты), без лишней разметки.
2. **Что подготовить:**
   - Список целей главного руководителя (строка или список dict с полями `title`, `description`, `kpi`).
   - Тексты: Бизнес-план, Стратегия, Регламент, Положение о департаменте, известные цели подчинённых (если есть).
3. **Объём:** в промпт подставляется до ~40 000 символов на документ (настраивается). При очень больших файлах можно предварительно извлечь релевантные фрагменты по эмбеддингам (retrieval) и передать только их.

### Что передать локальной LLM

В модель уходит **два сообщения**:

- **system:** описание задачи (каскадировать цели, опираться на документы, определить релевантность, выдать таблицу). Текст берётся из `CASCADE_LEADER_TO_TABLE_SYSTEM`.
- **user:** структурированный блок: цели руководителя + разделы «Бизнес-план», «Стратегия», «Регламент», «Положение о департаменте», «Известные цели подчинённых» + список подчинённых. Шаблон — `CASCADE_LEADER_TO_TABLE_USER`.

Модель возвращает таблицу (markdown) с колонками: Подчинённый/Департамент | Цель главного руководителя | Каскадированная цель | Источник | KPI.

### Пример вызова

```python
from kpi_agent_core import (
    cascade_leader_goals_to_table,
    load_documents_from_paths,
    format_leader_goals,
)
from src.services.llm import chat_completion

def invoke_llm(messages):
    return chat_completion(messages, model="llama3.2", temperature=0.2) or ""

# Загрузить документы из файлов (или передать строки в build_documents_dict)
documents = load_documents_from_paths(
    business_plan_path="raw/business_plan.txt",
    strategy_path="raw/strategy.txt",
    regulation_path="raw/regulation.txt",
    department_regulation_path="raw/department_regulation.txt",
    known_goals_path="raw/known_goals.txt",
)

leader_goals = [
    {"title": "Рост выручки", "description": "...", "kpi": ["Выручка +10%"]},
    {"title": "Снижение рисков", "description": "...", "kpi": ["NPL < 5%"]},
]
subordinates = ["Директор по кредитам", "Директор по рискам", "Директор IT"]

table_markdown = cascade_leader_goals_to_table(
    leader_goals,
    documents,
    subordinates,
    invoke_llm,
)
# table_markdown — готовая таблица в markdown для сохранения или отображения
```

Документы можно собрать и из строк (например, после retrieval): `build_documents_dict(business_plan=text1, strategy=text2, ...)`.

### Мало документов — без эмбеддингов (рекомендуется по умолчанию)

Когда документов мало (например 5 фиксированных: бизнес-план, стратегия, регламент и т.д.), эмбеддинги не обязательны. Используйте **`cascade_leader_goals_to_table_simple`**: каждый документ обрезается по длине (по секциям с начала) до `max_chars_per_doc`, затем вызывается тот же каскад в таблицу. Никаких вызовов Ollama для эмбеддингов.

```python
from kpi_agent_core import cascade_leader_goals_to_table_simple, load_documents_from_paths

documents = load_documents_from_paths(
    business_plan_path="raw/business_plan.txt",
    strategy_path="raw/strategy.txt",
    regulation_path="raw/regulation.txt",
    department_regulation_path="raw/department_regulation.txt",
    known_goals_path="raw/known_goals.txt",
)

table_markdown = cascade_leader_goals_to_table_simple(
    leader_goals,
    documents,
    subordinates,
    invoke_llm,
    max_chars_per_doc=6000,
)
```

### Интеллектуальная предобработка больших документов (опционально)

Имеет смысл, когда документы очень длинные и нужно отобрать по смыслу релевантные фрагменты. При 5 документах умеренного размера достаточно `cascade_leader_goals_to_table_simple` (см. выше). Если же нужно «выжать» релевантные куски из огромных файлов:

1. **Разбиение на чанки** — документ режется на фрагменты (~1800 символов с перекрытием 200) или по секциям (заголовки ##, нумерованные списки).
2. **Отбор по релевантности** — цели руководителя и каждый чанк превращаются в эмбеддинги (Ollama); для каждого типа документа отбираются топ‑k чанков, наиболее близких по смыслу к целям.
3. **Компактный контекст** — в промпт попадает только отобранный текст (по умолчанию до 6000 символов на документ). Модель получает релевантные фрагменты, а не весь объём.

Использование: вместо `cascade_leader_goals_to_table` вызвать **`cascade_leader_goals_to_table_with_preprocess`** и передать дополнительно `get_embedding` (например `get_embedding_ollama` из ядра):

```python
from kpi_agent_core import (
    cascade_leader_goals_to_table_with_preprocess,
    load_documents_from_paths,
    get_embedding_ollama,
)
from src.services.llm import chat_completion

def invoke_llm(messages):
    return chat_completion(messages, model="llama3.2", temperature=0.2) or ""

documents = load_documents_from_paths(
    business_plan_path="raw/business_plan.txt",
    strategy_path="raw/strategy.txt",
    regulation_path="raw/regulation.txt",
    department_regulation_path="raw/department_regulation.txt",
    known_goals_path="raw/known_goals.txt",
)

table_markdown = cascade_leader_goals_to_table_with_preprocess(
    leader_goals,
    documents,
    subordinates,
    invoke_llm,
    get_embedding=get_embedding_ollama,
    top_k_per_doc=5,
    max_chars_per_doc=6000,
)
```

Опционально можно заранее построить индекс чанков (`build_chunk_index`), сохранить его (`save_chunk_index`) и при следующем запуске загружать (`load_chunk_index`), чтобы не пересчитывать эмбеддинги чанков при каждом каскаде.

## Запуск тестов

```bash
cd kpi-agent-core
pip install -e ".[dev]"
pytest
```
