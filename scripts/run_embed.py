"""
Скрипт для проверки создания эмбеддинга по одному документу.

Запуск (из корня проекта ai-kpi или из kpi-agent-core):

  # эмбеддинг для файла из uploads бэкенда (подставьте свой путь при необходимости)
  python -m kpi_agent_core.scripts.run_embed

  # или указать файл и куда сохранить
  python -m kpi_agent_core.scripts.run_embed path/to/document.txt --out output/embed.json

Требуется: Ollama запущен, модель загружена: ollama pull qwen3-embedding
"""
import argparse
import sys
from pathlib import Path

# Добавить src в путь, если скрипт вызывают из корня репозитория
_core_src = Path(__file__).resolve().parent.parent / "src"
if _core_src.exists() and str(_core_src) not in sys.path:
    sys.path.insert(0, str(_core_src))

from kpi_agent_core import embed_document, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_BASE_URL


def main():
    parser = argparse.ArgumentParser(description="Создать эмбеддинг для одного текстового документа")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Путь к .txt документу (по умолчанию — пример из uploads бэкенда)",
    )
    parser.add_argument(
        "--out", "-o",
        default="embed_result.json",
        help="Файл для сохранения результата (по умолчанию embed_result.json)",
    )
    parser.add_argument(
        "--type",
        default="business_plan_checklist",
        help="Тип документа (business_plan_checklist, strategy_checklist и т.д.)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Модель Ollama (по умолчанию {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"URL Ollama (по умолчанию {DEFAULT_OLLAMA_BASE_URL})",
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        # Путь к примеру в uploads бэкенда (относительно корня ai-kpi)
        root = Path(__file__).resolve().parent.parent.parent  # ai-kpi
        path = root / "kpi-agent-backend" / "uploads" / "business_plan_checklist" / "eb241170-69f7-40a8-acaa-ab7d42892aea_checklist_business_plan.txt"
        if not path.exists():
            print("Файл по умолчанию не найден:", path)
            print("Укажите путь к .txt документу: python -m kpi_agent_core.scripts.run_embed path/to/doc.txt")
            sys.exit(1)

    if not path.exists():
        print("Файл не найден:", path)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    print(f"Документ: {path.name}, символов: {len(text)}")
    print(f"Модель: {args.model}, URL: {args.base_url}")
    print("Создаём эмбеддинг...")

    try:
        result = embed_document(
            text=text,
            document_type=args.type,
            document_id=path.stem,
            model=args.model,
            base_url=args.base_url,
            save_path=args.out,
        )
    except Exception as e:
        print("Ошибка:", e)
        sys.exit(1)

    dim = len(result["embedding"])
    print(f"Готово. Размерность вектора: {dim}")
    print(f"Результат сохранён в: {args.out}")


if __name__ == "__main__":
    main()
