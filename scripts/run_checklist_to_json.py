"""
Скрипт: преобразование файла чеклиста (txt или docx) в JSON.

Запуск из папки kpi-agent-core:

  python scripts/run_checklist_to_json.py путь/к/файлу.txt
  python scripts/run_checklist_to_json.py путь/к/файлу.docx -o out/checklist.json

Для .docx нужен: pip install ".[docx]"
"""
import argparse
import sys
from pathlib import Path

_core_src = Path(__file__).resolve().parent.parent / "src"
if _core_src.exists() and str(_core_src) not in sys.path:
    sys.path.insert(0, str(_core_src))

from kpi_agent_core.checklist_to_json import file_to_checklist_json, save_checklist_json


def main():
    parser = argparse.ArgumentParser(description="Преобразовать чеклист (txt/docx) в JSON")
    parser.add_argument("file", help="Путь к файлу .txt или .docx")
    parser.add_argument(
        "-o", "--out",
        default=None,
        help="Путь для сохранения JSON (по умолчанию out/<имя_файла>.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Только вывести JSON в stdout, не сохранять в файл",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Ошибка: файл не найден: {path}", file=sys.stderr)
        sys.exit(1)

    data = file_to_checklist_json(path)
    if data.get("error"):
        print(f"Ошибка: {data['error']}", file=sys.stderr)
        sys.exit(1)

    if args.no_save:
        import json
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    out_path = Path(args.out) if args.out else Path("out") / f"{path.stem}.json"
    save_checklist_json(data, out_path)
    print(f"Сохранено: {out_path} (секций: {len(data.get('sections', []))})")


if __name__ == "__main__":
    main()
