"""
Преобразование загруженных файлов (txt, docx) в JSON-формат для чеклистов.

Пронумерованные чеклисты разбираются в структуру: заголовок документа,
разделы с пунктами [ ]. Локальная LLM получает JSON вместо сырого текста —
удобнее для обработки и парсинга ответов.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Разделитель секций: строка из длинных тире или знаков равенства
SECTION_SEP_RE = re.compile(r"^[=\s\-—–−]{20,}\s*$")
# Заголовок секции: "1. НАЗВАНИЕ" или "1) НАЗВАНИЕ"
SECTION_HEADER_RE = re.compile(r"^(\d+)[.)]\s*(.+)$")
# Пункт чеклиста: "[ ] текст" или "[x] текст"
CHECKLIST_ITEM_RE = re.compile(r"^\s*\[\s*[ xX✓]\s*]\s*(.*)$")


def _normalize_line(line: str) -> str:
    return line.replace("\r", "").strip()


def checklist_text_to_json(text: str) -> Dict[str, Any]:
    """
    Разобрать текст пронумерованного чеклиста в JSON.

    Ожидаемый формат:
    - Заголовок документа в начале (до первого разделителя или "1. Раздел").
    - Разделы: строка-разделитель (———), затем строка "N. Название раздела", затем пункты [ ].

    Возвращает структуру:
    {
      "title": "ЧЕКЛИСТ ПО ...",
      "sections": [
        { "number": 1, "title": "...", "items": ["пункт 1", "пункт 2"] },
        ...
      ]
    }
    """
    lines = [_normalize_line(ln) for ln in text.splitlines()]
    # Разбить на блоки по разделителям
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if SECTION_SEP_RE.match(line):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    title = ""
    sections: List[Dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        block = [ln for ln in blocks[i] if ln]
        if not block:
            i += 1
            continue
        first = block[0]
        head_match = SECTION_HEADER_RE.match(first)
        if head_match:
            num_str, sect_title = head_match.groups()
            # Пункты могут быть в этом же блоке (block[1:]) или в следующем (после разделителя)
            items: List[str] = []
            for line in block[1:]:
                item_match = CHECKLIST_ITEM_RE.match(line)
                if item_match:
                    items.append(item_match.group(1).strip())
                elif line.endswith(":") or not line.startswith("["):
                    if items and not line.startswith("["):
                        items[-1] = items[-1] + " " + line
            # Если в блоке только заголовок — пункты в следующем блоке
            if not items and i + 1 < len(blocks):
                next_block = [ln for ln in blocks[i + 1] if ln]
                for line in next_block:
                    item_match = CHECKLIST_ITEM_RE.match(line)
                    if item_match:
                        items.append(item_match.group(1).strip())
                    elif items and not line.startswith("["):
                        items[-1] = items[-1] + " " + line
                i += 1  # пропустить блок с пунктами
            sections.append({
                "number": int(num_str),
                "title": sect_title.strip(),
                "items": items,
            })
            i += 1
            continue
        # Блок не начинается с "N. " — либо заголовок документа, либо мусор
        if i == 0 and not sections:
            title = " ".join(block).strip() or "Чеклист"
        elif block and CHECKLIST_ITEM_RE.match(block[0]):
            items = []
            for line in block:
                item_match = CHECKLIST_ITEM_RE.match(line)
                if item_match:
                    items.append(item_match.group(1).strip())
            if items:
                sections.append({"title": "Пункты", "items": items})
        i += 1
    return {"title": title, "sections": sections}


def checklist_json_to_text_for_llm(data: Dict[str, Any], max_section_items: Optional[int] = None) -> str:
    """
    Преобразовать JSON чеклиста обратно в компактный текст для вставки в промпт
    (если нужно передать не сырой JSON, а читаемую строку).
    """
    parts = [data.get("title", "Чеклист")]
    for sec in data.get("sections", []):
        num = sec.get("number", "")
        title = sec.get("title", "")
        parts.append(f"\n{num}. {title}" if num else f"\n{title}")
        items = sec.get("items", [])
        if max_section_items:
            items = items[:max_section_items]
        for it in items:
            parts.append(f"  [ ] {it}")
    return "\n".join(parts).strip()


def file_to_checklist_json(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Загрузить файл (txt или docx) и преобразовать в JSON чеклиста.

    - .txt: читается как текст в заданной кодировке.
    - .docx: извлекается текст параграфов (требуется python-docx).

    Возвращает структуру от checklist_text_to_json; если файл не txt/docx или пустой —
    возвращает {"title": "", "sections": []} с полем "source_path".
    """
    path = Path(path)
    if not path.exists():
        return {"title": "", "sections": [], "source_path": str(path), "error": "file_not_found"}
    suffix = path.suffix.lower()
    text: Optional[str] = None
    if suffix == ".txt":
        try:
            text = path.read_text(encoding=encoding)
        except Exception as e:
            return {"title": "", "sections": [], "source_path": str(path), "error": str(e)}
    elif suffix in (".docx", ".doc"):
        text = _extract_text_from_docx(path)
        if text is None:
            return {"title": "", "sections": [], "source_path": str(path), "error": "docx_read_failed"}
    else:
        return {"title": "", "sections": [], "source_path": str(path), "error": "unsupported_extension"}
    if not (text or text.strip()):
        return {"title": "", "sections": [], "source_path": str(path)}
    out = checklist_text_to_json(text)
    out["source_path"] = str(path)
    return out


def _extract_text_from_docx(path: Path) -> Optional[str]:
    """Извлечь текст из .docx через python-docx."""
    try:
        from docx import Document
    except ImportError:
        return None
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return None


def save_checklist_json(data: Dict[str, Any], out_path: Union[str, Path], indent: int = 2) -> Path:
    """Сохранить JSON чеклиста в файл."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in data.items() if k != "error"}  # не сохраняем error в файл
    out_path.write_text(json.dumps(to_save, ensure_ascii=False, indent=indent), encoding="utf-8")
    return out_path


def documents_dict_to_json_strings(
    documents: Dict[str, str],
    keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Преобразовать словарь документов (ключ → сырой текст) в словарь ключ → JSON-строка.
    Каждый текст разбирается как чеклист (checklist_text_to_json); результат сериализуется в строку.
    Удобно подставлять в промпт: модель получает JSON вместо сырого текста.
    """
    keys = keys or list(documents.keys())
    out = {}
    for k in keys:
        raw = documents.get(k) or ""
        if not raw.strip():
            out[k] = json.dumps({"title": "", "sections": []}, ensure_ascii=False)
            continue
        data = checklist_text_to_json(raw)
        out[k] = json.dumps(data, ensure_ascii=False, indent=0)
    return out
