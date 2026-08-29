from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)


def _quoted(text: str) -> str | None:
    match = re.search(r'["“](.+?)["”]', text)
    return match.group(1).strip() if match else None


def detect_intent(text: str) -> dict:
    """
    Rule-based NLU kept intentionally small for the final demo.

    For destructive task commands, quote the title for deterministic extraction:
      Thêm deadline "Báo cáo NLP"
      Xóa deadline "Báo cáo NLP"
    """
    t = _norm(text)
    quoted = _quoted(text)
    entities = {"raw_text": text}
    if quoted:
        entities["title"] = quoted
        entities["subject"] = quoted

    if any(k in t for k in ["may gio", "bay gio la", "gio hien tai"]):
        intent = "GET_TIME"

    elif any(k in t for k in ["ngay may", "hom nay ngay", "hom nay la ngay"]):
        intent = "GET_DATE"

    elif "phong" in t and any(k in t for k in ["hoc", "mon"]):
        intent = "GET_COURSE_ROOM"
        if not quoted:
            # Lightweight extraction: text after "môn"/"mon" and before "học"/"hoc" if possible.
            m = re.search(r"(?:môn|mon)\s+(.+?)(?:\s+học|\s+hoc|\s+ở|\s+o|\?|$)", text, re.I)
            if m:
                entities["subject"] = m.group(1).strip(" ?.")

    elif any(k in t for k in ["lich hoc", "hom nay toi hoc", "toi hoc mon gi"]):
        intent = "GET_SCHEDULE"

    elif any(k in t for k in ["ghi chu rieng", "private note", "ghi chu cua toi"]):
        intent = "READ_PRIVATE_NOTE"

    elif any(k in t for k in ["deadline", "con task", "con viec", "nhiem vu"]):
        if any(k in t for k in ["xoa", "huy"]):
            intent = "DELETE_TASK"
        elif any(k in t for k in ["them", "tao"]):
            intent = "ADD_TASK"
        else:
            intent = "GET_TASKS"

    else:
        intent = "UNKNOWN"

    return {
        "intent": intent,
        "entities": entities,
        "confidence": 1.0 if intent != "UNKNOWN" else 0.0,
    }
