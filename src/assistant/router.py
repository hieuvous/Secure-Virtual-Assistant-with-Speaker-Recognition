from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)


def detect_intent(text: str) -> dict:
    t = _norm(text)

    if any(k in t for k in ["may gio", "bay gio la may gio"]):
        intent = "GET_TIME"
    elif any(k in t for k in ["ngay may", "hom nay ngay", "hom nay la ngay"]):
        intent = "GET_DATE"
    elif "phong" in t and any(k in t for k in ["hoc", "mon"]):
        intent = "GET_COURSE_ROOM"
    elif any(k in t for k in ["lich hoc", "hom nay toi hoc", "toi hoc mon gi"]):
        intent = "GET_SCHEDULE"
    elif any(k in t for k in ["deadline", "con task", "con viec", "nhiem vu"]):
        if any(k in t for k in ["xoa", "huy"]):
            intent = "DELETE_TASK"
        elif any(k in t for k in ["them", "tao"]):
            intent = "ADD_TASK"
        else:
            intent = "GET_TASKS"
    elif any(k in t for k in ["ghi chu rieng", "private note", "ghi chu cua toi"]):
        intent = "READ_PRIVATE_NOTE"
    elif "doi" in t and any(k in t for k in ["lich", "gio hoc"]):
        intent = "UPDATE_SCHEDULE"
    else:
        intent = "UNKNOWN"

    # Starter entity extraction: preserve original text and add simple quoted-span extraction.
    quoted = re.findall(r'["“](.+?)["”]', text)
    entities = {"raw_text": text}
    if quoted:
        entities["quoted_text"] = quoted[0]

    return {
        "intent": intent,
        "entities": entities,
        "confidence": 1.0 if intent != "UNKNOWN" else 0.0,
    }
