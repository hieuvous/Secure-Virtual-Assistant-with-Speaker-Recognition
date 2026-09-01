from __future__ import annotations

import re
import unicodedata


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)

def _extract_task_title(text: str) -> str | None:
    # Nếu text có ngoặc kép thì ưu tiên title trong ngoặc
    quoted = _quoted(text)
    if quoted:
        return quoted

    # Cho phép:
    # Thêm deadline báo cáo NLP
    # Thêm deadline, báo cáo NLP
    # Xóa task: học tiếng Anh
    match = re.search(
        r"(?:thêm|them|tạo|tao|xóa|xoa|hủy|huy)"
        r"\s+(?:deadline|task|nhiệm\s*vụ|nhiem\s+vu)"
        r"\s*[,;:\-]?\s*(.+?)\s*[.!?]*$",
        text.strip(),
        flags=re.IGNORECASE,
    )

    if match:
        title = match.group(1).strip()
        return title if title else None

    return None

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

    if any(k in t for k in ["may gio", "bay gio la", "gio hien tai", "xem gio", "thoi gian hien tai"]):
        intent = "GET_TIME"

    elif any(k in t for k in ["ngay may", "hom nay ngay", "hom nay la ngay", "thu may", "ngay bao nhieu", "hom nay la thu"]):
        intent = "GET_DATE"

    elif any(k in t for k in ["phong hoc", "hoc o phong", "hoc o dau", "phong nao", "dia diem hoc"]) or ("phong" in t and "mon" in t):
        intent = "GET_COURSE_ROOM"
        if not quoted:
            # Lightweight extraction: text after "môn"/"mon" and before "học"/"hoc" if possible.
            m = re.search(r"(?:môn|mon)\s+(.+?)(?:\s+học|\s+hoc|\s+ở|\s+o|\?|$)", text, re.I)
            if m:
                entities["subject"] = m.group(1).strip(" ?.")

    elif any(k in t for k in ["lich hoc", "hom nay toi hoc", "toi hoc mon gi", "thoi khoa bieu", "ngay mai hoc gi", "co mon nao"]):
        intent = "GET_SCHEDULE"

    elif any(k in t for k in ["ghi chu rieng", "private note", "ghi chu cua toi", "mo ghi chu", "xem ghi chu", "doc ghi chu"]):
        intent = "READ_PRIVATE_NOTE"

    # elif any(k in t for k in ["deadline", "con task", "con viec", "nhiem vu"]):
    #     if any(k in t for k in ["xoa", "huy"]):
    #         intent = "DELETE_TASK"

    #         if not quoted:
    #             m = re.search(
    #                 r"(?:xoa|huy)\s+(?:deadline|task|nhiem vu)\s+(.+)",
    #                 t
    #             )
    #             if m:
    #                 entities["title"] = m.group(1).strip()

    #     elif any(k in t for k in ["them", "tao"]):
    #         intent = "ADD_TASK"

    #         if not quoted:
    #             m = re.search(
    #                 r"(?:them|tao)\s+(?:deadline|task|nhiem vu)\s+(.+)",
    #                 t
    #             )
    #             if m:
    #                 entities["title"] = m.group(1).strip()

    #     else:
    #         intent = "GET_TASKS"

    elif any(k in t for k in ["deadline", "con task", "con viec", "nhiem vu"]):
        if any(k in t for k in ["xoa", "huy"]):
            intent = "DELETE_TASK"

            title = _extract_task_title(text)
            if title:
                entities["title"] = title

        elif any(k in t for k in ["them", "tao"]):
            intent = "ADD_TASK"

            title = _extract_task_title(text)
            if title:
                entities["title"] = title

        else:
            intent = "GET_TASKS"
    else:
        intent = "UNKNOWN"

    return {
        "intent": intent,
        "entities": entities,
        "confidence": 1.0 if intent != "UNKNOWN" else 0.0,
    }
