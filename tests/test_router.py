from src.assistant.router import detect_intent
from src.assistant.permissions import get_auth_requirement


def test_get_time():
    assert detect_intent("Bây giờ là mấy giờ?")["intent"] == "GET_TIME"


def test_get_tasks():
    assert detect_intent("Tôi còn deadline nào?")["intent"] == "GET_TASKS"


def test_sensitive_permission():
    assert get_auth_requirement("READ_PRIVATE_NOTE") == "VERIFICATION"
