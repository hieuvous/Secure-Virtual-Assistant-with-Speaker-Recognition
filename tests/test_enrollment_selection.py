from src.speaker.enrollment_selection import random_select


def test_random_selection_is_reproducible():
    candidates = [f"sentence {i}" for i in range(10)]
    a = random_select(candidates, n=5, seed=42)
    b = random_select(candidates, n=5, seed=42)
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert len(a) == 5
