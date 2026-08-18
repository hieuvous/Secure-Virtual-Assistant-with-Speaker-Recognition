import numpy as np
from src.speaker.scoring import cosine_score


def test_cosine_self():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_score(x, x) - 1.0) < 1e-6
