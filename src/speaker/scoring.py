from __future__ import annotations

import numpy as np
from src.speaker.embedding import l2_normalize


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))
