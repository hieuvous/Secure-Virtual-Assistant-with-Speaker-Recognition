import numpy as np
from src.speaker.embedding import l2_normalize

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
