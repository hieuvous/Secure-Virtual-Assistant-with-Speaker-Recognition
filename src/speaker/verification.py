from __future__ import annotations

import numpy as np

from src.config import load_thresholds
from src.speaker.embedding import extract_embedding
from src.speaker.scoring import cosine_score


def verify_speaker(
    audio_path: str,
    claimed_user_id: int,
    embedding_path: str,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = float(load_thresholds()["sv_threshold"])

    query = extract_embedding(audio_path)
    profile = np.load(embedding_path)
    score = cosine_score(query, profile)

    return {
        "user_id": int(claimed_user_id),
        "accepted": bool(score >= threshold),
        "score": score,
        "threshold": float(threshold),
    }
