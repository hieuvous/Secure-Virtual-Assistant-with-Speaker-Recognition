from __future__ import annotations

import numpy as np

from src.config import load_thresholds
from src.speaker.embedding import extract_embedding
from src.speaker.scoring import cosine_score


def identify_speaker(
    audio_path: str,
    profiles: list[dict],
    threshold: float | None = None,
) -> dict:
    """
    profiles: [{"user_id": 1, "name": "Hieu", "embedding_path": "..."}]
    """
    if threshold is None:
        threshold = float(load_thresholds()["sid_threshold"])

    if not profiles:
        return {
            "user_id": None,
            "name": None,
            "score": float("-inf"),
            "threshold": float(threshold),
            "is_unknown": True,
        }

    query = extract_embedding(audio_path)
    best = None
    for profile in profiles:
        ref = np.load(profile["embedding_path"])
        score = cosine_score(query, ref)
        row = {**profile, "score": score}
        if best is None or score > best["score"]:
            best = row

    is_unknown = best["score"] < threshold
    return {
        "user_id": None if is_unknown else int(best["user_id"]),
        "name": None if is_unknown else best.get("name"),
        "score": float(best["score"]),
        "threshold": float(threshold),
        "is_unknown": bool(is_unknown),
    }
