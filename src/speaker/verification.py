from pathlib import Path
import numpy as np

from src.config import load_thresholds
from src.speaker.embedding import extract_embedding
from src.speaker.scoring import cosine_score


def verify_embedding(query, reference, threshold: float) -> dict:
    score = cosine_score(query, reference)
    return {
        "accepted": bool(score >= threshold),
        "score": float(score),
        "threshold": float(threshold),
    }


def verify_speaker(
    audio_path: str,
    claimed_user_id: int,
    embedding_path: str,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = float(load_thresholds()["sv_threshold"])

    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Speaker profile not found: {path}")

    query = extract_embedding(audio_path, use_vad=True)
    reference = np.load(path)
    return {
        "user_id": int(claimed_user_id),
        **verify_embedding(query, reference, threshold),
    }
