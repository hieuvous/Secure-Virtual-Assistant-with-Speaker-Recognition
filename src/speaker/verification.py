from pathlib import Path

import numpy as np

from src.config import ROOT, load_thresholds
from src.speaker.embedding import embedding_to_numpy, extract_embedding
from src.speaker.scoring import cosine_score


def verify_embedding(query, reference, threshold: float) -> dict:
    score = cosine_score(query, reference)
    return {
        "accepted": bool(score >= threshold),
        "score": float(score),
        "threshold": float(threshold),
    }


def _reference_to_numpy(reference_embedding) -> np.ndarray:
    """Accept a database vector and retain old SQLite path compatibility."""
    if isinstance(reference_embedding, (str, Path)) and not str(reference_embedding).lstrip().startswith("["):
        path = Path(reference_embedding)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Speaker profile not found: {path}")
        return embedding_to_numpy(np.load(path))
    return embedding_to_numpy(reference_embedding)


def verify_speaker(
    audio_path: str,
    claimed_user_id: int,
    reference_embedding,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = float(load_thresholds()["sv_threshold"])

    query = extract_embedding(audio_path, use_vad=True)
    return {
        "user_id": int(claimed_user_id),
        **verify_embedding(query, _reference_to_numpy(reference_embedding), threshold),
    }
