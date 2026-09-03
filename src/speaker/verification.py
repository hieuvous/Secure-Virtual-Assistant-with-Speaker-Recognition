import numpy as np

from src.config import load_thresholds
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
    """Convert a Supabase pgvector/list/array profile to a NumPy vector."""
    if isinstance(reference_embedding, str) and not reference_embedding.lstrip().startswith("["):
        raise RuntimeError(
            "Local speaker embedding paths are no longer supported. "
            "Fetch speaker_profiles.embedding from Supabase."
        )
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
