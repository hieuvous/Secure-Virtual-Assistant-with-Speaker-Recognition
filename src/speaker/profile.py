from __future__ import annotations

import numpy as np

from src.speaker.embedding import extract_embedding, l2_normalize


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("No enrollment embeddings.")
    normalized = [l2_normalize(embedding) for embedding in embeddings]
    centroid = np.mean(np.stack(normalized, axis=0), axis=0)
    return l2_normalize(centroid)


def create_speaker_profile(
    user_id: int,
    audio_paths: list[str],
    sentence_ids: list[int] | None = None,
) -> dict:
    """Create the normalized centroid that repositories persist in the chosen backend."""
    if not audio_paths:
        raise ValueError("At least one enrollment recording is required.")

    embeddings = [extract_embedding(path, use_vad=True) for path in audio_paths]
    profile = aggregate_embeddings(embeddings)
    return {
        "user_id": int(user_id),
        "embedding": profile,
        "num_samples": len(audio_paths),
        "embedding_dim": int(profile.shape[0]),
        "sentence_ids": sentence_ids,
        "profile_method": "mean_l2_normalized_embedding",
        "vad": True,
    }
