from __future__ import annotations

from pathlib import Path
import numpy as np

from src.config import ROOT
from src.speaker.embedding import extract_embedding, l2_normalize


def create_speaker_profile(
    user_id: int,
    audio_paths: list[str],
    sentence_ids: list[int] | None = None,
) -> dict:
    if not audio_paths:
        raise ValueError("At least one enrollment recording is required.")

    embeddings = [extract_embedding(p) for p in audio_paths]
    profile = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

    user_dir = ROOT / "data" / "users" / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = user_dir / "speaker_embedding.npy"
    np.save(embedding_path, profile)

    return {
        "user_id": int(user_id),
        "embedding_path": str(embedding_path),
        "num_samples": len(audio_paths),
        "embedding_dim": int(profile.shape[0]),
        "sentence_ids": sentence_ids,
    }
