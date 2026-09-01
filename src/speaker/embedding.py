from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from src.config import load_settings
from src.speech.preprocessing import load_mono_16k
from src.speech.vad import get_vad
from src.speaker.model import get_ecapa


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero embedding.")
    return x / n


def embedding_to_numpy(value, dimension: int = 192) -> np.ndarray:
    """Convert a pgvector response (list or '[...]' string) to a 1-D array."""
    if isinstance(value, str):
        value = json.loads(value)
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size != dimension:
        raise ValueError(
            f"Expected a {dimension}-D speaker embedding, got {embedding.size} values."
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("Speaker embedding contains non-finite values.")
    return embedding


def embedding_to_pgvector(value, dimension: int = 192) -> str:
    """Serialize an embedding in pgvector's text input format."""
    embedding = embedding_to_numpy(value, dimension)
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def extract_embedding(audio_path: str | Path, use_vad: bool | None = None) -> np.ndarray:
    cfg = load_settings()["speaker"]
    if use_vad is None:
        use_vad = bool(cfg.get("use_vad", True))

    wav = get_vad().trim(audio_path) if use_vad else load_mono_16k(audio_path)
    emb = get_ecapa().encode_waveform(wav).numpy()
    return l2_normalize(emb)
