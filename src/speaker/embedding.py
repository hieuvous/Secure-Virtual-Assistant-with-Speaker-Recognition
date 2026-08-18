from __future__ import annotations

from pathlib import Path
import numpy as np

from src.speech.preprocessing import load_mono_16k
from src.speaker.model import get_ecapa


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero embedding.")
    return x / n


def extract_embedding(audio_path: str | Path) -> np.ndarray:
    wav = load_mono_16k(audio_path)
    emb = get_ecapa().encode_waveform(wav).numpy()
    return l2_normalize(emb)
