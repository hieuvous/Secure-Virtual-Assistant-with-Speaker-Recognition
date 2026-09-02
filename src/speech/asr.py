from __future__ import annotations

from threading import Lock
from faster_whisper import WhisperModel

from src.config import load_settings


class FasterWhisperService:
    def __init__(self):
        cfg = load_settings()["asr"]
        self.model_size = cfg["model_size"]
        self.language = cfg.get("language", "vi")
        self.beam_size = int(cfg.get("beam_size", 5))
        self.model = WhisperModel(
            self.model_size,
            device=cfg.get("device", "cpu"),
            compute_type=cfg.get("compute_type", "int8"),
        )

    def transcribe(self, audio_path: str) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", self.language),
            "language_probability": float(
                getattr(info, "language_probability", 0.0) or 0.0
            ),
        }


_asr_instance: FasterWhisperService | None = None
_asr_initialization_lock = Lock()


def get_asr() -> FasterWhisperService:
    """Create the ASR model once, including during overlapping Streamlit reruns.

    ``lru_cache`` does not serialize concurrent cache misses.  On a cold model
    cache that could start two Hugging Face downloads, which races tqdm's
    process-global progress-bar lock.
    """
    global _asr_instance
    if _asr_instance is None:
        with _asr_initialization_lock:
            if _asr_instance is None:
                _asr_instance = FasterWhisperService()
    return _asr_instance
