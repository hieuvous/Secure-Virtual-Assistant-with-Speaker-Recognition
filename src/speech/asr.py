from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_asr() -> FasterWhisperService:
    return FasterWhisperService()
