from __future__ import annotations

from threading import Lock
import time

from faster_whisper import WhisperModel

from src.config import load_settings


class FasterWhisperService:
    def __init__(self):
        cfg = load_settings()["asr"]

        self.model_size = cfg["model_size"]
        self.language = cfg.get("language", "vi")
        self.beam_size = int(cfg.get("beam_size", 3))

        self.initial_prompt = cfg.get("initial_prompt")
        self.hotwords = cfg.get("hotwords")

        self.model = WhisperModel(
            self.model_size,
            device=cfg.get("device", "cpu"),
            compute_type=cfg.get("compute_type", "int8"),
        )

    def transcribe(self, audio_path: str) -> dict:
        started = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,

            language=self.language,
            beam_size=self.beam_size,

            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
            },

            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,

            # Mỗi command của assistant là một câu độc lập.
            condition_on_previous_text=False,

            temperature=0.0,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        elapsed = time.perf_counter() - started

        return {
            "text": text,

            "language": getattr(
                info,
                "language",
                self.language,
            ),

            "language_probability": float(
                getattr(
                    info,
                    "language_probability",
                    0.0,
                ) or 0.0
            ),

            "elapsed_seconds": round(
                elapsed,
                3,
            ),
        }


_asr_instance: FasterWhisperService | None = None
_asr_initialization_lock = Lock()


def get_asr() -> FasterWhisperService:
    global _asr_instance

    if _asr_instance is None:
        with _asr_initialization_lock:
            if _asr_instance is None:
                _asr_instance = FasterWhisperService()

    return _asr_instance