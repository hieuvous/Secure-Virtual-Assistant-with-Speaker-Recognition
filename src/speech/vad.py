from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib

import torch
import torchaudio
from speechbrain.inference.VAD import VAD

from src.config import ROOT, load_settings, project_path
from src.speech.preprocessing import load_mono_16k, TARGET_SR


class VADService:
    """SpeechBrain CRDNN VAD aligned with the final DEV/TEST protocol."""

    def __init__(self):
        cfg = load_settings()["speaker"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cache_dir = project_path(cfg["vad_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = VAD.from_hparams(
            source=cfg["vad_source"],
            savedir=str(cache_dir),
            run_opts={"device": self.device},
        )

    def trim(self, audio_path: str | Path) -> torch.Tensor:
        """Return mono 16-kHz speech-only waveform [1, time]. Falls back to original."""
        audio_path = Path(audio_path)
        waveform = load_mono_16k(audio_path)

        # VAD works on a canonical 16-kHz WAV so boundary times match waveform indexing.
        runtime = ROOT / "data" / "runtime" / "vad"
        runtime.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(
            (str(audio_path.resolve()) + str(audio_path.stat().st_mtime_ns)).encode("utf-8")
        ).hexdigest()[:16]
        canonical = runtime / f"{key}.wav"
        if not canonical.exists():
            torchaudio.save(str(canonical), waveform, TARGET_SR)

        try:
            boundaries = self.model.get_speech_segments(str(canonical))
            pieces = []
            for boundary in boundaries:
                start = int(float(boundary[0]) * TARGET_SR)
                end = int(float(boundary[1]) * TARGET_SR)
                if end > start:
                    pieces.append(waveform[:, start:end])

            if pieces:
                trimmed = torch.cat(pieces, dim=1)
                # Avoid pathological near-empty VAD output.
                if trimmed.shape[1] >= int(0.25 * TARGET_SR):
                    return trimmed
        except Exception:
            pass

        return waveform


@lru_cache(maxsize=1)
def get_vad() -> VADService:
    return VADService()
