from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import re
import shutil

import torch
import torchaudio
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy

from src.config import ROOT, load_settings, project_path
from src.speech.preprocessing import load_mono_16k, TARGET_SR


class VADService:
    """SpeechBrain CRDNN VAD aligned with the final DEV/TEST protocol."""

    def __init__(self):
        cfg = load_settings()["speaker"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cache_dir = project_path(cfg["vad_cache_dir"])
        self._discard_broken_cache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = VAD.from_hparams(
            source=cfg["vad_source"],
            savedir=str(cache_dir),
            run_opts={"device": self.device},
            # Keep actual model files in the project cache.  The SpeechBrain
            # default is SYMLINK, whose target can be a user-specific HF cache.
            local_strategy=LocalStrategy.COPY,
        )

    @staticmethod
    def _discard_broken_cache(cache_dir: Path) -> None:
        """Remove only a VAD cache whose YAML is a stale local-cache path."""
        hparams = cache_dir / "hyperparams.yaml"
        if not hparams.exists() and not hparams.is_symlink():
            return

        try:
            contents = hparams.read_text(encoding="utf-8").strip()
        except OSError:
            contents = ""

        # A normal HyperPyYAML file is never just an absolute cache path.
        stale_path = re.fullmatch(r"(?:[A-Za-z]:[\\/]|/).+", contents)
        if hparams.is_symlink() or stale_path:
            shutil.rmtree(cache_dir)

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
