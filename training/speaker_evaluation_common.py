"""Shared runtime-aligned audio/embedding helpers for DEV-only calibration."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.VAD import VAD
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from src.speech.preprocessing import load_mono_16k

TARGET_SR = 16000
DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_VAD_SOURCE = "speechbrain/vad-crdnn-libriparty"


def l2_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero embedding.")
    return value / norm


def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def make_profile(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot create a speaker profile without enrollment embeddings.")
    return l2_normalize(np.mean(np.stack([l2_normalize(item) for item in embeddings]), axis=0))


class RuntimeAlignedEmbedder:
    """SpeechBrain VAD + ECAPA extraction matching ``extract_embedding``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_source: str = DEFAULT_MODEL_SOURCE,
        vad_source: str = DEFAULT_VAD_SOURCE,
        model_cache_dir: str | Path = "models/eval_pretrained_ecapa",
        vad_cache_dir: str | Path = "models/eval_vad_crdnn",
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._temporary_dir = tempfile.TemporaryDirectory(prefix="speaker_eval_16k_")
        self._canonical_dir = Path(self._temporary_dir.name)
        self._embeddings: dict[str, np.ndarray] = {}
        self._canonical_paths: dict[str, Path] = {}
        self.model = EncoderClassifier.from_hparams(
            source=model_source, savedir=str(model_cache_dir),
            run_opts={"device": self.device}, local_strategy=LocalStrategy.COPY,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["embedding_model"] if isinstance(payload, dict) and "embedding_model" in payload else payload
        self.model.mods.embedding_model.load_state_dict(state, strict=True)
        self.model.mods.embedding_model.eval()
        self.vad = VAD.from_hparams(
            source=vad_source, savedir=str(vad_cache_dir),
            run_opts={"device": self.device}, local_strategy=LocalStrategy.COPY,
        )

    def close(self) -> None:
        self._temporary_dir.cleanup()

    def _load_mono_16k(
        self,
        path: str,
    ) -> tuple[torch.Tensor, Path]:

        # Reuse EXACTLY the same preprocessing as runtime.
        waveform = load_mono_16k(
            path
        ).cpu()

        canonical_path = self._canonical_paths.get(
            path
        )

        if canonical_path is None:
            key = hashlib.sha1(
                str(
                    Path(path).resolve()
                ).encode("utf-8")
            ).hexdigest()

            canonical_path = (
                self._canonical_dir
                / f"{key}.wav"
            )

            torchaudio.save(
                str(canonical_path),
                waveform,
                TARGET_SR,
            )

            self._canonical_paths[
                path
            ] = canonical_path

        return waveform, canonical_path
    def embed(self, path: str) -> np.ndarray:
        if path in self._embeddings:
            return self._embeddings[path]
        waveform, canonical_path = self._load_mono_16k(path)
        try:
            boundaries = self.vad.get_speech_segments(str(canonical_path))
            pieces = []
            for boundary in boundaries:
                start = int(float(boundary[0]) * TARGET_SR)
                end = int(float(boundary[1]) * TARGET_SR)
                if end > start:
                    pieces.append(waveform[:, start:end])
            if pieces:
                trimmed = torch.cat(pieces, dim=1)
                if trimmed.shape[1] >= int(0.25 * TARGET_SR):
                    waveform = trimmed
        except Exception:
            pass  # Same runtime fallback when VAD cannot produce usable speech.
        with torch.inference_mode():
            embedding = self.model.encode_batch(waveform.to(self.device), normalize=False)
        result = l2_normalize(embedding.squeeze().detach().cpu().numpy())
        self._embeddings[path] = result
        return result
