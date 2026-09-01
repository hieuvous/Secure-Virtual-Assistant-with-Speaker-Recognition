from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import shutil

import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from src.config import load_settings, project_path


class ECAPAService:
    """Base SpeechBrain pipeline + released Vietnam-Celeb ECAPA embedding weights."""

    def __init__(self):
        cfg = load_settings()["speaker"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cache_dir = project_path(cfg["model_cache_dir"])
        self._discard_broken_cache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = EncoderClassifier.from_hparams(
            source=cfg["model_source"],
            savedir=str(cache_dir),
            run_opts={"device": self.device},
            # Keep this generated cache independent from a machine's HF cache.
            local_strategy=LocalStrategy.COPY,
        )

        self.checkpoint_path = project_path(cfg["local_checkpoint"])
        self.using_finetuned = False
        self.checkpoint_metadata = {}

        if self.checkpoint_path.exists():
            payload = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state = payload["embedding_model"] if isinstance(payload, dict) and "embedding_model" in payload else payload

            # Strict=True is intentional: the uploaded release checkpoint was
            # verified against SpeechBrain ECAPA-TDNN and all keys match.
            self.model.mods.embedding_model.load_state_dict(state, strict=True)
            self.using_finetuned = True

            if isinstance(payload, dict):
                self.checkpoint_metadata = {
                    "epoch": payload.get("epoch"),
                    "embedding_dim": payload.get("embedding_dim"),
                    "best_val_loss": payload.get("best_val_loss"),
                    "num_training_speakers": len(payload.get("label_map", {})),
                }

        self.model.mods.embedding_model.eval()

    @staticmethod
    def _discard_broken_cache(cache_dir: Path) -> None:
        """Remove only a pretrained cache whose YAML is a stale local path."""
        hparams = cache_dir / "hyperparams.yaml"
        if not hparams.exists() and not hparams.is_symlink():
            return

        try:
            contents = hparams.read_text(encoding="utf-8").strip()
        except OSError:
            contents = ""

        stale_path = re.fullmatch(r"(?:[A-Za-z]:[\\/]|/).+", contents)
        if hparams.is_symlink() or stale_path:
            shutil.rmtree(cache_dir)

    def encode_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav, normalize=False)
        emb = emb.squeeze()
        emb = torch.nn.functional.normalize(emb, p=2, dim=0)
        return emb.detach().cpu()


@lru_cache(maxsize=1)
def get_ecapa() -> ECAPAService:
    return ECAPAService()
