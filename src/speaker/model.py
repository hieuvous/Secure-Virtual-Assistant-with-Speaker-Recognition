from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from src.config import load_settings, project_path


class ECAPAService:
    """Loads pretrained ECAPA and optionally overlays a local fine-tuned embedding state."""

    def __init__(self):
        cfg = load_settings()["speaker"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = project_path(cfg["model_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = EncoderClassifier.from_hparams(
            source=cfg["model_source"],
            savedir=str(cache_dir),
            run_opts={"device": self.device},
        )

        self.checkpoint_path = project_path(cfg["local_checkpoint"])
        self.using_finetuned = False
        if self.checkpoint_path.exists():
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
            state = payload.get("embedding_model", payload)
            self.model.mods.embedding_model.load_state_dict(state, strict=False)
            self.using_finetuned = True

        self.model.mods.embedding_model.eval()

    def encode_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav, normalize=False)
        emb = emb.squeeze()
        emb = torch.nn.functional.normalize(emb, dim=0)
        return emb.detach().cpu()


@lru_cache(maxsize=1)
def get_ecapa() -> ECAPAService:
    return ECAPAService()
