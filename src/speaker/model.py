from __future__ import annotations

from functools import lru_cache
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from src.config import load_settings, project_path


class ECAPAService:
    """Base SpeechBrain pipeline + released Vietnam-Celeb ECAPA embedding weights."""

    def __init__(self):
        cfg = load_settings()["speaker"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cache_dir = project_path(cfg["model_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = EncoderClassifier.from_hparams(
            source=cfg["model_source"],
            savedir=str(cache_dir),
            run_opts={"device": self.device},
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
