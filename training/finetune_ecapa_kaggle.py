"""
Practical ECAPA fine-tuning starter for Kaggle.

This is intentionally simpler than reproducing the full SpeechBrain VoxCeleb recipe.
It:
1) loads speechbrain/spkrec-ecapa-voxceleb,
2) keeps the pretrained feature pipeline,
3) fine-tunes the ECAPA embedding model,
4) trains a new AAM-Softmax-style speaker classification head,
5) saves ONLY the fine-tuned embedding model state for drop-in local inference.

Expected CSV columns:
speaker_id,path
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.classifiers import EncoderClassifier


TARGET_SR = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk-seconds", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(self, csv_path: str, label_map: dict[str, int], chunk_seconds: float, training: bool):
        self.df = pd.read_csv(csv_path)
        self.df["speaker_id"] = self.df["speaker_id"].astype(str)
        self.label_map = label_map
        self.num_samples = int(TARGET_SR * chunk_seconds)
        self.training = training

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0).float()

        if wav.numel() >= self.num_samples:
            if self.training:
                start = random.randint(0, wav.numel() - self.num_samples)
            else:
                start = (wav.numel() - self.num_samples) // 2
            wav = wav[start : start + self.num_samples]
        else:
            pad = self.num_samples - wav.numel()
            wav = F.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spk = str(row["speaker_id"])
        return self._load(str(row["path"])), self.label_map[spk]


class AAMSoftmaxHead(nn.Module):
    """Small self-contained additive angular margin head."""
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return logits


def get_embeddings(pretrained, wavs, lens):
    feats = pretrained.mods.compute_features(wavs)
    feats = pretrained.mods.mean_var_norm(feats, lens)
    try:
        emb = pretrained.mods.embedding_model(feats, lens)
    except TypeError:
        emb = pretrained.mods.embedding_model(feats)
    return emb.squeeze(1) if emb.ndim == 3 else emb


def run_epoch(loader, pretrained, head, optimizer, device, training: bool):
    pretrained.mods.embedding_model.train(training)
    head.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for wavs, labels in loader:
            wavs = wavs.to(device)
            labels = labels.to(device)
            lens = torch.ones(wavs.shape[0], device=device)

            embeddings = get_embeddings(pretrained, wavs, lens)
            logits = head(embeddings, labels)
            loss = F.cross_entropy(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(pretrained.mods.embedding_model.parameters()) + list(head.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

            total_loss += float(loss.item()) * wavs.shape[0]
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total += wavs.shape[0]

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df = pd.read_csv(args.train_csv)
    train_df["speaker_id"] = train_df["speaker_id"].astype(str)
    speakers = sorted(train_df["speaker_id"].unique())
    label_map = {spk: i for i, spk in enumerate(speakers)}

    val_df = pd.read_csv(args.val_csv)
    val_df["speaker_id"] = val_df["speaker_id"].astype(str)
    unknown_val = set(val_df["speaker_id"]) - set(label_map)
    if unknown_val:
        raise ValueError(
            "val.csv for classification training must contain the SAME speakers as train.csv. "
            f"Found unknown val speakers: {list(unknown_val)[:5]}"
        )

    train_ds = AudioDataset(args.train_csv, label_map, args.chunk_seconds, True)
    val_ds = AudioDataset(args.val_csv, label_map, args.chunk_seconds, False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    pretrained = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir="/kaggle/working/pretrained_ecapa",
        run_opts={"device": str(device)},
    )
    pretrained.mods.embedding_model.train()

    # Determine embedding dimension safely using one batch.
    wav0, _ = next(iter(train_loader))
    wav0 = wav0[:1].to(device)
    with torch.no_grad():
        emb0 = get_embeddings(
            pretrained, wav0, torch.ones(1, device=device)
        )
    embedding_dim = int(emb0.shape[-1])
    print("Embedding dim:", embedding_dim, "Classes:", len(label_map))

    head = AAMSoftmaxHead(
        embedding_dim, len(label_map), margin=args.margin, scale=args.scale
    ).to(device)

    optimizer = torch.optim.Adam(
        list(pretrained.mods.embedding_model.parameters()) + list(head.parameters()),
        lr=args.lr,
    )

    best_val = float("inf")
    history = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader, pretrained, head, optimizer, device, training=True
        )
        val_metrics = run_epoch(
            val_loader, pretrained, head, None, device, training=False
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "embedding_model": pretrained.mods.embedding_model.state_dict(),
                    "label_map": label_map,
                    "training_args": vars(args),
                    "embedding_dim": embedding_dim,
                    "best_val_loss": best_val,
                },
                output,
            )
            print("Saved best checkpoint:", output)

    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("History:", history_path)


if __name__ == "__main__":
    main()
