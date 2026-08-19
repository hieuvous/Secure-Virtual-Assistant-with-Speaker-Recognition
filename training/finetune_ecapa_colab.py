"""Fine-tune the project's pretrained ECAPA-TDNN model on Google Colab.

The output and checkpoints keep an ``embedding_model`` key, so they can be
loaded by the existing local app and evaluation script.
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
from speechbrain.inference.classifiers import EncoderClassifier
from torch.utils.data import DataLoader, Dataset


TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output", required=True,
                        help="Path for the best inference-compatible model.")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory for epoch checkpoints and best_model.pt.")
    parser.add_argument("--cache-dir", default="./pretrained_ecapa",
                        help="SpeechBrain pretrained-model cache directory.")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint_epoch_N.pt or best_model.pt.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(self, csv_path, label_map, chunk_seconds, training):
        self.df = pd.read_csv(csv_path)
        self.df["speaker_id"] = self.df["speaker_id"].astype(str)
        self.label_map = label_map
        self.num_samples = int(TARGET_SR * chunk_seconds)
        self.training = training

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0).float()
        if wav.numel() >= self.num_samples:
            start = (random.randint(0, wav.numel() - self.num_samples)
                     if self.training else (wav.numel() - self.num_samples) // 2)
            return wav[start:start + self.num_samples]
        return F.pad(wav, (0, self.num_samples - wav.numel()))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self._load(str(row["path"])), self.label_map[str(row["speaker_id"])]


class AAMSoftmaxHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = float(margin), float(scale)
        self.cos_m, self.sin_m = math.cos(self.margin), math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


def get_embeddings(pretrained, wavs, lens):
    feats = pretrained.mods.compute_features(wavs)
    feats = pretrained.mods.mean_var_norm(feats, lens)
    try:
        embeddings = pretrained.mods.embedding_model(feats, lens)
    except TypeError:
        embeddings = pretrained.mods.embedding_model(feats)
    return embeddings.squeeze(1) if embeddings.ndim == 3 else embeddings


def run_epoch(loader, pretrained, head, optimizer, device, training):
    pretrained.mods.embedding_model.train(training)
    head.train(training)
    total_loss = total_correct = total = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for wavs, labels in loader:
            wavs, labels = wavs.to(device), labels.to(device)
            embeddings = get_embeddings(pretrained, wavs, torch.ones(len(wavs), device=device))
            logits = head(embeddings, labels)
            loss = F.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(pretrained.mods.embedding_model.parameters()) + list(head.parameters()), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(wavs)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += len(wavs)
    return {"loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


def checkpoint_payload(epoch, pretrained, head, optimizer, best_val_loss, label_map, args, embedding_dim):
    return {
        "epoch": epoch,
        "embedding_model": pretrained.mods.embedding_model.state_dict(),
        "classification_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "label_map": label_map,
        "training_args": vars(args),
        "embedding_dim": embedding_dim,
    }


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CUDA unavailable; using CPU)"))

    train_df = pd.read_csv(args.train_csv)
    train_df["speaker_id"] = train_df["speaker_id"].astype(str)
    label_map = {speaker: i for i, speaker in enumerate(sorted(train_df["speaker_id"].unique()))}
    val_df = pd.read_csv(args.val_csv)
    unknown_val = set(val_df["speaker_id"].astype(str)) - set(label_map)
    if unknown_val:
        raise ValueError("val.csv must contain only speakers present in train.csv; found: " + str(sorted(unknown_val)[:5]))

    train_loader = DataLoader(AudioDataset(args.train_csv, label_map, args.chunk_seconds, True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(AudioDataset(args.val_csv, label_map, args.chunk_seconds, False),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    if not len(train_loader) or not len(val_loader):
        raise ValueError("train.csv and val.csv must both contain at least one usable audio row.")

    cache_dir, checkpoint_dir, output = Path(args.cache_dir), Path(args.checkpoint_dir), Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pretrained = EncoderClassifier.from_hparams(source=args.model_source, savedir=str(cache_dir),
                                                run_opts={"device": str(device)})
    with torch.no_grad():
        wav0, _ = next(iter(train_loader))
        embedding_dim = int(get_embeddings(pretrained, wav0[:1].to(device), torch.ones(1, device=device)).shape[-1])
    head = AAMSoftmaxHead(embedding_dim, len(label_map), args.margin, args.scale).to(device)
    optimizer = torch.optim.Adam(list(pretrained.mods.embedding_model.parameters()) + list(head.parameters()), lr=args.lr)

    start_epoch, best_val = 1, float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        required = {"epoch", "embedding_model", "classification_head", "optimizer", "best_val_loss", "label_map"}
        missing = required - set(resume)
        if missing:
            raise ValueError(f"Resume checkpoint is missing required keys: {sorted(missing)}")
        if resume["label_map"] != label_map:
            raise ValueError("Resume label_map does not match the current train.csv.")
        pretrained.mods.embedding_model.load_state_dict(resume["embedding_model"])
        head.load_state_dict(resume["classification_head"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch, best_val = int(resume["epoch"]) + 1, float(resume["best_val_loss"])
        print(f"Resumed from epoch {resume['epoch']}; continuing at epoch {start_epoch}.")

    history_path = checkpoint_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume and history_path.exists() else []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(train_loader, pretrained, head, optimizer, device, True)
        val_metrics = run_epoch(val_loader, pretrained, head, optimizer, device, False)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"],
               "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]}
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best = checkpoint_payload(epoch, pretrained, head, optimizer, best_val, label_map, args, embedding_dim)
            torch.save(best, checkpoint_dir / "best_model.pt")
            torch.save(best, output)
            print(f"Saved best model: {output}")
        epoch_checkpoint = checkpoint_payload(epoch, pretrained, head, optimizer, best_val, label_map, args, embedding_dim)
        epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(epoch_checkpoint, epoch_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Saved epoch checkpoint: {epoch_path}")


if __name__ == "__main__":
    main()
