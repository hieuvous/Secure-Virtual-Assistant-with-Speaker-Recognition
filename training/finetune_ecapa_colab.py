"""
Consolidated reproducible trainer compatible with the released checkpoint format.

IMPORTANT:
- The final uploaded Epoch-10 checkpoint is authoritative.
- This script is a cleaned reconstruction for reruns; it is not claimed to be
  byte-for-byte identical to the original standalone script called by the notebook.
"""

from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.classifiers import EncoderClassifier

SR = 16000


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--dev-csv")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--cache-dir", default="/content/pretrained_ecapa_cache")
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-max-utts-per-speaker", type=int, default=8)
    p.add_argument("--chunk-seconds", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    return p.parse_args()


class DS(Dataset):
    def __init__(self, csv_path, labels, seconds, train):
        self.df = pd.read_csv(csv_path)
        self.df["speaker_id"] = self.df["speaker_id"].astype(str)
        self.labels = labels
        self.n = int(seconds * SR)
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        wav, sr = torchaudio.load(str(r["path"]))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.squeeze(0).float()
        if len(wav) >= self.n:
            m = len(wav) - self.n
            start = random.randint(0, m) if self.train and m > 0 else m // 2
            wav = wav[start:start+self.n]
        else:
            wav = F.pad(wav, (0, self.n-len(wav)))
        return wav, self.labels[str(r["speaker_id"])]


class AAM(nn.Module):
    def __init__(self, d, n, margin=.2, scale=30.):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, d))
        nn.init.xavier_uniform_(self.w)
        self.m, self.s = margin, scale
        self.cm, self.sm = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi-margin)
        self.mm = math.sin(math.pi-margin)*margin

    def forward(self, x, y):
        x, w = F.normalize(x, dim=1), F.normalize(self.w, dim=1)
        c = F.linear(x, w).clamp(-1+1e-7, 1-1e-7)
        sine = torch.sqrt(torch.clamp(1-c*c, min=1e-7))
        phi = c*self.cm - sine*self.sm
        phi = torch.where(c > self.th, phi, c-self.mm)
        oh = torch.zeros_like(c)
        oh.scatter_(1, y[:,None], 1.)
        return (oh*phi + (1-oh)*c)*self.s


def emb(model, wav, lens):
    f = model.mods.compute_features(wav)
    f = model.mods.mean_var_norm(f, lens)
    e = model.mods.embedding_model(f, lens)
    return e.squeeze(1) if e.ndim == 3 else e


def run(loader, model, head, opt, device, training):
    model.mods.embedding_model.train(training)
    head.train(training)
    loss_sum = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            lens = torch.ones(wav.shape[0], device=device)
            e = emb(model, wav, lens)
            logits = head(e, y)
            loss = F.cross_entropy(logits, y)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.mods.embedding_model.parameters()) + list(head.parameters()), 5.0
                )
                opt.step()
            loss_sum += float(loss)*len(y)
            correct += int((logits.argmax(1)==y).sum())
            total += len(y)
    return loss_sum/max(total,1), correct/max(total,1)


def main():
    a = args_parser()
    random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tr = pd.read_csv(a.train_csv)
    va = pd.read_csv(a.val_csv)
    tr["speaker_id"] = tr["speaker_id"].astype(str)
    va["speaker_id"] = va["speaker_id"].astype(str)
    speakers = sorted(tr["speaker_id"].unique())
    if set(speakers) != set(va["speaker_id"].unique()):
        raise ValueError("train/val speaker classes must match.")
    label_map = {s:i for i,s in enumerate(speakers)}

    train_ds = DS(a.train_csv, label_map, a.chunk_seconds, True)
    val_ds = DS(a.val_csv, label_map, a.chunk_seconds, False)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.num_workers, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=torch.cuda.is_available())

    model = EncoderClassifier.from_hparams(
        source=a.model_source, savedir=a.cache_dir, run_opts={"device": str(device)}
    )
    for p in model.mods.compute_features.parameters():
        p.requires_grad = False

    x0, _ = next(iter(train_dl))
    with torch.no_grad():
        d = int(emb(model, x0[:1].to(device), torch.ones(1, device=device)).shape[-1])
    head = AAM(d, len(label_map), a.margin, a.scale).to(device)
    opt = torch.optim.Adam(
        list(model.mods.embedding_model.parameters()) + list(head.parameters()), lr=a.lr
    )

    start_epoch, best = 1, float("inf")
    if a.resume:
        r = torch.load(a.resume, map_location="cpu", weights_only=False)
        state = r["embedding_model"] if isinstance(r, dict) and "embedding_model" in r else r
        model.mods.embedding_model.load_state_dict(state, strict=True)
        start_epoch = int(r.get("epoch", 0)) + 1 if isinstance(r, dict) else 1
        best = float(r.get("best_val_loss", best)) if isinstance(r, dict) else best
        print("Resume embedding from epoch:", start_epoch-1)

    ckpt_dir = Path(a.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    history, bad = [], 0

    for epoch in range(start_epoch, a.epochs+1):
        tl, ta = run(train_dl, model, head, opt, device, True)
        vl, vaa = run(val_dl, model, head, None, device, False)
        row = {"epoch": epoch, "train_loss": tl, "train_acc": ta,
               "val_loss": vl, "val_acc": vaa}
        history.append(row)
        print(json.dumps(row))

        payload = {
            "epoch": epoch,
            "embedding_model": model.mods.embedding_model.state_dict(),
            "best_val_loss": min(best, vl),
            "label_map": label_map,
            "training_args": vars(a),
            "embedding_dim": d,
        }
        torch.save(payload, ckpt_dir/"latest_checkpoint.pt")

        if vl < best - a.min_delta:
            best, bad = vl, 0
            payload["best_val_loss"] = best
            torch.save(payload, out)
        else:
            bad += 1

        if bad >= a.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    (out.with_suffix(".history.json")).write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
