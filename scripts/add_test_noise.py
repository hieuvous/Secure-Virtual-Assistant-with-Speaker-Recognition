"""Create a noisy copy of a WAV at a target SNR using synthetic Gaussian noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    wav, sr = torchaudio.load(args.input)
    signal_power = wav.pow(2).mean().clamp_min(1e-12)
    noise = torch.randn_like(wav)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)

    target_ratio = 10 ** (args.snr_db / 10.0)
    scale = torch.sqrt(signal_power / (target_ratio * noise_power))
    noisy = (wav + scale * noise).clamp(-1.0, 1.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), noisy, sr)
    print(f"Saved {args.snr_db} dB noisy audio to {out}")


if __name__ == "__main__":
    main()
