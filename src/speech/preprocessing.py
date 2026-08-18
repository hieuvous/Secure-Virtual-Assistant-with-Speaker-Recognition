from __future__ import annotations

from pathlib import Path
import torch
import torchaudio

TARGET_SR = 16000


def load_mono_16k(audio_path: str | Path) -> torch.Tensor:
    """Load audio as mono float tensor with shape [1, time] at 16 kHz."""
    wav, sr = torchaudio.load(str(audio_path))
    if wav.ndim != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    peak = wav.abs().max()
    if peak > 1.0:
        wav = wav / peak
    return wav.float()


def save_uploaded_audio(uploaded_file, output_path: str | Path) -> Path:
    """Save a Streamlit UploadedFile/audio_input object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(uploaded_file.getvalue())
    return output_path
