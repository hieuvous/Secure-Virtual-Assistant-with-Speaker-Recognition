"""Request-scoped temporary audio helpers.

These helpers intentionally keep raw microphone audio outside the repository
and outside persistent storage. Callers must consume paths only inside the
context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence


@contextmanager
def temporary_audio_paths(
    recordings: Sequence[bytes], *, prefix: str, basename: str = "sample"
) -> Iterator[list[Path]]:
    """Materialize in-memory WAV bytes for one operation, then delete them."""
    if not recordings:
        raise ValueError("At least one audio recording is required.")

    with TemporaryDirectory(prefix=prefix) as directory:
        root = Path(directory)
        paths: list[Path] = []
        for index, recording in enumerate(recordings, start=1):
            if not isinstance(recording, bytes) or not recording:
                raise ValueError(f"Audio recording {index} is missing or empty.")
            path = root / f"{basename}_{index}.wav"
            path.write_bytes(recording)
            paths.append(path)
        yield paths


@contextmanager
def temporary_audio_path(recording: bytes, *, prefix: str, filename: str = "query.wav") -> Iterator[Path]:
    """Materialize one in-memory WAV for the duration of a request."""
    with temporary_audio_paths([recording], prefix=prefix, basename=Path(filename).stem) as paths:
        yield paths[0]
