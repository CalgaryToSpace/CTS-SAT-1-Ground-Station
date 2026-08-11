"""Ephemeral download + format conversion helpers for observation audio.

Files are always written under a caller-supplied temporary directory (see
:func:`tempfile.TemporaryDirectory` in ``pipeline.py``) and are never
persisted long-term: downloading a SatNOGS ``.ogg`` recording again is cheap,
so there is no on-disk cache to invalidate.
"""

from __future__ import annotations

__all__ = ["convert_ogg_to_wav", "download_audio"]

import subprocess
from typing import TYPE_CHECKING

import requests
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def download_audio(url: str, dest_dir: Path) -> Path:
    """Download an observation's audio file into dest_dir and return its path."""
    dest_path = dest_dir / (url.rsplit("/", 1)[-1] or "audio.ogg")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    logger.debug(f"Downloaded {url} -> {dest_path} ({dest_path.stat().st_size} bytes)")
    return dest_path


def convert_ogg_to_wav(ogg_path: Path, wav_path: Path | None = None) -> Path:
    """Convert an .ogg recording to .wav via `sox` (gr_satellites reads wav/raw)."""
    if wav_path is None:
        wav_path = ogg_path.with_suffix(".wav")
    try:
        subprocess.run(  # noqa: S603
            ["sox", str(ogg_path), str(wav_path)],  # noqa: S607
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(f"sox failed on {ogg_path}: {exc.stderr.decode(errors='replace')}")
        raise
    return wav_path
