"""What a file says about itself.

When it was made (`file_timestamp`, in UTC): the capture time embedded in a
camera image, the start of an MPI recording (from the file's header, or the
satellite's analysis of it), or the last-modified time the ADCS reported for
a file copied off its SD card. All come from the satellite's own clocks.

And a camera image's metadata JSON (`file_metadata`): its capture settings,
time sync and ADCS state.
"""

from __future__ import annotations

__all__ = [
    "parse_adcs_modified_time",
    "parse_camera_metadata",
    "parse_capture_time",
    "parse_mpi_analysis_start_time",
]

import json
import re
from datetime import UTC, datetime
from typing import Any, cast

from loguru import logger
from PIL import ExifTags, Image

from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS

# An unset clock counts from 1970 (OBC) or 1980 (the ADCS's MS-DOS dates).
_MIN_PLAUSIBLE = datetime(2020, 1, 1, tzinfo=UTC)

# `camera_capture_v2`'s metadata JSON is the first line of a .img and the EXIF
# description of a .jpg. Matched by itself, as either may be partly downlinked.
_EPOCH_MS_RE = re.compile(rb'"epoch_ms":(\d{13})[,}]')
_CAMERA_METADATA_START = b'{"blob":"camera_capture'
_METADATA_MAX_BYTES = 1024
_CAMERA_SUFFIXES = (".img", ".jpg")

# An MPI file's first bytes. Only at offset 0: the timestamps further in are
# of later blocks, not of the start.
_MPI_HEADER_RE = re.compile(rb'\{"mpi_start":1,[^{}]{0,250}"timestamp_ms":(\d{13})\}')

_EXIF_START = b"Exif\x00\x00"
_EXIF_DATETIME_LEN = len("YYYY:MM:DD HH:MM:SS")


def _plausible(value: datetime | None) -> datetime | None:
    return value if value is not None and value >= _MIN_PLAUSIBLE else None


def _from_epoch_ms(value: object) -> datetime | None:
    if not isinstance(value, int) or not 0 < value <= MAX_VALID_EPOCH_MS:
        return None
    return _plausible(datetime.fromtimestamp(value / 1000, UTC))


def parse_mpi_analysis_start_time(earliest: object) -> datetime | None:
    """An `analyze_mpi_data` response's `earliest`, the file's first time sync."""
    if not isinstance(earliest, dict):  # null, or lost to a missing frame
        return None
    return _from_epoch_ms(cast("dict[str, Any]", earliest).get("timestamp_ms"))


def parse_adcs_modified_time(value: object) -> datetime | None:
    """An `adcs_get_latest_sd_file_blob_v2` response's `datetime`."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(f"{value}+00:00", "%Y-%m-%d %H:%M:%S%z")
    except ValueError:  # garbled, or the all-zero date of an unset clock
        return None
    return _plausible(parsed.astimezone(UTC))


def _exif_time(image: bytes) -> datetime | None:
    """EXIF's `DateTimeOriginal`, in UTC (assumed where it gives no offset)."""
    # Not `Image.open`, which fails on a partial image that still has its EXIF.
    start = image.find(_EXIF_START)
    if start < 0:
        return None
    try:
        exif = Image.Exif()
        exif.load(image[start:])
        tags = exif.get_ifd(ExifTags.IFD.Exif)
        taken = tags.get(ExifTags.Base.DateTimeOriginal)
        # One cut short by missing bytes can still parse, as the wrong time.
        if not taken or len(taken) != _EXIF_DATETIME_LEN:
            return None
        offset = tags.get(ExifTags.Base.OffsetTimeOriginal) or "+00:00"
        return datetime.strptime(f"{taken}{offset}", "%Y:%m:%d %H:%M:%S%z").astimezone(
            UTC
        )
    except Exception as e:  # noqa: BLE001 -- a damaged downlink is just undated
        logger.debug(f"Unreadable EXIF: {type(e).__name__}: {e}")
        return None


def parse_capture_time(
    file_path: str | None, data: bytes, image: bytes | None
) -> datetime | None:
    """When a camera or MPI file was captured, from its reassembled bytes
    (`data`) and the image detected in them, if any.

    An MPI file has its `mpi_start` header. For a camera file, the metadata's
    `epoch_ms` is preferred to EXIF: it has milliseconds, is in the .img too,
    and survives losing a .jpg's first packet (which takes the EXIF header
    with it). Only camera files are searched for it, as a `.run` log quotes
    the same JSON from the capture's response.
    """
    if match := _MPI_HEADER_RE.match(data):
        return _from_epoch_ms(int(match[1]))
    if file_path and file_path.lower().endswith(_CAMERA_SUFFIXES):
        match = _EPOCH_MS_RE.search(data[:_METADATA_MAX_BYTES])
        if match:
            return _from_epoch_ms(int(match[1]))
    return _plausible(_exif_time(image)) if image else None


def parse_camera_metadata(file_path: str | None, data: bytes) -> str | None:
    """A camera file's metadata JSON (see `_EPOCH_MS_RE`), exactly as sent, if
    all of it arrived.
    """
    if not (file_path and file_path.lower().endswith(_CAMERA_SUFFIXES)):
        return None
    start = data.find(_CAMERA_METADATA_START, 0, _METADATA_MAX_BYTES)
    if start < 0:
        return None
    # One character per byte, so the JSON's end is found whatever follows it.
    text = data[start : start + _METADATA_MAX_BYTES].decode("latin-1")
    try:
        metadata, end = json.JSONDecoder().raw_decode(text)
    except ValueError:  # cut by missing bytes (zeros)
        return None
    return text[:end] if isinstance(metadata, dict) else None
