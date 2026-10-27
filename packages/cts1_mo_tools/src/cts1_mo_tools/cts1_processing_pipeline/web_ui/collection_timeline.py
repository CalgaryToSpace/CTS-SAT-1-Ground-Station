"""When the satellite collected each beacon, MPI frame, and GNSS sample, by
its own clock -- for the web UI's MPI Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars. Every
time here is self-reported (or, for MPI frames, reassembled from
self-reported timestamps), never when the ground station received it:

  - Beacons: the OBC's `unix_epoch_time_ms` when it built the beacon.
    Ones stamped before launch are left out: early in the mission, before
    its clock was first set, the OBC stamped beacons with stale
    pre-launch times.
  - MPI frames: step 6's `collected_at` (interpolated/extrapolated from the
    MPI file's own timestamps).
  - GNSS samples: the receiver's own solution time,
    `gnss_solution_utc_time`.

Something downlinked many times (a re-sent beacon, a GNSS sample in more
than one ring buffer dump) still only has one collection time, so each
source is deduplicated by it.

The distinct times are then binned (`BIN_SECONDS`), keeping how many fell in
each bin: there can be ~100k of them, far more than the chart can show
apart or the browser can take quickly.
"""

from __future__ import annotations

__all__ = [
    "BIN_SECONDS",
    "FALLBACK_LAUNCHED_AT",
    "TIMELINE_SOURCES",
    "launched_at",
    "load_collection_timeline",
]

import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import requests
from cachetools import TTLCache, cached
from loguru import logger

from cts1_mo_tools import satnogs_data
from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)

from .overpasses import DEFAULT_NORAD_ID

if TYPE_CHECKING:
    from pathlib import Path

BASIC_BEACONS = "Basic beacons"
EXTENDED_BEACONS = "Extended beacons"
MPI_FRAMES = "MPI frames"
GNSS_SAMPLES = "GNSS samples"
# Top to bottom, as the timeline shows them.
TIMELINE_SOURCES = (BASIC_BEACONS, EXTENDED_BEACONS, MPI_FRAMES, GNSS_SAMPLES)

_SOURCE_ENUM = pl.Enum(TIMELINE_SOURCES)
_SCHEMA = {
    "source": _SOURCE_ENUM,
    "collected_at": pl.Datetime("us", "UTC"),
    "count": pl.UInt32,
}

# The width of each timeline bin.
BIN_SECONDS = 15

# What SatNOGS DB said when this was written -- used until it answers.
FALLBACK_LAUNCHED_AT = datetime(2026, 5, 3, 6, 59, tzinfo=UTC)
# After failing to fetch the launch date, how long to use the fallback
# before trying again (rather than waiting on SatNOGS every page load).
_LAUNCH_DATE_RETRY_SEC = 10 * 60

_CACHE_TTL_SECONDS = 30


_launch_date_lock = threading.Lock()
_launch_date: datetime | None = None
_launch_date_failed_at: float | None = None  # time.monotonic()


def launched_at() -> datetime:
    """When the satellite launched, per SatNOGS DB -- fetched once, then
    kept for good. If SatNOGS can't be reached, `FALLBACK_LAUNCHED_AT`,
    trying again after `_LAUNCH_DATE_RETRY_SEC`. Blocking -- call it off
    the event loop.
    """
    global _launch_date, _launch_date_failed_at  # noqa: PLW0603
    with _launch_date_lock:
        if _launch_date is not None:
            return _launch_date
        if (
            _launch_date_failed_at is not None
            and time.monotonic() - _launch_date_failed_at < _LAUNCH_DATE_RETRY_SEC
        ):
            return FALLBACK_LAUNCHED_AT
        try:
            _launch_date = satnogs_data.fetch_launch_date(DEFAULT_NORAD_ID)
        except (requests.RequestException, LookupError, TypeError, ValueError) as e:
            _launch_date_failed_at = time.monotonic()
            logger.warning(
                f"Couldn't fetch the launch date from SatNOGS DB ({e}); using "
                f"{FALLBACK_LAUNCHED_AT:%Y-%m-%d %H:%M} UTC for now."
            )
            return FALLBACK_LAUNCHED_AT
        return _launch_date


def _beacons(decoded: pl.LazyFrame) -> pl.LazyFrame:
    launched_at_ms = int(launched_at().timestamp() * 1000)
    return decoded.filter(
        pl.col("packet_type").is_in(("BEACON_BASIC", "BEACON_EXTENDED"))
        & pl.col("csp_crc_valid")
        & pl.col("unix_epoch_time_ms").is_between(launched_at_ms, MAX_VALID_EPOCH_MS)
    ).select(
        source=pl.when(pl.col("packet_type") == "BEACON_BASIC")
        .then(pl.lit(BASIC_BEACONS))
        .otherwise(pl.lit(EXTENDED_BEACONS))
        .cast(_SOURCE_ENUM),
        collected_at=pl.from_epoch("unix_epoch_time_ms", time_unit="ms")
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("UTC"),
    )


def _gnss_samples(decoded: pl.LazyFrame) -> pl.LazyFrame:
    return decoded.filter(
        (pl.col("packet_type") == "GNSS_BESTXYZB_SAMPLE")
        & pl.col("csp_crc_valid")
        & pl.col("gnss_crc_valid")
        & pl.col("gnss_solution_utc_time").is_not_null()
    ).select(
        source=pl.lit(GNSS_SAMPLES, dtype=_SOURCE_ENUM),
        collected_at=pl.col("gnss_solution_utc_time")
        .str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
        .dt.convert_time_zone("UTC"),
    )


def _mpi_frames(mpi_frames: pl.LazyFrame) -> pl.LazyFrame:
    return mpi_frames.select(
        source=pl.lit(MPI_FRAMES, dtype=_SOURCE_ENUM),
        collected_at=pl.col("collected_at"),
    )


@cached(TTLCache(maxsize=2, ttl=_CACHE_TTL_SECONDS))
def load_collection_timeline(data_dir: Path) -> pl.DataFrame:
    """One row per `BIN_SECONDS` bin of each source with anything in it:
    `collected_at` (the bin's start) and `count`, how many distinct
    collection times fell in it -- see the module docstring -- ordered by
    source (`TIMELINE_SOURCES`), then time. Sources whose pipeline output
    doesn't exist yet are just left out. Blocking (it may fetch the launch
    date) -- call it off the event loop.
    """
    parts: list[pl.LazyFrame] = []
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    if decoded_path.exists():
        decoded = pl.scan_parquet(decoded_path)
        parts += [_beacons(decoded), _gnss_samples(decoded)]
    mpi_path = data_dir / step_6_pipeline.OUTPUT_FILENAME
    if mpi_path.exists():
        parts.append(_mpi_frames(pl.scan_parquet(mpi_path)))
    if not parts:
        return pl.DataFrame(schema=_SCHEMA)
    return (
        pl.concat(parts)
        .drop_nulls("collected_at")
        .unique()
        .group_by("source", pl.col("collected_at").dt.truncate(f"{BIN_SECONDS}s"))
        .agg(count=pl.len().cast(pl.UInt32))
        .sort("source", "collected_at")
        .collect()
    )
