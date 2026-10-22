"""When the satellite collected each beacon, MPI frame, and GNSS sample, by
its own clock -- for the web UI's MPI Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars. Every
time here is self-reported (or, for MPI frames, reassembled from
self-reported timestamps), never when the ground station received it:

  - Beacons: the OBC's `unix_epoch_time_ms` when it built the beacon.
  - MPI frames: step 6's `collected_at` (interpolated/extrapolated from the
    MPI file's own timestamps).
  - GNSS samples: the receiver's own solution time,
    `gnss_solution_utc_time`.

Something downlinked many times (a re-sent beacon, a GNSS sample in more
than one ring buffer dump) still only has one collection time, so each
source is deduplicated by it.
"""

from __future__ import annotations

__all__ = ["TIMELINE_SOURCES", "load_collection_timeline"]

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from cachetools import TTLCache, cached

from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

BASIC_BEACONS = "Basic beacons"
EXTENDED_BEACONS = "Extended beacons"
MPI_FRAMES = "MPI frames"
GNSS_SAMPLES = "GNSS samples"
# Top to bottom, as the timeline shows them.
TIMELINE_SOURCES = (BASIC_BEACONS, EXTENDED_BEACONS, MPI_FRAMES, GNSS_SAMPLES)

_SOURCE_ENUM = pl.Enum(TIMELINE_SOURCES)
_SCHEMA = {"source": _SOURCE_ENUM, "collected_at": pl.Datetime("us", "UTC")}

# Before launch: an OBC clock that's never been synced counts up from 1970.
_MIN_VALID_EPOCH_MS = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)

_CACHE_TTL_SECONDS = 30


def _beacons(decoded: pl.LazyFrame) -> pl.LazyFrame:
    return decoded.filter(
        pl.col("packet_type").is_in(("BEACON_BASIC", "BEACON_EXTENDED"))
        & pl.col("csp_crc_valid")
        & pl.col("unix_epoch_time_ms").is_between(
            _MIN_VALID_EPOCH_MS, MAX_VALID_EPOCH_MS
        )
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
    """One row per distinct (`source`, `collected_at`) -- see the module
    docstring -- ordered by source (`TIMELINE_SOURCES`), then time. Sources
    whose pipeline output doesn't exist yet are just left out.
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
        .sort("source", "collected_at")
        .collect()
    )
