"""What the satellite's beacons said around each MPI collection: its ADCS
modes and attitude, whether it was slewing, its clock and time sync.

A collection's beacons are those whose own onboard timestamp
(`unix_epoch_time_ms`, which the decoder's `utc_time` is made from) falls
within the collection's start/end -- the MPI data's embedded timestamps --
widened by `BEACON_WINDOW_MARGIN` either side, so even a short recording
is bracketed by a beacon. Each beacon counts once, however many times it
was received (`common.dedupe_by_beacon_time`). Only extended beacons carry
the ADCS and clock fields; basic beacons just add to `beacon_count`.
"""

from __future__ import annotations

__all__ = [
    "BEACON_SUMMARY_SCHEMA",
    "BEACON_WINDOW_MARGIN",
    "SLEW_ANGLE_DEG",
    "SLEW_RATE_DEG_PER_SEC",
    "angle_range",
    "load_beacons",
    "with_beacon_summary",
]

import itertools
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import orjson
import polars as pl

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    BEACON_EXTENDED_PACKET_TYPES,
    BEACON_PACKET_TYPES,
)
from cts1_mo_tools.cts1_processing_pipeline.common import dedupe_by_beacon_time

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

# How far before/after a collection a beacon still counts as around it --
# about one extended beacon period, so there's one either side.
BEACON_WINDOW_MARGIN = timedelta(seconds=10)
# A collection is "slewing" if any attitude angle swings through at least
# this much over its beacons, or its body rate gets above
# `SLEW_RATE_DEG_PER_SEC` (the rate alone is noisy: a median ~0.2 deg/s
# and often over 0.5 while pointing steadily).
SLEW_ANGLE_DEG = 10.0
SLEW_RATE_DEG_PER_SEC = 1.0

_TIME_MS = "unix_epoch_time_ms"
_AXES = ("roll", "pitch", "yaw")
_ANGLE_COLUMNS = {axis: f"adcs_estimated_{axis}_angle_deg" for axis in _AXES}
_BEACON_COLUMNS = (
    _TIME_MS,
    "packet_type",
    "adcs_attitude_estimation_mode",
    "adcs_control_mode",
    *_ANGLE_COLUMNS.values(),
    "adcs_angular_rate_norm_deg_per_sec",
    "obc_active_oscillator_MHz",
    "duration_since_last_time_sync_sec",
    "last_time_sync_source",
)

BEACON_SUMMARY_SCHEMA = {
    # Distinct beacons (by onboard time) around the collection, of any
    # type, and how many of them were extended beacons.
    "beacon_count": pl.Int64,
    "beacon_extended_count": pl.Int64,
    # The distinct values reported, in the order first seen -- JSON arrays,
    # e.g. '["6 - MEMS gyro EKF"]'.
    "beacon_adcs_estimation_modes": pl.String,
    "beacon_adcs_control_modes": pl.String,
    # Each estimated attitude angle's range: the smallest arc covering
    # every value (see `angle_range`) -- so `min` > `max` when it wraps
    # through +-180.
    **{
        f"beacon_{axis}_{end}_deg": pl.Float64
        for axis in _AXES
        for end in ("min", "max")
    },
    # The widest of those arcs, and the fastest body rate reported.
    "beacon_max_attitude_change_deg": pl.Float64,
    "beacon_max_angular_rate_deg_per_sec": pl.Float64,
    # "yes"/"no" (see `SLEW_ANGLE_DEG`); null with no ADCS beacons.
    "beacon_slew": pl.String,
    "beacon_oscillator_mhz": pl.String,  # JSON array, e.g. "[25]"
    # The range of time since the clock was last synced, as the beacons
    # reported it, and how it was synced (a JSON array).
    "beacon_time_sync_age_min_sec": pl.Int64,
    "beacon_time_sync_age_max_sec": pl.Int64,
    "beacon_time_sync_sources": pl.String,
}


def _window_ms(started_at: datetime, ended_at: datetime) -> tuple[int, int]:
    return (
        int((started_at - BEACON_WINDOW_MARGIN).timestamp() * 1000),
        int((ended_at + BEACON_WINDOW_MARGIN).timestamp() * 1000),
    )


def load_beacons(path: Path, collections: pl.DataFrame) -> pl.DataFrame:
    """The beacons in step 3's `everything_decoded.parquet` (`path`) around
    any of `collections` -- just the columns `with_beacon_summary` uses, one
    row per onboard time, oldest first.
    """
    windows = [
        _window_ms(start, end)
        for start, end in collections.select(
            "collection_started_at", "collection_ended_at"
        ).iter_rows()
    ]
    if not windows:
        return pl.DataFrame()
    lf = pl.scan_parquet(path).filter(
        pl.col("packet_type").is_in(BEACON_PACKET_TYPES)
        & pl.any_horizontal(
            pl.col(_TIME_MS).is_between(low, high) for low, high in windows
        )
    )
    return (
        dedupe_by_beacon_time(lf, _TIME_MS)
        .select(*_BEACON_COLUMNS)
        .with_columns(pl.col("packet_type").cast(pl.String))
        .collect()
    )


def angle_range(values: Sequence[float]) -> tuple[float, float, float] | None:
    """The smallest arc covering every angle in `values` (degrees): its
    (start, end, span), start/end in [-180, 180) -- so start > end when it
    wraps through +-180, e.g. yaw from 170 to -170 is a 20 degree span.
    None for no values.
    """
    if not values:
        return None
    angles = sorted(v % 360 for v in values)
    # The arc is everything but the widest gap between neighbouring angles.
    gaps = [b - a for a, b in itertools.pairwise(angles)]
    gaps.append(angles[0] + 360 - angles[-1])  # Round past 360.
    widest = max(range(len(gaps)), key=gaps.__getitem__)

    def signed(angle: float) -> float:
        return (angle + 180) % 360 - 180

    start, end = angles[(widest + 1) % len(angles)], angles[widest]
    return signed(start), signed(end), 360 - gaps[widest]


def _distinct_json(series: pl.Series) -> str | None:
    values = series.drop_nulls().unique(maintain_order=True).to_list()
    return orjson.dumps(values).decode() if values else None


def _summary(beacons: pl.DataFrame) -> dict[str, Any]:
    extended = beacons.filter(pl.col("packet_type").is_in(BEACON_EXTENDED_PACKET_TYPES))
    row: dict[str, Any] = {
        "beacon_count": beacons.height,
        "beacon_extended_count": extended.height,
        "beacon_adcs_estimation_modes": _distinct_json(
            extended["adcs_attitude_estimation_mode"]
        ),
        "beacon_adcs_control_modes": _distinct_json(extended["adcs_control_mode"]),
        "beacon_max_angular_rate_deg_per_sec": extended[
            "adcs_angular_rate_norm_deg_per_sec"
        ].max(),
        "beacon_oscillator_mhz": _distinct_json(extended["obc_active_oscillator_MHz"]),
        "beacon_time_sync_age_min_sec": beacons[
            "duration_since_last_time_sync_sec"
        ].min(),
        "beacon_time_sync_age_max_sec": beacons[
            "duration_since_last_time_sync_sec"
        ].max(),
        "beacon_time_sync_sources": _distinct_json(beacons["last_time_sync_source"]),
    }
    spans: list[float] = []
    for axis, column in _ANGLE_COLUMNS.items():
        arc = angle_range(extended[column].drop_nulls().to_list())
        row[f"beacon_{axis}_min_deg"], row[f"beacon_{axis}_max_deg"] = (
            (arc[0], arc[1]) if arc else (None, None)
        )
        if arc:
            spans.append(arc[2])
    rate = row["beacon_max_angular_rate_deg_per_sec"]
    row["beacon_max_attitude_change_deg"] = max(spans, default=None)
    row["beacon_slew"] = (
        None
        if not spans and rate is None
        else "yes"
        if max(spans, default=0) >= SLEW_ANGLE_DEG
        or (rate is not None and rate > SLEW_RATE_DEG_PER_SEC)
        else "no"
    )
    return row


def with_beacon_summary(
    collections: pl.DataFrame, beacons: pl.DataFrame
) -> pl.DataFrame:
    """`collections` (`mpi_collections.parquet`'s rows) with their
    `BEACON_SUMMARY_SCHEMA` columns filled in from `beacons` (from
    `load_beacons`). A collection with no beacons around it gets 0 counts
    and nulls.
    """
    rows: list[dict[str, Any]] = []
    for start, end in collections.select(
        "collection_started_at", "collection_ended_at"
    ).iter_rows():
        low, high = _window_ms(start, end)
        rows.append(_summary(beacons.filter(pl.col(_TIME_MS).is_between(low, high))))
    summary = pl.DataFrame(rows, schema=BEACON_SUMMARY_SCHEMA)
    return collections.with_columns(summary)
