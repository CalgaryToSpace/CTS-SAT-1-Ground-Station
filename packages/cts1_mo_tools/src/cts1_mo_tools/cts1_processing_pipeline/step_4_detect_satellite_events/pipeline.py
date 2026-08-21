"""Step 4: detect satellite events (reboots, uplinks) from beacon counters.

Every beacon (`BEACON_BASIC` and `BEACON_EXTENDED` both carry the same
fixed timing fields, so both are considered together, sorted by
`received_at`) carries a handful of counters that only ever count up while
the satellite keeps running, and reset to (near) zero the instant something
happens:

  - `uptime_ms`: the OBC's own uptime -- resets on an OBC reboot.
  - `eps_uptime_sec`: the EPS's uptime -- resets on an EPS reboot/reset.
  - `duration_since_last_uplink_ms`: time since the last received uplink --
    resets whenever a new uplink command comes in.

The first beacon where one of these counters is lower than the previous
beacon's is the first beacon received *after* that event -- the event
itself happened `counter value` before that beacon's own `received_at`.
This step finds every such drop across the whole beacon history and stacks
the three kinds of event into one unpivoted table, one row per detected
event, each carrying the OBC/EPS reboot-reason fields and EPS fault count
as reported by the detecting beacon.

Like steps 2 and 3, this is parquet-in-parquet-out: no database involved,
and cheap enough to reprocess from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_FILENAME",
    "compute_satellite_events",
    "run",
]

from typing import TYPE_CHECKING, NamedTuple

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_OUTPUT_DIR = step_3_pipeline.DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME = "satellite_events_from_beacons.parquet"

BEACON_PACKET_TYPES = ("BEACON_BASIC", "BEACON_EXTENDED")


class _EventSpec(NamedTuple):
    counter_column: str
    event_type: str
    ms_per_unit: float  # multiply the raw counter by this to get milliseconds


# The three counters that only ever count up until the event that resets
# them -- see the module docstring.
EVENT_SPECS: tuple[_EventSpec, ...] = (
    _EventSpec("uptime_ms", "OBC Reboot", 1.0),
    _EventSpec("eps_uptime_sec", "EPS Reboot", 1000.0),
    _EventSpec("duration_since_last_uplink_ms", "Uplinked Commands", 1.0),
)

# The output schema, in column order -- also what an empty result (no
# beacons decoded yet) is built with, so downstream readers always see the
# same columns regardless of how much history exists.
_OUTPUT_SCHEMA = {
    "event_type": pl.String,
    "detected_at": pl.Datetime("us", "UTC"),
    "estimated_event_at": pl.Datetime("us", "UTC"),
    "time_since_event_when_detected_ms": pl.Int64,
    "detecting_packet_type": pl.String,
    "obc_reboot_reason": pl.String,
    "eps_reboot_reason": pl.String,
    "eps_reset_count": pl.Int64,
}


def _events_for_counter(beacons: pl.DataFrame, spec: _EventSpec) -> pl.DataFrame:
    """Every row of `beacons` where `spec.counter_column` is lower than the
    immediately preceding beacon's -- the first beacon after a `spec.event_type`
    event, per the module docstring.
    """
    counter = pl.col(spec.counter_column)
    previous = counter.shift(1)
    dropped = beacons.filter(
        counter.is_not_null() & previous.is_not_null() & (counter < previous)
    )

    time_since_event_ms = (counter * spec.ms_per_unit).round(0).cast(pl.Int64)
    return dropped.select(
        event_type=pl.lit(spec.event_type),
        detected_at=pl.col("received_at"),
        estimated_event_at=(
            pl.col("received_at") - pl.duration(milliseconds=time_since_event_ms)
        ),
        time_since_event_when_detected_ms=time_since_event_ms,
        detecting_packet_type=pl.col("packet_type"),
        obc_reboot_reason=pl.col("reboot_reason"),
        eps_reboot_reason=pl.col("eps_reset_cause"),
        eps_reset_count=pl.col("eps_total_fault_count").cast(pl.Int64),
    )


def compute_satellite_events(decoded_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: `everything_decoded` -> one row per detected event.

    Args:
        decoded_packets: `everything_decoded.parquet`, as written by step 3.

    Returns:
        One row per detected OBC reboot / EPS reboot / uplinked-commands
        event, newest first -- see the module docstring for the detection
        rule and `_OUTPUT_SCHEMA` for the columns.
    """
    if decoded_packets.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    beacons = decoded_packets.filter(
        pl.col("packet_type").is_in(BEACON_PACKET_TYPES)
    ).sort("received_at")
    if beacons.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    events = pl.concat(
        [_events_for_counter(beacons, spec) for spec in EVENT_SPECS],
        how="vertical_relaxed",
    )
    return events.sort("detected_at", descending=True)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Recompute `satellite_events_from_beacons.parquet` from step 3's output.

    Reads `everything_decoded.parquet` from `output_dir` (where step 3
    writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    decoded_path = output_dir / step_3_pipeline.OUTPUT_FILENAME
    if not decoded_path.exists():
        msg = f"{decoded_path} not found -- run step_3 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {decoded_path}")
    decoded_packets = pl.read_parquet(decoded_path)

    result_df = compute_satellite_events(decoded_packets)

    out_path = output_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)

    counts = result_df["event_type"].value_counts().sort("event_type")
    logger.info(
        f"Done. {result_df.height:,} event(s) written to {out_path}: "
        + ", ".join(f"{row['event_type']}={row['count']}" for row in counts.to_dicts())
    )
