"""Loading + shaping `everything_decoded.parquet` for the web UI.

Pure data layer: no NiceGUI/rendering concerns live here, just polars. Every
loader here scans the parquet file lazily and pushes its `packet_type`/
`received_at` filters down into that scan, so a query only materializes the
(small) slice of rows it actually needs instead of the whole file -- the
default 24h chart window is typically a few hundred rows out of a file that
only grows over the mission's lifetime.
"""

from __future__ import annotations

__all__ = [
    "ATTITUDE_COLUMNS",
    "ATTITUDE_MODE_COLUMNS",
    "BEACON_PACKET_TYPES",
    "DEFAULT_PARQUET_PATH",
    "latest_beacons",
    "latest_local_max_pending_tcmd_count",
    "load_attitude_window",
    "load_beacon_window",
    "load_bulk_file_downlink_packets",
    "load_packet_counts_per_window",
    "load_tcmd_response_packets",
]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime, timedelta
    from pathlib import Path

DEFAULT_PARQUET_PATH = (
    step_3_pipeline.DEFAULT_DATA_DIR / step_3_pipeline.OUTPUT_FILENAME
)

BEACON_PACKET_TYPES = ("BEACON_BASIC", "BEACON_EXTENDED")

# The ADCS attitude estimate + body rates, only on `BEACON_EXTENDED` packets.
ATTITUDE_COLUMNS = (
    "adcs_estimated_roll_angle_deg",
    "adcs_estimated_pitch_angle_deg",
    "adcs_estimated_yaw_angle_deg",
    "adcs_estimated_rate_x_deg_per_sec",
    "adcs_estimated_rate_y_deg_per_sec",
    "adcs_estimated_rate_z_deg_per_sec",
)
# Carried alongside each attitude frame, but not what makes a row a frame.
ATTITUDE_MODE_COLUMNS = ("adcs_attitude_estimation_mode", "adcs_control_mode")


def _scan(path: Path) -> pl.LazyFrame | None:
    """Lazily open `path`, or None if the pipeline hasn't produced it yet."""
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def load_packet_counts_per_window(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    since: datetime | None = None,
    every: timedelta,
) -> pl.DataFrame:
    """Decoded packet counts (any type) received at/after `since`, bucketed
    into `every`-wide windows of `received_at` and split by `packet_type`.

    One row per non-empty `(window_start, packet_type)` pair, oldest window
    first. Windows are aligned to the Unix epoch (so 6h windows start at
    00/06/12/18 UTC). The aggregation runs inside the lazy scan, so only the
    two needed columns are ever read, and only the (tiny) counts table is
    materialized.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame(
            schema={
                "window_start": pl.Datetime("us", "UTC"),
                "packet_type": pl.String,
                "count": pl.UInt32,
            }
        )
    lf = lf.select("received_at", "packet_type")
    if since is not None:
        lf = lf.filter(pl.col("received_at") >= since)
    return (
        lf.group_by(
            window_start=pl.col("received_at").dt.truncate(every),
            packet_type=pl.col("packet_type"),
        )
        .agg(count=pl.len())
        .sort("window_start", "packet_type")
        .collect()
    )


def load_beacon_window(
    path: Path = DEFAULT_PARQUET_PATH, *, since: datetime | None = None
) -> pl.DataFrame:
    """Beacon packets (BASIC or EXTENDED) received at/after `since`, oldest first.

    Both the packet-type and time filters are applied on the lazy frame
    before `.collect()`, so only beacon rows in the requested window are ever
    materialized.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type").is_in(BEACON_PACKET_TYPES))
    if since is not None:
        lf = lf.filter(pl.col("received_at") >= since)
    return lf.sort("received_at").collect()


def load_attitude_window(
    path: Path = DEFAULT_PARQUET_PATH, *, since: datetime | None = None
) -> pl.DataFrame:
    """`received_at` + `ATTITUDE_COLUMNS` + `ATTITUDE_MODE_COLUMNS` for every
    extended beacon received at/after `since` that carries any
    `ATTITUDE_COLUMNS` value, oldest first -- the frames
    the attitude playback steps through. Only those columns are
    materialized, so a multi-day window stays small.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(
        (pl.col("packet_type") == "BEACON_EXTENDED")
        & pl.any_horizontal(pl.col(c).is_not_null() for c in ATTITUDE_COLUMNS)
    )
    if since is not None:
        lf = lf.filter(pl.col("received_at") >= since)
    return (
        lf.select("received_at", *ATTITUDE_COLUMNS, *ATTITUDE_MODE_COLUMNS)
        .sort("received_at")
        .collect()
    )


def _filter_to_ranges(
    lf: pl.LazyFrame, ranges: Sequence[tuple[datetime, datetime]]
) -> pl.LazyFrame:
    """Restrict `lf` to rows whose `received_at` falls in any of `ranges`
    (each an inclusive [start, end] window). An empty `ranges` means no time
    filter at all -- every row.
    """
    if not ranges:
        return lf
    in_any_range = pl.any_horizontal(
        [
            pl.col("received_at").is_between(start, end, closed="both")
            for start, end in ranges
        ]
    )
    return lf.filter(in_any_range)


def load_bulk_file_downlink_packets(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    ranges: Sequence[tuple[datetime, datetime]] = (),
) -> pl.DataFrame:
    """`BULK_FILE_DOWNLINK` packets restricted to the union of `ranges`,
    ordered by `bulk_file_offset` (the order file-reassembly cares about,
    not receipt order), then by `received_at` within one offset.

    That secondary sort is what makes a retransmitted offset's copies arrive
    oldest-first, so `file_reassembly`'s conflict resolution has a stable
    order to fall back on when two copies share a `received_at` tick.

    `ranges=()` (the default) means no time filter -- every such packet ever
    decoded. The web UI's File Reassembler page is what narrows this down to
    one or more `received_at` windows isolating a single download.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type") == "BULK_FILE_DOWNLINK")
    lf = _filter_to_ranges(lf, ranges)
    return lf.sort("bulk_file_offset", "received_at").collect()


def load_tcmd_response_packets(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    ranges: Sequence[tuple[datetime, datetime]] = (),
) -> pl.DataFrame:
    """`TCMD_RESPONSE` packets restricted to the union of `ranges`, oldest
    first -- see `load_bulk_file_downlink_packets` for the `ranges` contract.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type") == "TCMD_RESPONSE")
    lf = _filter_to_ranges(lf, ranges)
    return lf.sort("received_at").collect()


def latest_beacons(
    path: Path = DEFAULT_PARQUET_PATH,
    n: int = 10,
    *,
    packet_types: Sequence[str] = BEACON_PACKET_TYPES,
) -> pl.DataFrame:
    """The `n` most recently received beacon packets, newest first.

    Ignores any time window -- this is "whatever the most recent beacon(s)
    are", even if that's older than the chart window -- so the UI can still
    show *something* when nothing has come down in the last 24h.

    `packet_types` narrows which beacon type(s) count -- e.g. pass just
    `("BEACON_EXTENDED",)` to get the latest beacon carrying the
    extended-only fields (ADCS, extended EPS/OBC telemetry), skipping over
    any more-recent `BEACON_BASIC` rows in between.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    return (
        lf.filter(pl.col("packet_type").is_in(packet_types))
        .sort("received_at", descending=True)
        .head(n)
        .collect()
    )


def latest_local_max_pending_tcmd_count(
    path: Path = DEFAULT_PARQUET_PATH,
) -> tuple[int, datetime] | None:
    """The most recent local maximum of `pending_queued_tcmd_count`, and the
    `received_at` of the beacon that reported it.

    Walks backwards from the newest beacon for as long as the pending count
    keeps rising (or holds steady), and stops at the first beacon whose
    older neighbour reports a *lower* count -- i.e. the peak of the most
    recent climb. With a queue that gets loaded by an uplink and then
    drains, that's "how many telecommands were queued up before the
    satellite started working through them". On a plateau, the *oldest*
    beacon at the peak value wins, so the time is when the peak was first
    seen. If the count never drops walking backwards, the oldest beacon is
    the peak.

    Only the two needed columns are materialized. Returns None if no beacon
    has reported a pending count yet.
    """
    lf = _scan(path)
    if lf is None:
        return None
    df = (
        lf.filter(
            pl.col("packet_type").is_in(BEACON_PACKET_TYPES)
            & pl.col("pending_queued_tcmd_count").is_not_null()
        )
        .select("received_at", "pending_queued_tcmd_count")
        .sort("received_at", descending=True)
        .collect()
    )
    if df.is_empty():
        return None

    count = pl.col("pending_queued_tcmd_count")
    # Newest-first, so `shift(-1)` is the next-*older* beacon. The first row
    # whose older neighbour is lower is the peak; `fill_null(True)` makes
    # the oldest row the fallback when no such drop exists.
    peak_idx = df.select(
        (count.shift(-1) < count).fill_null(value=True).arg_true().first()
    ).item()
    row = df.row(peak_idx, named=True)
    return int(row["pending_queued_tcmd_count"]), row["received_at"]
