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
    "ADCS_POWER_CONTROL_CHANNELS",
    "ATTITUDE_COLUMNS",
    "ATTITUDE_MODE_COLUMNS",
    "BEACON_PACKET_TYPES",
    "DEFAULT_PARQUET_PATH",
    "DEFAULT_REASSEMBLED_TCMD_PATH",
    "adcs_channel_states",
    "adcs_error_states",
    "eps_channel_states",
    "latest_beacons",
    "latest_local_max_pending_tcmd_count",
    "load_attitude_window",
    "load_beacon_window",
    "load_bulk_file_downlink_packets",
    "load_packet_counts_per_window",
    "load_reassembled_tcmd_responses",
    "newest_rows",
    "round_down_datetime",
]

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
from cachetools import TTLCache, cached

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    ADCS_ERROR_BITS,
    ADCS_FLAG_BITS,
    EPS_CHANNEL_MAP,
    PACKET_TYPE_ENUM,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DEFAULT_PARQUET_PATH = (
    step_3_pipeline.DEFAULT_DATA_DIR / step_3_pipeline.OUTPUT_FILENAME
)
DEFAULT_REASSEMBLED_TCMD_PATH = (
    step_5_pipeline.DEFAULT_DATA_DIR / step_5_pipeline.OUTPUT_FILENAME
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

# A channel whose name ends in this isn't wired to anything, so it's hidden
# unless it's actually been switched on.
_UNUSED_CHANNEL_SUFFIX = "_UNUSED"

# The ADCS components the power-control telecommand switches, in its argument
# order (CubeControl signal, CubeControl motor, CubeSense 1/2, CubeStar,
# CubeWheel 1/2/3, Motor, GPS) -- named as in `ADCS_ENABLED_BITS`. That also
# reports `GPS_LNA_POWER`, which the telecommand doesn't control, so it's left out.
ADCS_POWER_CONTROL_CHANNELS = (
    "CUBECONTROL_SIGNAL",
    "CUBECONTROL_MOTOR",
    "CUBESENSE1",
    "CUBESENSE2",
    "CUBESTAR",
    "CUBEWHEEL1",
    "CUBEWHEEL2",
    "CUBEWHEEL3",
    "MOTOR_DRIVER",
    "GPS_RECEIVER",
)

# Cached loader results expire after this long, so new data shows up.
_CACHE_TTL_SECONDS = 30


def round_down_datetime(dt: datetime, step: timedelta) -> datetime:
    """Round `dt` down to a multiple of `step` (counted from midnight), e.g.
    to the nearest 5 minutes -- so a `since` computed from "now" stays the
    same across repeat loads, and they hit the loaders' cache.
    """
    return dt - (dt - datetime.min.replace(tzinfo=dt.tzinfo)) % step


def _scan(path: Path) -> pl.LazyFrame | None:
    """Lazily open `path`, or None if the pipeline hasn't produced it yet."""
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def newest_rows(lf: pl.LazyFrame, n: int) -> pl.DataFrame:
    """`lf`'s `n` most recently received rows, newest first -- the same
    result as `lf.sort("received_at", descending=True).head(n)`.

    That one-liner materializes every column of every row before sorting
    (gigabytes, once the whole mission's history is in the file), so this
    finds the `n`th-newest `received_at` from that one column first, then
    fetches only rows at/after it. That filter is pushed into the scan, which
    skips older row groups outright and only decodes the wide columns for
    the rows that match.
    """
    cutoff = lf.select(pl.col("received_at").top_k(n).min()).collect().item()
    if cutoff is None:
        return lf.head(0).collect()
    return (
        lf.filter(pl.col("received_at") >= cutoff)
        .sort("received_at", descending=True)
        .head(n)
        .collect()
    )


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
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
                "packet_type": PACKET_TYPE_ENUM,
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


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
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


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
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
    lf: pl.LazyFrame,
    ranges: Sequence[tuple[datetime, datetime]],
    *,
    column: str = "received_at",
) -> pl.LazyFrame:
    """Restrict `lf` to rows whose `column` falls in any of `ranges` (each an
    inclusive [start, end] window). An empty `ranges` means no time filter at
    all -- every row.
    """
    if not ranges:
        return lf
    in_any_range = pl.any_horizontal(
        [pl.col(column).is_between(start, end, closed="both") for start, end in ranges]
    )
    return lf.filter(in_any_range)


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
def load_bulk_file_downlink_packets(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    ranges: tuple[tuple[datetime, datetime], ...] = (),
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


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
def load_reassembled_tcmd_responses(
    path: Path = DEFAULT_REASSEMBLED_TCMD_PATH,
    *,
    ranges: tuple[tuple[datetime, datetime], ...] = (),
) -> pl.DataFrame | None:
    """Step 5's reassembled telecommand responses (one row per response,
    multi-packet ones already joined back together) whose
    `first_received_at` falls in the union of `ranges`, oldest first -- see
    `load_bulk_file_downlink_packets` for the `ranges` contract.

    Returns None (rather than an empty frame) if step 5 hasn't produced its
    output yet, so the caller can say so instead of implying there were
    simply no responses.
    """
    lf = _scan(path)
    if lf is None:
        return None
    lf = _filter_to_ranges(lf, ranges, column="first_received_at")
    return lf.sort("first_received_at").collect()


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
def latest_beacons(
    path: Path = DEFAULT_PARQUET_PATH,
    n: int = 10,
    *,
    packet_types: tuple[str, ...] = BEACON_PACKET_TYPES,
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
    return newest_rows(lf.filter(pl.col("packet_type").is_in(packet_types)), n)


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


def _drop_never_on(states: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Drop those of `columns` that are never True in `states`."""
    return states.drop([c for c in columns if not states[c].any()])


def _json_list_states(
    beacons: pl.DataFrame, column: str, names: Sequence[str]
) -> pl.DataFrame:
    """`received_at` plus one bool column per `names` entry, in that order,
    True where `column`'s JSON list of names (e.g. `adcs_powered_list`)
    includes it. Rows where `column` is null are dropped.
    """
    if column not in beacons.columns:
        return pl.DataFrame()
    return (
        beacons.select("received_at", column)
        .drop_nulls()
        .select(
            "received_at",
            *(
                pl.col(column)
                .cast(pl.String)
                .str.contains(f'"{name}"', literal=True)
                .alias(name)
                for name in names
            ),
        )
    )


def eps_channel_states(beacons: pl.DataFrame) -> pl.DataFrame:
    """`received_at` plus one bool column per EPS power channel (named as in
    `EPS_CHANNEL_MAP`, in bit order), True where the channel is enabled.

    Decoded from `eps_enabled_channels_bitfield` rather than the
    `eps_enabled_channels_list` JSON, since the list collapses the three
    stack channels into one "STACK_X3" entry. One row per beacon (BASIC or
    EXTENDED) that reported the bitfield, in `beacons`' order. `*_UNUSED`
    channels are left out unless one was on at some point in `beacons`.
    """
    column = "eps_enabled_channels_bitfield"
    if column not in beacons.columns:
        return pl.DataFrame()
    bits = (
        pl.col(column)
        .cast(pl.String)
        .str.strip_prefix("0x")
        .str.to_integer(base=16, strict=False)
        .cast(pl.UInt32)
    )
    states = beacons.select("received_at", bits.alias(column)).drop_nulls()
    states = states.select(
        "received_at",
        *(
            ((pl.col(column) & (1 << bit_num)) != 0).alias(name)
            for bit_num, name in EPS_CHANNEL_MAP.items()
        ),
    )
    return _drop_never_on(
        states, [c for c in states.columns if c.endswith(_UNUSED_CHANNEL_SUFFIX)]
    )


def adcs_channel_states(beacons: pl.DataFrame) -> pl.DataFrame:
    """`received_at` plus one bool column per `ADCS_POWER_CONTROL_CHANNELS`
    entry, in that order, True where the component is on.

    Decoded from `adcs_powered_list`, which only `BEACON_EXTENDED` packets
    carry -- so there's one row per extended beacon in `beacons`, in order.
    """
    return _json_list_states(beacons, "adcs_powered_list", ADCS_POWER_CONTROL_CHANNELS)


def adcs_error_states(beacons: pl.DataFrame) -> pl.DataFrame:
    """`received_at` plus one bool column per ADCS error (`adcs_errors`) or
    overcurrent detection (`adcs_flags`), True where it's set -- one row per
    extended beacon in `beacons`. Columns are in bit order, so each camera's
    errors and overcurrents sit together.

    Only those set at some point in `beacons` are kept: there are 24 of
    them, and almost all are clear almost all the time.
    """
    by_bit = sorted(
        (bit_num, column, name)
        for column, bits in (
            ("adcs_errors", ADCS_ERROR_BITS),
            ("adcs_flags", ADCS_FLAG_BITS),
        )
        for bit_num, name in bits
    )
    if not {"adcs_errors", "adcs_flags"} <= set(beacons.columns):
        return pl.DataFrame()
    states = beacons.filter(pl.col("adcs_errors").is_not_null()).select(
        "received_at",
        *(
            pl.col(column)
            .cast(pl.String)
            .str.contains(f'"{name}"', literal=True)
            .fill_null(value=False)
            .alias(name)
            for _, column, name in by_bit
        ),
    )
    return _drop_never_on(states, [name for _, _, name in by_bit])
