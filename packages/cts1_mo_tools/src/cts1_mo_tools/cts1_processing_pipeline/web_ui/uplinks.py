"""Finished uplink passes, from the beacons' telecommand counters.

Every beacon carries `total_tcmd_queued_count` (telecommands uplinked since
boot, only ever counting up) and `duration_since_last_uplink_ms` (meaningful
only once that count is above 0). The beacon's own `utc_time` minus that
duration is the time of the most recent uplink -- the same instant (to
within a few ms, or a couple of seconds across an onboard clock resync) for
every beacon until the next uplink comes in.

An uplink pass lasts up to `UPLINK_WINDOW`, and beacons received partway
through one only show the count so far -- so a pass's command count is the
count after it minus the count from a beacon received *before the window
could have started*, never anything from the middle of it.

Pure data layer: no NiceGUI/rendering concerns live here, just polars.
"""

from __future__ import annotations

__all__ = [
    "PASS_TCMD_LOOKBACK",
    "PASS_TCMD_MARGIN",
    "UPLINK_WINDOW",
    "FinishedUplink",
    "detect_finished_uplinks",
    "latest_finished_uplink",
    "load_beacon_tcmd_counts",
    "tcmds_uplinked_per_pass",
]

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
from cachetools import TTLCache, cached

from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS

from .data import BEACON_PACKET_TYPES, DEFAULT_PARQUET_PATH, round_down_datetime

if TYPE_CHECKING:
    from pathlib import Path

# The longest an uplink pass lasts. Uplinks closer together than this are
# one pass; a pass's count before is taken from a beacon at least this long
# before its last uplink.
UPLINK_WINDOW = timedelta(minutes=15)

# Two beacons are from the same boot if their implied boot times (`utc_time`
# minus `uptime_sec`) agree to within this -- they drift apart by a few
# seconds whenever the onboard clock is resynced, but a reboot moves it by
# far more.
SAME_BOOT_TOLERANCE = timedelta(minutes=3)

# `tcmds_uplinked_per_pass` compares the beacon from before AOS minus this
# with the one from after LOS plus this.
PASS_TCMD_MARGIN = timedelta(seconds=1)
# How far before the first pass's AOS (minus `PASS_TCMD_MARGIN`)
# `tcmds_uplinked_per_pass`'s beacons are loaded from -- a pass with no beacon
# in that long before it gets no count, rather than one from a stale beacon.
PASS_TCMD_LOOKBACK = timedelta(days=1)

_CACHE_TTL_SECONDS = 30


@dataclass(frozen=True, slots=True)
class FinishedUplink:
    """One row of `detect_finished_uplinks` -- see there for each field."""

    uplink_finished_at: datetime
    tcmd_count_before: int | None
    tcmd_count_after: int
    tcmd_count_delta: int | None


_OUTPUT_SCHEMA = {
    "uplink_finished_at": pl.Datetime("us", "UTC"),
    "tcmd_count_before": pl.Int64,
    "tcmd_count_after": pl.Int64,
    "tcmd_count_delta": pl.Int64,
}


def detect_finished_uplinks(beacons: pl.DataFrame) -> pl.DataFrame:
    """One row per finished uplink pass, newest first.

    Args:
        beacons: Beacon rows with `utc_time` (ISO string or UTC datetime),
            `uptime_sec`, `total_tcmd_queued_count`, and
            `duration_since_last_uplink_ms`, in any order. Should already be
            filtered to CRC-valid beacons -- a bit-flipped one carries
            near-random counters.

    Returns:
        - `uplink_finished_at`: the pass's last uplink.
        - `tcmd_count_after`: `total_tcmd_queued_count` once it finished.
        - `tcmd_count_before`: the count from the last beacon received at
          least `UPLINK_WINDOW` before `uplink_finished_at` (0 if the
          satellite booted within that window). Null if there's no such
          beacon from the same boot and after the previous pass -- e.g. the
          pass is the first one in `beacons`.
        - `tcmd_count_delta`: telecommands uplinked during the pass, null
          whenever `tcmd_count_before` is.

        A pass only counts as finished once a beacon received at least
        `UPLINK_WINDOW` after its first known uplink still shows it as the
        latest one; until then, more uplinks could still be coming in.
    """
    if beacons.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    utc_time = pl.col("utc_time")
    if beacons.schema.get("utc_time") == pl.String:
        utc_time = utc_time.str.to_datetime(time_zone="UTC")
    count = pl.col("total_tcmd_queued_count")
    rows = (
        beacons.select(
            at=utc_time,
            boot_at=utc_time - pl.duration(milliseconds=pl.col("uptime_sec") * 1000),
            count=count,
            last_uplink_at=pl.when(count > 0).then(
                utc_time
                - pl.duration(milliseconds=pl.col("duration_since_last_uplink_ms"))
            ),
        )
        .drop_nulls(["at", "count"])
        .sort("at")
    )

    passes = (
        rows.drop_nulls("last_uplink_at")
        .sort("last_uplink_at")
        .with_columns(
            pass_id=(pl.col("last_uplink_at").diff() > UPLINK_WINDOW)
            .fill_null(value=False)
            .cum_sum()
        )
        .sort("pass_id", "at")
        .group_by("pass_id", maintain_order=True)
        .agg(
            # As reported by the earliest beacon showing the final count --
            # the one least likely to have been through a clock resync since.
            uplink_finished_at=pl.col("last_uplink_at")
            .filter(pl.col("count") == pl.col("count").max())
            .first(),
            boot_at=pl.col("boot_at")
            .filter(pl.col("count") == pl.col("count").max())
            .first(),
            tcmd_count_after=pl.col("count").max(),
            is_finished=pl.col("at").max()
            >= pl.col("last_uplink_at").min() + UPLINK_WINDOW,
        )
        .with_columns(
            window_start=pl.col("uplink_finished_at") - UPLINK_WINDOW,
            previous_finished_at=pl.col("uplink_finished_at").shift(1),
        )
        .sort("window_start")
    )

    before = rows.select(
        "at",
        before_at="at",
        before_boot_at="boot_at",
        before_count="count",
    )
    passes = passes.join_asof(
        before,
        left_on="window_start",
        right_on="at",
        strategy="backward",
        allow_exact_matches=False,
    )

    usable_before = (
        ((pl.col("before_boot_at") - pl.col("boot_at")).abs() <= SAME_BOOT_TOLERANCE)
        & (
            pl.col("previous_finished_at").is_null()
            | (pl.col("before_at") > pl.col("previous_finished_at"))
        )
        & (pl.col("before_count") <= pl.col("tcmd_count_after"))
    )
    count_before = (
        pl.when(pl.col("boot_at") >= pl.col("window_start"))
        .then(0)
        .when(usable_before)
        .then(pl.col("before_count"))
    )
    return (
        passes.filter("is_finished")
        .with_columns(tcmd_count_before=count_before)
        .with_columns(
            tcmd_count_delta=pl.col("tcmd_count_after") - pl.col("tcmd_count_before")
        )
        .select(pl.col(name).cast(dtype) for name, dtype in _OUTPUT_SCHEMA.items())
        .sort("uplink_finished_at", descending=True)
    )


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
def _finished_uplinks_since(path: Path, since: datetime) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    beacons = (
        pl.scan_parquet(path)
        .filter(
            pl.col("packet_type").is_in(BEACON_PACKET_TYPES)
            & pl.col("csp_crc_valid")
            & (pl.col("received_at") >= since)
        )
        .select(
            "utc_time",
            "uptime_sec",
            "total_tcmd_queued_count",
            "duration_since_last_uplink_ms",
        )
        .collect()
    )
    return detect_finished_uplinks(beacons)


def latest_finished_uplink(
    path: Path = DEFAULT_PARQUET_PATH, *, window: timedelta = timedelta(hours=24)
) -> FinishedUplink | None:
    """The newest row of `detect_finished_uplinks` over beacons received in
    the last `window`, or None if no pass finished in that time.
    """
    now = round_down_datetime(datetime.now(UTC), timedelta(minutes=5))
    uplinks = _finished_uplinks_since(path, now - window)
    if uplinks.is_empty():
        return None
    return FinishedUplink(**uplinks.row(0, named=True))


_PASS_TCMD_SCHEMA = {
    "tcmd_count_before": pl.Int64,
    "tcmd_count_after": pl.Int64,
    "tcmd_count_delta": pl.Int64,
}


@cached(TTLCache(maxsize=3, ttl=_CACHE_TTL_SECONDS))
def _load_beacon_tcmd_counts_since(path: Path, since: datetime) -> pl.DataFrame:
    schema = {
        "beacon_utc_time": pl.Datetime("us", "UTC"),
        "total_tcmd_queued_count": pl.Int64,
        "count_since": pl.Datetime("us", "UTC"),
    }
    if not path.exists():
        return pl.DataFrame(schema=schema)
    since_ms = int(since.timestamp() * 1000)
    epoch_ms = pl.col("unix_epoch_time_ms")
    count = pl.col("total_tcmd_queued_count").cast(pl.Int64)
    count_since_ms = (
        pl.when(count > 0)
        .then(epoch_ms - pl.col("duration_since_last_uplink_ms"))
        .otherwise(epoch_ms - (pl.col("uptime_sec") * 1000).cast(pl.Int64))
    )
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("packet_type").is_in(BEACON_PACKET_TYPES)
            & pl.col("csp_crc_valid")
            # Also drops beacons that aren't time-synced yet (epoch 0).
            & (epoch_ms >= since_ms)
            & (epoch_ms <= MAX_VALID_EPOCH_MS)
        )
        .select(
            beacon_utc_time=pl.from_epoch(epoch_ms, time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(schema["beacon_utc_time"]),
            total_tcmd_queued_count=count,
            count_since=pl.from_epoch(count_since_ms, time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(schema["count_since"]),
        )
        .drop_nulls(["beacon_utc_time", "total_tcmd_queued_count"])
        .sort("beacon_utc_time")
        .collect()
    )


def load_beacon_tcmd_counts(
    path: Path = DEFAULT_PARQUET_PATH, *, first_aos: datetime
) -> pl.DataFrame:
    """CRC-valid beacons' own clock time (`beacon_utc_time`, from
    `unix_epoch_time_ms` -- not `received_at`, which some decoders only know
    to the observation), `total_tcmd_queued_count`, and `count_since` (when
    that count last changed, on the same clock: the last uplink, or boot if
    the count is 0; null if unknown), oldest first --
    everything `tcmds_uplinked_per_pass` needs for passes from `first_aos`
    on. Only those columns (plus the ones filtered on) are read, and only
    from beacons timestamped in the last `PASS_TCMD_LOOKBACK` before
    `first_aos - PASS_TCMD_MARGIN` or since.
    """
    since = round_down_datetime(
        first_aos - PASS_TCMD_MARGIN - PASS_TCMD_LOOKBACK, timedelta(hours=1)
    )
    return _load_beacon_tcmd_counts_since(path, since)


def tcmds_uplinked_per_pass(
    beacon_counts: pl.DataFrame, passes: pl.DataFrame
) -> pl.DataFrame:
    """Telecommands uplinked during each pass, from the beacons around it.

    Args:
        beacon_counts: As from `load_beacon_tcmd_counts`.
        passes: One row per pass, with UTC datetime `aos` and `los` columns.

    Returns:
        `passes`' rows, in the same order, with:
        - `tcmd_count_before`: `total_tcmd_queued_count` from the last beacon
          timestamped before `aos - PASS_TCMD_MARGIN` -- or 0 if that's more
          than `tcmd_count_after`, since the count only drops when a reboot
          resets it.
        - `tcmd_count_after`: the count from the first beacon timestamped at
          or after `los + PASS_TCMD_MARGIN`.
        - `tcmd_count_delta`: after minus before.

        Each is null when there's no such beacon (e.g. the pass isn't over
        yet). Each beacon must also come from the gap between this pass and
        the neighbouring one (the previous pass's `los + PASS_TCMD_MARGIN`
        onwards, or before the next pass's `aos - PASS_TCMD_MARGIN`): with no
        beacon in that gap, the nearest one is from beyond the neighbour, and
        would count the neighbour's telecommands for both passes. The
        exceptions use a beacon's `count_since`, whichever side of the
        neighbour it was received on: its count is the count before if it
        already held at `aos - PASS_TCMD_MARGIN`, and the count after if it
        held from before the next pass's `aos - PASS_TCMD_MARGIN` -- an uplink
        between the two passes (e.g. a little past a predicted LOS) counts
        for the earlier one, as it would with a beacon in the gap.
    """
    counts = beacon_counts.select(
        pl.col("beacon_utc_time").cast(passes.schema["aos"]),
        count=pl.col("total_tcmd_queued_count"),
        count_since=pl.col("count_since").cast(passes.schema["aos"]),
    ).sort("beacon_utc_time")
    return (
        passes.with_row_index("_order")
        .with_columns(
            _before_at=pl.col("aos") - PASS_TCMD_MARGIN,
            _after_at=pl.col("los") + PASS_TCMD_MARGIN,
        )
        .sort("_before_at")
        .with_columns(
            _previous_after_at=pl.col("_after_at").shift(1),
            _next_before_at=pl.col("_before_at").shift(-1),
        )
        .join_asof(
            counts.select(
                _before_beacon_at="beacon_utc_time", tcmd_count_before="count"
            ),
            left_on="_before_at",
            right_on="_before_beacon_at",
            strategy="backward",
            allow_exact_matches=False,
        )
        # The first beacon from `_before_at` on: if its count already held
        # then, it's the count at `_before_at` exactly. (A later beacon's
        # count can only have changed more recently, so this is the only one
        # worth checking.)
        .join_asof(
            counts.select(
                _held_beacon_at="beacon_utc_time",
                _held_count="count",
                _held_since="count_since",
            ),
            left_on="_before_at",
            right_on="_held_beacon_at",
            strategy="forward",
        )
        .sort("_after_at")
        .join_asof(
            counts.select(
                _after_beacon_at="beacon_utc_time",
                tcmd_count_after="count",
                _after_since="count_since",
            ),
            left_on="_after_at",
            right_on="_after_beacon_at",
            strategy="forward",
        )
        .with_columns(
            tcmd_count_before=pl.when(pl.col("_held_since") <= pl.col("_before_at"))
            .then(pl.col("_held_count"))
            .when(
                pl.col("_previous_after_at").is_null()
                | (pl.col("_before_beacon_at") >= pl.col("_previous_after_at"))
            )
            .then(pl.col("tcmd_count_before")),
            tcmd_count_after=pl.when(
                pl.col("_next_before_at").is_null()
                | (pl.col("_after_beacon_at") < pl.col("_next_before_at"))
                | (pl.col("_after_since") < pl.col("_next_before_at"))
            ).then(pl.col("tcmd_count_after")),
        )
        .with_columns(
            tcmd_count_before=pl.when(
                pl.col("tcmd_count_before") > pl.col("tcmd_count_after")
            )
            .then(0)
            .otherwise(pl.col("tcmd_count_before"))
        )
        .with_columns(
            tcmd_count_delta=pl.col("tcmd_count_after") - pl.col("tcmd_count_before")
        )
        .with_columns(
            pl.col(name).cast(dtype) for name, dtype in _PASS_TCMD_SCHEMA.items()
        )
        .sort("_order")
        .drop(pl.selectors.starts_with("_"))
    )
