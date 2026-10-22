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
    "UPLINK_WINDOW",
    "FinishedUplink",
    "detect_finished_uplinks",
    "latest_finished_uplink",
]

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
from cachetools import TTLCache, cached

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
