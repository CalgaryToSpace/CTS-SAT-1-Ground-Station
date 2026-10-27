from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.uplinks import (
    PASS_TCMD_MARGIN,
    detect_finished_uplinks,
    tcmds_uplinked_per_pass,
)

_BASE = datetime(2026, 1, 1, tzinfo=UTC)

# Where `_beacon`'s nonzero counts came from by default: an uplink long
# before any test's pass, itself reported as a (much older) finished pass.
_PREVIOUS_UPLINK = -500


def _at(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _beacon(
    minutes: float,
    count: int,
    last_uplink_minutes: float | None = _PREVIOUS_UPLINK,
    *,
    boot_minutes: float = -10_000,
) -> dict[str, Any]:
    """A beacon at `minutes`, `count` telecommands since a boot at
    `boot_minutes`, the latest uplinked at `last_uplink_minutes`.
    """
    at = _at(minutes)
    since_uplink_ms = (
        # The satellite reports a meaningless duration before any uplink.
        123_456
        if count == 0 or last_uplink_minutes is None
        else round((minutes - last_uplink_minutes) * 60_000)
    )
    return {
        "utc_time": at.isoformat(timespec="milliseconds"),
        "uptime_sec": (minutes - boot_minutes) * 60,
        "total_tcmd_queued_count": count,
        "duration_since_last_uplink_ms": since_uplink_ms,
    }


def _since_start(result: pl.DataFrame) -> pl.DataFrame:
    """`result` without the `_PREVIOUS_UPLINK` pass."""
    return result.filter(pl.col("uplink_finished_at") > _at(_PREVIOUS_UPLINK))


def test_empty_input_returns_empty_with_expected_schema() -> None:
    result = detect_finished_uplinks(pl.DataFrame())
    assert result.height == 0
    assert result.schema["uplink_finished_at"] == pl.Datetime("us", "UTC")
    assert result.schema["tcmd_count_delta"] == pl.Int64


def test_compares_counts_from_before_and_after_the_window() -> None:
    """Mid-pass beacons only see part of the pass's uplinks, so the count
    before comes from a beacon received at least 15 minutes before the last
    uplink -- not the mid-pass beacon's 15.
    """
    rows = [
        _beacon(-10, 10),
        # Inside the window: a beacon from before the pass's first uplink can
        # still be this close to its last one.
        _beacon(2, 10),
        # Pass: uplinks from ~t=5 to t=12, one beacon caught partway through.
        _beacon(8, 15, last_uplink_minutes=7.5),
        _beacon(13, 30, last_uplink_minutes=12),
        _beacon(40, 30, last_uplink_minutes=12),
    ]
    result = _since_start(detect_finished_uplinks(pl.DataFrame(rows)))
    assert result.to_dicts() == [
        {
            "uplink_finished_at": _at(12),
            "tcmd_count_before": 10,
            "tcmd_count_after": 30,
            "tcmd_count_delta": 20,
        }
    ]


def test_input_order_does_not_matter() -> None:
    rows = [
        _beacon(40, 30, last_uplink_minutes=12),
        _beacon(-10, 10),
        _beacon(13, 30, last_uplink_minutes=12),
    ]
    result = _since_start(detect_finished_uplinks(pl.DataFrame(rows)))
    assert result["tcmd_count_delta"].to_list() == [20]


def test_pass_still_in_progress_is_not_finished() -> None:
    """No beacon yet from 15 minutes after the first uplink seen, so more
    uplinks could still be coming in.
    """
    rows = [
        _beacon(0, 10),
        _beacon(21, 15, last_uplink_minutes=20),
        _beacon(30, 18, last_uplink_minutes=29),
    ]
    result = _since_start(detect_finished_uplinks(pl.DataFrame(rows)))
    assert result.is_empty()


def test_multiple_passes_newest_first() -> None:
    rows = [
        _beacon(-10, 10),
        _beacon(13, 30, last_uplink_minutes=12),
        _beacon(60, 30, last_uplink_minutes=12),
        _beacon(105, 33, last_uplink_minutes=104),
        _beacon(130, 33, last_uplink_minutes=104),
    ]
    result = detect_finished_uplinks(pl.DataFrame(rows))
    assert result["uplink_finished_at"].to_list() == [
        _at(104),
        _at(12),
        _at(_PREVIOUS_UPLINK),
    ]
    # Nothing from before the oldest pass to compare against.
    assert result["tcmd_count_delta"].to_list() == [3, 20, None]


def test_clock_resync_jitter_is_one_pass() -> None:
    """The same uplink, seen a couple of seconds apart across an onboard
    clock resync, is still one pass.
    """
    rows = [
        _beacon(-10, 10),
        _beacon(13, 30, last_uplink_minutes=12),
        _beacon(40, 30, last_uplink_minutes=12 + 2 / 60),
    ]
    result = _since_start(detect_finished_uplinks(pl.DataFrame(rows)))
    assert result["uplink_finished_at"].to_list() == [_at(12)]
    assert result["tcmd_count_delta"].to_list() == [20]


def test_no_beacon_before_the_window_gives_null_delta() -> None:
    rows = [
        _beacon(13, 30, last_uplink_minutes=12),
        _beacon(40, 30, last_uplink_minutes=12),
    ]
    result = detect_finished_uplinks(pl.DataFrame(rows))
    assert result["tcmd_count_after"].item() == 30
    assert result["tcmd_count_before"].item() is None
    assert result["tcmd_count_delta"].item() is None


def test_reboot_within_the_window_counts_from_zero() -> None:
    rows = [
        _beacon(-30, 500),
        _beacon(6, 0, boot_minutes=5),
        _beacon(13, 7, last_uplink_minutes=12, boot_minutes=5),
        _beacon(40, 7, last_uplink_minutes=12, boot_minutes=5),
    ]
    result = detect_finished_uplinks(pl.DataFrame(rows))
    assert result["tcmd_count_before"].to_list() == [0, None]
    assert result["tcmd_count_delta"].to_list() == [7, None]


def test_zero_count_beacons_before_the_window_count_as_zero() -> None:
    """After a reboot, the count before a pass is 0 -- and those beacons'
    `duration_since_last_uplink_ms` isn't an uplink time at all.
    """
    rows = [
        _beacon(-60, 0, boot_minutes=-70),
        _beacon(-10, 0, boot_minutes=-70),
        _beacon(13, 7, last_uplink_minutes=12, boot_minutes=-70),
        _beacon(40, 7, last_uplink_minutes=12, boot_minutes=-70),
    ]
    result = detect_finished_uplinks(pl.DataFrame(rows))
    assert result.to_dicts() == [
        {
            "uplink_finished_at": _at(12),
            "tcmd_count_before": 0,
            "tcmd_count_after": 7,
            "tcmd_count_delta": 7,
        }
    ]


def test_before_beacon_from_a_previous_boot_gives_null_delta() -> None:
    """A reboot between the last beacon before the window and the window
    itself: uplinks could have come in since the reboot, so the count before
    the window is unknown.
    """
    rows = [
        _beacon(-60, 500),
        _beacon(13, 7, last_uplink_minutes=12, boot_minutes=-40),
        _beacon(40, 7, last_uplink_minutes=12, boot_minutes=-40),
    ]
    result = detect_finished_uplinks(pl.DataFrame(rows))
    assert result["uplink_finished_at"].to_list()[0] == _at(12)
    assert result["tcmd_count_delta"].to_list()[0] is None


def _counts(*beacons: tuple[float, int] | tuple[float, int, float]) -> pl.DataFrame:
    """Beacon `(beacon_utc_time minutes, total_tcmd_queued_count)` pairs,
    optionally with a third `count_since` minutes (null if left out).
    """
    return pl.DataFrame(
        {
            "beacon_utc_time": [_at(b[0]) for b in beacons],
            "total_tcmd_queued_count": [b[1] for b in beacons],
            "count_since": [_at(b[2]) if len(b) > 2 else None for b in beacons],
            "pending_queued_tcmd_count": [None] * len(beacons),
        },
        schema={
            "beacon_utc_time": pl.Datetime("us", "UTC"),
            "total_tcmd_queued_count": pl.Int64,
            "count_since": pl.Datetime("us", "UTC"),
            "pending_queued_tcmd_count": pl.Int64,
        },
    )


def _passes(*aos_los_minutes: tuple[float, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "aos": [_at(aos) for aos, _ in aos_los_minutes],
            "los": [_at(los) for _, los in aos_los_minutes],
        },
        schema={"aos": pl.Datetime("us", "UTC"), "los": pl.Datetime("us", "UTC")},
    )


def test_tcmds_per_pass_uses_beacons_outside_margin() -> None:
    margin = PASS_TCMD_MARGIN.total_seconds() / 60
    counts = _counts(
        (-20, 10),
        (-margin - 0.1, 11),  # Last one more than the margin before AOS.
        (-margin + 0.1, 12),  # Within the margin: ignored.
        (5, 14),  # Mid-pass: ignored.
        (10 + margin - 0.1, 18),  # Within the margin after LOS: ignored.
        (10 + margin, 20),  # First one at least the margin after LOS.
        (30, 25),
    )
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10)))
    assert result.select(
        "tcmd_count_before", "tcmd_count_after", "tcmd_count_delta"
    ).row(0) == (11, 20, 9)


def test_tcmds_per_pass_null_without_beacons_on_either_side() -> None:
    counts = _counts((2, 10), (30, 12), (50, 15))
    # Order preserved: the first pass has no beacon before, the last none after.
    result = tcmds_uplinked_per_pass(counts, _passes((100, 110), (2, 12), (40, 44)))
    assert result["tcmd_count_delta"].to_list() == [None, None, 3]


def test_tcmds_per_pass_counts_from_zero_after_reboot() -> None:
    # The count reset to 0 during the pass, then 3 more came in.
    counts = _counts((-10, 40), (20, 3))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10)))
    assert result.select(
        "tcmd_count_before", "tcmd_count_after", "tcmd_count_delta"
    ).row(0) == (0, 3, 3)


def test_tcmds_per_pass_not_shared_across_beaconless_gap() -> None:
    counts = _counts((-10, 10), (20, 12), (80, 12), (200, 17))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    assert result["tcmd_count_delta"].to_list() == [2, 5]
    # No beacon between the two passes (from 15 to 85): either side's nearest
    # beacon is from beyond the other pass, so neither gets a count -- rather
    # than both getting the same 7.
    gapless = _counts((-10, 10), (200, 17))
    result = tcmds_uplinked_per_pass(gapless, _passes((0, 10), (90, 100)))
    assert result["tcmd_count_delta"].to_list() == [None, None]


def test_tcmds_per_pass_shared_gap_beacon_counts_once() -> None:
    # One beacon in the gap is "after" for the first pass and "before" for the
    # second.
    counts = _counts((-10, 10), (50, 12), (200, 17))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    assert result["tcmd_count_delta"].to_list() == [2, 5]


def test_tcmds_per_pass_count_since_spans_beaconless_gap() -> None:
    # No beacon between the passes, but the one after the second shows its
    # count unchanged since minute 10.5 -- within the first pass's margin -- so
    # the first pass uplinked all 7 and the second none.
    counts = _counts((-10, 10), (200, 17, 10.5))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    assert result.select("tcmd_count_before", "tcmd_count_after").rows() == [
        (10, 17),
        (17, 17),
    ]


def test_tcmds_per_pass_count_since_after_the_instant_is_not_used() -> None:
    # The count last changed at minute 95, during the second pass: it's no
    # help for either side of the gap.
    counts = _counts((-10, 10), (200, 17, 95))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    assert result.select("tcmd_count_before", "tcmd_count_after").rows() == [
        (10, None),
        (None, 17),
    ]


def test_tcmds_per_pass_count_since_late_uplink_counts_for_earlier_pass() -> None:
    # The last uplink came in at minute 20, well past the first pass's
    # LOS + margin but long before the second pass: it's the first pass's.
    counts = _counts((-10, 10), (200, 17, 20))
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    assert result.select("tcmd_count_before", "tcmd_count_after").rows() == [
        (10, 17),
        (17, 17),
    ]


def test_tcmds_per_pass_pending_counts_from_gap_beacons_only() -> None:
    counts = _counts((-10, 10), (200, 17, 20)).with_columns(
        pending_queued_tcmd_count=pl.Series([4, 1])
    )
    result = tcmds_uplinked_per_pass(counts, _passes((0, 10), (90, 100)))
    # The minute-200 beacon's total held since minute 20, but its pending
    # count says nothing about minute 20 -- so the first pass gets no pending
    # after, nor the second a pending before.
    assert result.select(
        "pending_tcmd_count_before",
        "pending_tcmd_count_after",
        "pending_tcmd_count_delta",
    ).rows() == [(4, None, None), (None, 1, None)]
    one_pass = tcmds_uplinked_per_pass(counts, _passes((0, 10)))
    assert one_pass.select(
        "pending_tcmd_count_before",
        "pending_tcmd_count_after",
        "pending_tcmd_count_delta",
    ).row(0) == (4, 1, -3)
