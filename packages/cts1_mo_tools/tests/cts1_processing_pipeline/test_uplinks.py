from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.uplinks import (
    detect_finished_uplinks,
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
