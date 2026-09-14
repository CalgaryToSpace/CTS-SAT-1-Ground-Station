from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_4_detect_satellite_events.pipeline import (  # noqa: E501
    compute_satellite_events,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_BASE_EPOCH_MS = int(_BASE.timestamp() * 1000)


def _epoch_ms(seconds: float) -> int:
    return _BASE_EPOCH_MS + round(seconds * 1000)


def _at(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


def _beacon(  # noqa: PLR0913
    *,
    seconds: float,
    packet_type: str = "BEACON_BASIC",
    uptime_ms: int,
    eps_uptime_sec: int,
    duration_since_last_uplink_ms: int,
    reboot_reason: str = "NORMAL",
    eps_reset_cause: str = "NORMAL",
    csp_crc_valid: bool = True,
    unix_epoch_time_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "unix_epoch_time_ms": (
            unix_epoch_time_ms if unix_epoch_time_ms is not None else _epoch_ms(seconds)
        ),
        "packet_type": packet_type,
        "uptime_ms": uptime_ms,
        "eps_uptime_sec": eps_uptime_sec,
        "duration_since_last_uplink_ms": duration_since_last_uplink_ms,
        "reboot_reason": reboot_reason,
        "eps_reset_cause": eps_reset_cause,
        "csp_crc_valid": csp_crc_valid,
    }


def _df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_empty_input_returns_empty_with_expected_schema() -> None:
    result = compute_satellite_events(pl.DataFrame())
    assert result.height == 0
    assert result.schema["event_type"] == pl.String
    assert result.schema["detected_at"] == pl.Datetime("us", "UTC")
    assert result.schema["time_since_event_when_detected_ms"] == pl.Int64


def test_no_beacons_returns_no_events() -> None:
    df = pl.DataFrame(
        [
            {
                "unix_epoch_time_ms": _epoch_ms(0),
                "packet_type": "LOG_MESSAGE",
                "uptime_ms": None,
                "eps_uptime_sec": None,
                "duration_since_last_uplink_ms": None,
                "reboot_reason": None,
                "eps_reset_cause": None,
                "csp_crc_valid": True,
            }
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_obc_reboot_detected_on_uptime_drop() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=10_000_000,  # well over RESYNC_TOLERANCE -- a real drop
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=30,
                uptime_ms=5_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=35_000,
                reboot_reason="WATCHDOG",
            ),
        ]
    )
    events = compute_satellite_events(df)
    obc = events.filter(pl.col("event_type") == "OBC Reboot")
    assert obc.height == 1
    row = obc.row(0, named=True)
    assert row["detected_at"] == _at(30)
    assert row["estimated_event_at"] == _at(25)
    assert row["time_since_event_when_detected_ms"] == 5_000
    assert row["obc_reboot_reason"] == "WATCHDOG"


def test_eps_reboot_detected_on_eps_uptime_drop() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100_000,  # well over RESYNC_TOLERANCE -- a real drop
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=90,
                uptime_ms=190_000,
                eps_uptime_sec=5,
                duration_since_last_uplink_ms=95_000,
                eps_reset_cause="BROWNOUT",
            ),
        ]
    )
    events = compute_satellite_events(df)
    eps = events.filter(pl.col("event_type") == "EPS Reboot")
    assert eps.height == 1
    row = eps.row(0, named=True)
    assert row["detected_at"] == _at(90)
    assert row["estimated_event_at"] == _at(85)
    assert row["time_since_event_when_detected_ms"] == 5_000
    assert row["eps_reboot_reason"] == "BROWNOUT"


def test_uplink_detected_on_duration_since_last_uplink_drop() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=10_000_000,  # over RESYNC_TOLERANCE
            ),
            _beacon(
                seconds=120,
                uptime_ms=220_000,
                eps_uptime_sec=220,
                duration_since_last_uplink_ms=1_200,
            ),
        ]
    )
    events = compute_satellite_events(df)
    uplink = events.filter(pl.col("event_type") == "Uplinked Commands")
    assert uplink.height == 1
    row = uplink.row(0, named=True)
    assert row["detected_at"] == _at(120)
    assert row["estimated_event_at"] == _at(118) + timedelta(milliseconds=800)
    assert row["time_since_event_when_detected_ms"] == 1_200


def test_no_drop_means_no_event() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=30,
                uptime_ms=130_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=35_000,
            ),
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_first_beacon_never_an_event() -> None:
    # A single beacon has nothing to compare against -- shouldn't crash or
    # be reported as every counter "dropping" from null.
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=5_000,
                eps_uptime_sec=5,
                duration_since_last_uplink_ms=1_000,
            )
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_basic_and_extended_beacons_both_considered_in_one_timeline() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                packet_type="BEACON_EXTENDED",
                uptime_ms=10_000_000,  # well over RESYNC_TOLERANCE -- a real drop
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=30,
                packet_type="BEACON_BASIC",
                uptime_ms=5_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=35_000,
            ),
        ]
    )
    events = compute_satellite_events(df)
    assert events.height == 1
    assert events.row(0, named=True)["detecting_packet_type"] == "BEACON_BASIC"


def test_crc_invalid_beacon_ignored_entirely() -> None:
    # A bit-flipped beacon that still fails its own CSP CRC shouldn't be
    # trusted for event detection -- its counters could be garbage -- and
    # shouldn't even count as "the previous beacon" for the next real one.
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=15,
                uptime_ms=7,  # garbage counter from a corrupted decode
                eps_uptime_sec=999_999,
                duration_since_last_uplink_ms=1,
                csp_crc_valid=False,
            ),
            _beacon(
                seconds=30,
                uptime_ms=130_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=35_000,
            ),
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_invalid_epoch_beacon_ignored_entirely() -> None:
    # unix_epoch_time_ms == 0 means "not yet time-synced" (same convention
    # as the decoder's own utc_time == None) -- shouldn't be trusted for
    # ordering/event detection, and shouldn't count as "the previous
    # beacon" either.
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=15,
                uptime_ms=1,
                eps_uptime_sec=1,
                duration_since_last_uplink_ms=1,
                unix_epoch_time_ms=0,
            ),
            _beacon(
                seconds=30,
                uptime_ms=130_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=35_000,
            ),
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_small_clock_resync_does_not_look_like_a_reboot() -> None:
    # A small onboard time-sync correction (the beacon's own UTC clock
    # nudged back ~10s) can locally swap two beacons' sort order without
    # any real event happening -- uptime keeps climbing normally the whole
    # time. Both the apparent counter "drop" and the beacon-to-beacon time
    # gap are tiny (well under RESYNC_TOLERANCE), so this must not be
    # reported as an OBC reboot from ages "ago".
    df = _df(
        [
            _beacon(
                seconds=100,
                uptime_ms=5_000_000,
                eps_uptime_sec=5_000,
                duration_since_last_uplink_ms=1_000,
            ),
            _beacon(
                # Resynced ~10s backward relative to the beacon above, even
                # though it was received ~10s *after* it (uptime is 10s
                # higher, consistent with normal progression).
                unix_epoch_time_ms=_epoch_ms(90),
                seconds=90,
                uptime_ms=5_010_000,
                eps_uptime_sec=5_010,
                duration_since_last_uplink_ms=11_000,
            ),
        ]
    )
    assert compute_satellite_events(df).height == 0


def test_large_drop_still_detected_despite_small_time_gap() -> None:
    # A genuine reset can happen between two beacons received close
    # together -- the small time gap alone shouldn't suppress it, only a
    # small time gap *and* a small counter drop together should.
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=10_000_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=5,  # well under RESYNC_TOLERANCE
                uptime_ms=2_000,  # but the drop itself is huge -- still real
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_005_000,
                reboot_reason="WATCHDOG",
            ),
        ]
    )
    events = compute_satellite_events(df)
    obc = events.filter(pl.col("event_type") == "OBC Reboot")
    assert obc.height == 1
    assert obc.row(0, named=True)["obc_reboot_reason"] == "WATCHDOG"


def test_ordering_uses_unix_epoch_time_ms_not_row_order() -> None:
    # Rows deliberately inserted out of chronological order (mirroring a
    # low-trust decoder reporting an observation's end time for every
    # frame in it, which can scramble `received_at` order) -- detection
    # must still follow the satellite's own clock, not row order.
    reboot = _beacon(
        seconds=30,
        uptime_ms=5_000,
        eps_uptime_sec=130,
        duration_since_last_uplink_ms=35_000,
        reboot_reason="WATCHDOG",
    )
    before = _beacon(
        seconds=0,
        uptime_ms=10_000_000,  # well over RESYNC_TOLERANCE -- a real drop
        eps_uptime_sec=100,
        duration_since_last_uplink_ms=5_000,
    )
    df = _df([reboot, before])  # inserted newest-first
    events = compute_satellite_events(df)
    obc = events.filter(pl.col("event_type") == "OBC Reboot")
    assert obc.height == 1
    assert obc.row(0, named=True)["detected_at"] == _at(30)


def test_events_sorted_newest_first() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=10_000_000,  # drops to beacon 1's -- real OBC reboot
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=95_000,
            ),
            _beacon(
                seconds=30,
                uptime_ms=5_000,
                eps_uptime_sec=100_000,  # drops to beacon 2's -- real EPS reboot
                duration_since_last_uplink_ms=10_000_000,  # drops -- real uplink
            ),
            _beacon(
                seconds=60,
                uptime_ms=35_000,
                eps_uptime_sec=5,
                duration_since_last_uplink_ms=1_000,
            ),
        ]
    )
    events = compute_satellite_events(df)
    detected_ats = events["detected_at"].to_list()
    assert detected_ats == sorted(detected_ats, reverse=True)
