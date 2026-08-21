from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_4_detect_satellite_events.pipeline import (  # noqa: E501
    compute_satellite_events,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _beacon(  # noqa: PLR0913
    *,
    seconds: float,
    packet_type: str = "BEACON_BASIC",
    uptime_ms: int,
    eps_uptime_sec: int,
    duration_since_last_uplink_ms: int,
    reboot_reason: str = "NORMAL",
    eps_reset_cause: str = "NORMAL",
    eps_total_fault_count: int = 0,
) -> dict:
    return {
        "received_at": _BASE + timedelta(seconds=seconds),
        "packet_type": packet_type,
        "uptime_ms": uptime_ms,
        "eps_uptime_sec": eps_uptime_sec,
        "duration_since_last_uplink_ms": duration_since_last_uplink_ms,
        "reboot_reason": reboot_reason,
        "eps_reset_cause": eps_reset_cause,
        "eps_total_fault_count": eps_total_fault_count,
    }


def _df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema_overrides={"received_at": pl.Datetime("us", "UTC")}
    )


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
                "received_at": _BASE,
                "packet_type": "LOG_MESSAGE",
                "uptime_ms": None,
                "eps_uptime_sec": None,
                "duration_since_last_uplink_ms": None,
                "reboot_reason": None,
                "eps_reset_cause": None,
                "eps_total_fault_count": None,
            }
        ],
        schema_overrides={"received_at": pl.Datetime("us", "UTC")},
    )
    assert compute_satellite_events(df).height == 0


def test_obc_reboot_detected_on_uptime_drop() -> None:
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
    assert row["detected_at"] == _BASE + timedelta(seconds=30)
    assert row["estimated_event_at"] == _BASE + timedelta(seconds=25)
    assert row["time_since_event_when_detected_ms"] == 5_000
    assert row["obc_reboot_reason"] == "WATCHDOG"


def test_eps_reboot_detected_on_eps_uptime_drop() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=5_000,
            ),
            _beacon(
                seconds=90,
                uptime_ms=190_000,
                eps_uptime_sec=5,
                duration_since_last_uplink_ms=95_000,
                eps_reset_cause="BROWNOUT",
                eps_total_fault_count=3,
            ),
        ]
    )
    events = compute_satellite_events(df)
    eps = events.filter(pl.col("event_type") == "EPS Reboot")
    assert eps.height == 1
    row = eps.row(0, named=True)
    assert row["detected_at"] == _BASE + timedelta(seconds=90)
    assert row["estimated_event_at"] == _BASE + timedelta(seconds=85)
    assert row["time_since_event_when_detected_ms"] == 5_000
    assert row["eps_reboot_reason"] == "BROWNOUT"
    assert row["eps_reset_count"] == 3


def test_uplink_detected_on_duration_since_last_uplink_drop() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=95_000,
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
    assert row["detected_at"] == _BASE + timedelta(seconds=120)
    assert row["estimated_event_at"] == _BASE + timedelta(seconds=118, milliseconds=800)
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
                uptime_ms=100_000,
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


def test_events_sorted_newest_first() -> None:
    df = _df(
        [
            _beacon(
                seconds=0,
                uptime_ms=100_000,
                eps_uptime_sec=100,
                duration_since_last_uplink_ms=95_000,
            ),
            _beacon(
                seconds=30,
                uptime_ms=5_000,
                eps_uptime_sec=130,
                duration_since_last_uplink_ms=125_000,
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
