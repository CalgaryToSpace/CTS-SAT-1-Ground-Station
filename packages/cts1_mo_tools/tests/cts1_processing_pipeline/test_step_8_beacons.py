from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    beacons,
)

_START = datetime(2026, 11, 14, 8, 33, 30, tzinfo=UTC)
_END = _START + timedelta(minutes=2)
_COLLECTIONS = pl.DataFrame(
    {"collection_started_at": [_START], "collection_ended_at": [_END]}
)


def _ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def _beacon(  # noqa: PLR0913
    at: datetime,
    *,
    packet_type: str = "BEACON_EXTENDED_V5PLUS",
    pitch: float | None = 0.0,
    yaw: float | None = 0.0,
    rate: float | None = 0.2,
    received_at: datetime = _END,
    is_stored_beacon: bool = False,
) -> dict[str, object]:
    extended = packet_type != "BEACON_BASIC"
    return {
        "unix_epoch_time_ms": _ms(at),
        "packet_type": packet_type,
        "adcs_attitude_estimation_mode": "6 - MEMS gyro EKF" if extended else None,
        "adcs_control_mode": "5 - XYZ-Wheel control" if extended else None,
        "adcs_estimated_roll_angle_deg": 0.0 if extended else None,
        "adcs_estimated_pitch_angle_deg": pitch if extended else None,
        "adcs_estimated_yaw_angle_deg": yaw if extended else None,
        "adcs_angular_rate_norm_deg_per_sec": rate if extended else None,
        "obc_active_oscillator_MHz": 25 if extended else None,
        "duration_since_last_time_sync_sec": 2777 + (at - _START).seconds,
        "last_time_sync_source": "EPS_RTC",
        "received_at": received_at,
        "is_stored_beacon": is_stored_beacon,
        "csp_crc_valid": True,
    }


def _summary(*rows: dict[str, object]) -> dict[str, object]:
    frame = pl.DataFrame(list(rows)).drop(
        "received_at", "is_stored_beacon", "csp_crc_valid"
    )
    (row,) = beacons.with_beacon_summary(_COLLECTIONS, frame).rows(named=True)
    return row


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-86.0, 60.0, 0.0], (-86.0, 60.0, 146.0)),
        ([170.0, -170.0, 179.0], (170.0, -170.0, 20.0)),  # Wraps through 180.
        ([5.0], (5.0, 5.0, 0.0)),
        ([], None),
    ],
)
def test_angle_range_is_the_smallest_arc(
    values: list[float], expected: tuple[float, float, float] | None
) -> None:
    assert beacons.angle_range(values) == expected


def test_summary_counts_beacons_in_the_window_only() -> None:
    margin = beacons.BEACON_WINDOW_MARGIN
    row = _summary(
        _beacon(_START - margin),  # Just in.
        _beacon(_START + timedelta(seconds=30), packet_type="BEACON_BASIC"),
        _beacon(_END + margin),  # Just in.
        _beacon(_END + margin + timedelta(seconds=1)),  # Out.
    )
    assert row["beacon_count"] == 3
    assert row["beacon_extended_count"] == 2
    assert row["beacon_adcs_estimation_modes"] == '["6 - MEMS gyro EKF"]'
    assert row["beacon_adcs_control_modes"] == '["5 - XYZ-Wheel control"]'
    assert row["beacon_oscillator_mhz"] == "[25]"
    assert row["beacon_time_sync_sources"] == '["EPS_RTC"]'
    assert row["beacon_slew"] == "no"


def test_summary_flags_a_slew_by_attitude_change_or_rate() -> None:
    swing = _summary(_beacon(_START, pitch=-86.0), _beacon(_END, pitch=60.0))
    assert swing["beacon_slew"] == "yes"
    assert swing["beacon_pitch_min_deg"] == -86.0
    assert swing["beacon_pitch_max_deg"] == 60.0
    assert swing["beacon_max_attitude_change_deg"] == 146.0

    fast = _summary(_beacon(_START, rate=beacons.SLEW_RATE_DEG_PER_SEC + 0.1))
    assert fast["beacon_slew"] == "yes"

    # Yaw wrapping through 180 is a small swing, not a slew.
    wrap = _summary(_beacon(_START, yaw=179.0), _beacon(_END, yaw=-179.0))
    assert wrap["beacon_max_attitude_change_deg"] == 2.0
    assert wrap["beacon_slew"] == "no"


def test_summary_with_no_beacons_is_zero_counts_and_nulls() -> None:
    far = _beacon(_END + timedelta(hours=1))
    row = _summary(far)
    assert (row["beacon_count"], row["beacon_extended_count"]) == (0, 0)
    assert row["beacon_slew"] is None
    assert row["beacon_adcs_control_modes"] is None


def test_load_beacons_reads_each_beacon_once_around_collections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "everything_decoded.parquet"
    pl.DataFrame(
        [
            _beacon(_START, received_at=_END, is_stored_beacon=True),
            _beacon(_START, received_at=_END + timedelta(hours=1)),  # Same beacon.
            _beacon(_START + timedelta(hours=1)),  # Not around the collection.
            {**_beacon(_START), "packet_type": "TCMD_RESPONSE"},  # Not a beacon.
        ]
    ).write_parquet(path)
    loaded = beacons.load_beacons(path, _COLLECTIONS)
    assert loaded.height == 1
    assert loaded["unix_epoch_time_ms"].to_list() == [_ms(_START)]
