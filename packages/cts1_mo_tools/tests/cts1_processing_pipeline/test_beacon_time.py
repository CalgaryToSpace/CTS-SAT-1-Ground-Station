"""Beacon views are keyed on each beacon's own onboard clock (`utc_time`),
not on `received_at` -- see `data.BEACON_TIME_COLUMN`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.charts import (
    BEACON_CHART_GROUPS,
    chart_option,
    receive_delay_option,
    receive_delay_vs_time_since_sync_option,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.data import (
    ATTITUDE_COLUMNS,
    ATTITUDE_MODE_COLUMNS,
    BEACON_TIME_COLUMN,
    beacon_receive_delay_medians,
    latest_beacons,
    latest_friendly_message,
    latest_local_max_pending_tcmd_count,
    load_attitude_window,
    load_beacon_window,
    v5plus_receive_delay_vs_time_since_sync,
)

_NOW = datetime.now(UTC).replace(microsecond=0)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "everything_decoded.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _beacon(  # noqa: PLR0913
    name: str,
    *,
    beacon_time: datetime | None,
    received_at: datetime,
    packet_type: str = "BEACON_EXTENDED_V5PLUS",
    is_stored_beacon: bool | None = False,
    pending: int = 0,
    csp_crc_valid: bool = True,
) -> dict[str, object]:
    return {
        "name": name,
        "packet_type": packet_type,
        "received_at": received_at,
        "unix_epoch_time_ms": 0 if beacon_time is None else _ms(beacon_time),
        "is_stored_beacon": is_stored_beacon,
        "csp_crc_valid": csp_crc_valid,
        "pending_queued_tcmd_count": pending,
        **dict.fromkeys(ATTITUDE_COLUMNS, 1.0),
        **dict.fromkeys(ATTITUDE_MODE_COLUMNS, "1 - MEMS rate sensing"),
        "adcs_wheel_speed_x_rpm": 100,
    }


def test_window_uses_onboard_clock_and_includes_stored_beacons(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        [
            # Live beacon, received just now.
            _beacon(
                "live",
                beacon_time=_NOW - timedelta(minutes=1),
                received_at=_NOW,
            ),
            # Stored beacon recorded 5h ago, received just now (after "live").
            _beacon(
                "stored",
                beacon_time=_NOW - timedelta(hours=5),
                received_at=_NOW + timedelta(seconds=1),
                is_stored_beacon=True,
            ),
            # Stored beacon recorded 3 days ago: outside a 24h window, even
            # though it was received just now.
            _beacon(
                "too_old",
                beacon_time=_NOW - timedelta(days=3),
                received_at=_NOW,
                is_stored_beacon=True,
            ),
            # Not time-synced yet (epoch 0): can't be placed in time.
            _beacon("unsynced", beacon_time=None, received_at=_NOW),
            # Clock glitch far in the future.
            _beacon(
                "future",
                beacon_time=_NOW + timedelta(days=30),
                received_at=_NOW,
            ),
            # A v2-v4 extended beacon: also shown.
            _beacon(
                "v4",
                beacon_time=_NOW - timedelta(hours=2),
                received_at=_NOW - timedelta(hours=2),
                packet_type="BEACON_EXTENDED",
                is_stored_beacon=None,
            ),
        ],
    )

    df = load_beacon_window(path, since=_NOW - timedelta(hours=24))
    assert df["name"].to_list() == ["stored", "v4", "live"]
    assert df.schema[BEACON_TIME_COLUMN] == pl.Datetime("us", "UTC")
    assert df[BEACON_TIME_COLUMN][0] == _NOW - timedelta(hours=5)


def test_latest_beacon_is_newest_by_onboard_clock(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _beacon(
                "live",
                beacon_time=_NOW - timedelta(minutes=1),
                received_at=_NOW - timedelta(minutes=1),
            ),
            # Received later, but recorded hours earlier.
            _beacon(
                "stored",
                beacon_time=_NOW - timedelta(hours=5),
                received_at=_NOW,
                is_stored_beacon=True,
            ),
            _beacon(
                "future",
                beacon_time=_NOW + timedelta(days=30),
                received_at=_NOW,
            ),
        ],
    )
    assert latest_beacons(path, n=1)["name"].to_list() == ["live"]


def test_attitude_window_uses_onboard_clock(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _beacon("b", beacon_time=_NOW - timedelta(hours=1), received_at=_NOW),
            _beacon(
                "a",
                beacon_time=_NOW - timedelta(hours=2),
                received_at=_NOW + timedelta(seconds=1),
                is_stored_beacon=True,
            ),
        ],
    )
    frames = load_attitude_window(path, since=_NOW - timedelta(hours=3))
    assert frames[BEACON_TIME_COLUMN].to_list() == [
        _NOW - timedelta(hours=2),
        _NOW - timedelta(hours=1),
    ]


def test_local_max_pending_reports_onboard_time(tmp_path: Path) -> None:
    peak_time = _NOW - timedelta(hours=2)
    path = _write(
        tmp_path,
        [
            _beacon("a", beacon_time=_NOW - timedelta(hours=3), received_at=_NOW),
            _beacon("peak", beacon_time=peak_time, received_at=_NOW, pending=5),
            _beacon(
                "c",
                beacon_time=_NOW - timedelta(hours=1),
                received_at=_NOW - timedelta(days=1),
                pending=2,
            ),
        ],
    )
    assert latest_local_max_pending_tcmd_count(path) == (5, peak_time)


def test_v5_fields_charted() -> None:
    specs = {spec.title: spec for _, group in BEACON_CHART_GROUPS for spec in group}
    df = pl.DataFrame(
        {
            BEACON_TIME_COLUMN: [_NOW - timedelta(minutes=1), _NOW],
            "adcs_wheel_speed_x_rpm": [100, 120],
            "adcs_wheel_speed_y_rpm": [0, 0],
            "adcs_wheel_speed_z_rpm": [None, -5],
            "eps_pcu_ch4_volt_in_mppt_V": [5.0, 5.1],
        }
    )
    option = chart_option(specs["Reaction Wheel Speeds"], df)
    assert option is not None
    assert option["series"][0]["data"][0] == [
        _ms(_NOW - timedelta(minutes=1)),
        100,
    ]

    option = chart_option(specs["MPPT Input Voltage"], df)
    assert option is not None
    assert [s["name"] for s in option["series"]] == ["ch4"]


def test_every_v5_field_has_a_chart() -> None:
    charted = {c for _, group in BEACON_CHART_GROUPS for s in group for c in s.columns}
    v5_fields = {
        "distinct_tcmds_pending",
        "duration_since_last_time_sync_sec",
        "adcs_wheel_speed_x_rpm",
        "adcs_wheel_speed_y_rpm",
        "adcs_wheel_speed_z_rpm",
        "adcs_pointing_error_deg",
        "adcs_innovation_vector_x",
        "adcs_innovation_vector_y",
        "adcs_innovation_vector_z",
        "adcs_estimated_gyro_bias_x_deg_per_sec",
        "adcs_estimated_gyro_bias_y_deg_per_sec",
        "adcs_estimated_gyro_bias_z_deg_per_sec",
        "gnss_power_consumed_W",
        "adcs_total_power_consumed_W",
        "eps_pcu_ch4_volt_in_mppt_V",
        "eps_pcu_ch4_curr_in_mppt_A",
        "eps_pcu_ch4_curr_ou_mppt_A",
        "bulk_downlink_file_path_sha256_prefix",
        "is_stored_beacon",
    }
    assert v5_fields <= charted


def test_window_dedupes_by_onboard_time_preferring_live_copy(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=1)
    path = _write(
        tmp_path,
        [
            # Replayed copy received first (e.g. a later pass got the live one
            # backfilled from another station's slower upload).
            _beacon(
                "stored_copy",
                beacon_time=t,
                received_at=_NOW - timedelta(minutes=5),
                is_stored_beacon=True,
            ),
            _beacon("live_copy", beacon_time=t, received_at=_NOW),
            _beacon(
                "other",
                beacon_time=t + timedelta(seconds=9),
                received_at=_NOW,
            ),
        ],
    )
    df = load_beacon_window(path, since=_NOW - timedelta(hours=2))
    assert df["name"].to_list() == ["live_copy", "other"]

    frames = load_attitude_window(path, since=_NOW - timedelta(hours=2))
    assert frames.height == 2


def test_dedupe_prefers_crc_valid_copy(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=1)
    path = _write(
        tmp_path,
        [
            _beacon(
                "live_bad_crc", beacon_time=t, received_at=_NOW, csp_crc_valid=False
            ),
            _beacon(
                "stored_ok", beacon_time=t, received_at=_NOW, is_stored_beacon=True
            ),
        ],
    )
    df = load_beacon_window(path, since=_NOW - timedelta(hours=2))
    assert df["name"].to_list() == ["stored_ok"]


def test_dedupe_prefers_earliest_received_among_equals(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=1)
    path = _write(
        tmp_path,
        [
            _beacon("later", beacon_time=t, received_at=_NOW),
            _beacon("earlier", beacon_time=t, received_at=_NOW - timedelta(minutes=1)),
        ],
    )
    df = load_beacon_window(path, since=_NOW - timedelta(hours=2))
    assert df["name"].to_list() == ["earlier"]


def _with_message(row: dict[str, object], message: str | None) -> dict[str, object]:
    return {**row, "friendly_message": message}


def test_friendly_message_from_the_beacon_itself(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=1)
    path = _write(
        tmp_path,
        [
            _with_message(
                _beacon("old", beacon_time=t - timedelta(minutes=5), received_at=_NOW),
                "older",
            ),
            _with_message(
                _beacon(
                    "b", beacon_time=t, received_at=_NOW, packet_type="BEACON_BASIC"
                ),
                "hello",
            ),
        ],
    )
    assert latest_friendly_message(path, at=t) == ("hello", t)


def test_friendly_message_filled_forward_up_to_30_minutes(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=1)
    basic_time = t - timedelta(minutes=20)
    path = _write(
        tmp_path,
        [
            _with_message(
                _beacon(
                    "basic",
                    beacon_time=basic_time,
                    received_at=_NOW,
                    packet_type="BEACON_BASIC",
                ),
                "hello",
            ),
            # The displayed v5 beacon: no friendly message.
            _with_message(_beacon("v5", beacon_time=t, received_at=_NOW), None),
            # Newer than the displayed beacon: not "filled forward" onto it.
            _with_message(
                _beacon(
                    "newer",
                    beacon_time=t + timedelta(minutes=1),
                    received_at=_NOW,
                    packet_type="BEACON_BASIC",
                ),
                "from the future",
            ),
        ],
    )
    assert latest_friendly_message(path, at=t) == ("hello", basic_time)


def test_friendly_message_not_filled_forward_past_30_minutes(tmp_path: Path) -> None:
    basic_time = _NOW - timedelta(hours=1)
    path = _write(
        tmp_path,
        [
            _with_message(
                _beacon(
                    "basic",
                    beacon_time=basic_time,
                    received_at=_NOW,
                    packet_type="BEACON_BASIC",
                ),
                "hello",
            ),
        ],
    )
    at = basic_time + timedelta(minutes=29)
    assert latest_friendly_message(path, at=at) == ("hello", basic_time)
    assert latest_friendly_message(path, at=basic_time + timedelta(minutes=31)) is None


def test_receive_delay_medians_per_window_and_sync_source(tmp_path: Path) -> None:
    hour = (_NOW - timedelta(hours=3)).replace(minute=0, second=0)

    def _synced(row: dict[str, object], source: str | None) -> dict[str, object]:
        return {**row, "last_time_sync_source": source}

    path = _write(
        tmp_path,
        [
            # First hour, GNSS: delays 2s, 4s, 600s (an outlier) -> median 4s.
            *(
                _synced(
                    _beacon(
                        f"gnss-{i}",
                        beacon_time=hour + timedelta(minutes=i),
                        received_at=hour + timedelta(minutes=i, seconds=delay),
                    ),
                    "GNSS_UART",
                )
                for i, delay in enumerate((2, 4, 600))
            ),
            # Same hour, after a switch to the EPS RTC: its own median.
            _synced(
                _beacon(
                    "rtc",
                    beacon_time=hour + timedelta(minutes=40),
                    received_at=hour + timedelta(minutes=40, seconds=-9),
                ),
                "EPS_RTC",
            ),
            # Second hour, no source: received 1s *before* its onboard time.
            _synced(
                _beacon(
                    "unknown",
                    beacon_time=hour + timedelta(hours=1, minutes=5),
                    received_at=hour + timedelta(hours=1, minutes=5, seconds=-1),
                ),
                None,
            ),
            # Stored: recorded in the first hour, received much later -- ignored.
            _synced(
                _beacon(
                    "stored",
                    beacon_time=hour + timedelta(minutes=30),
                    received_at=_NOW,
                    is_stored_beacon=True,
                ),
                "GNSS_UART",
            ),
        ],
    )
    beacons = load_beacon_window(path, since=_NOW - timedelta(hours=24))
    medians = beacon_receive_delay_medians(beacons, every=timedelta(hours=1))
    assert medians.rows() == [
        (hour, "EPS_RTC", -9.0, 1),
        (hour, "GNSS_UART", 4.0, 3),
        (hour + timedelta(hours=1), "UNKNOWN", -1.0, 1),
    ]

    option = receive_delay_option(medians, title="delay")
    assert option is not None
    # Firmware order (GNSS_UART before EPS_RTC), one series per source.
    assert [(s["name"], s["data"]) for s in option["series"]] == [
        ("GNSS_UART", [[_ms(hour), 4.0, 3]]),
        ("EPS_RTC", [[_ms(hour), -9.0, 1]]),
        ("UNKNOWN", [[_ms(hour + timedelta(hours=1)), -1.0, 1]]),
    ]


def test_receive_delay_option_none_without_live_beacons() -> None:
    medians = beacon_receive_delay_medians(pl.DataFrame(), every=timedelta(hours=1))
    assert receive_delay_option(medians, title="delay") is None


def test_v5plus_receive_delay_vs_time_since_sync(tmp_path: Path) -> None:
    t = _NOW - timedelta(hours=2)

    def _row(  # noqa: PLR0913
        name: str,
        *,
        minutes: int,
        delay: int,
        since_sync: int | None,
        mhz: int | None = 25,
        packet_type: str = "BEACON_EXTENDED_V5PLUS",
        is_stored_beacon: bool | None = False,
    ) -> dict[str, object]:
        return {
            **_beacon(
                name,
                beacon_time=t + timedelta(minutes=minutes),
                received_at=t + timedelta(minutes=minutes, seconds=delay),
                packet_type=packet_type,
                is_stored_beacon=is_stored_beacon,
            ),
            "last_time_sync_source": "EPS_RTC",
            "obc_active_oscillator_MHz": mhz,
            "duration_since_last_time_sync_sec": since_sync,
        }

    path = _write(
        tmp_path,
        [
            _row("a", minutes=0, delay=-5, since_sync=3600),
            _row("b", minutes=1, delay=-6, since_sync=3660, mhz=None),
            # Left out: at the clamp, stored, and not v5+.
            _row("clamped", minutes=2, delay=-60, since_sync=65535),
            _row("stored", minutes=3, delay=999, since_sync=100, is_stored_beacon=True),
            _row(
                "v4",
                minutes=4,
                delay=1,
                since_sync=None,
                packet_type="BEACON_EXTENDED",
                is_stored_beacon=None,
            ),
        ],
    )
    beacons = load_beacon_window(path, since=_NOW - timedelta(hours=24))
    points = v5plus_receive_delay_vs_time_since_sync(beacons)
    assert points.drop(BEACON_TIME_COLUMN).rows() == [
        ("EPS_RTC", 25, 3600, -5.0),
        ("EPS_RTC", None, 3660, -6.0),
    ]

    option = receive_delay_vs_time_since_sync_option(points, title="drift")
    assert option is not None
    assert [(s["name"], s["data"]) for s in option["series"]] == [
        ("EPS_RTC @ 25 MHz", [[1.0, -5.0, _ms(t), 3600]]),
        (
            "EPS_RTC @ ? MHz",
            [
                [
                    round(3660 / 3600, 4),
                    -6.0,
                    _ms(t + timedelta(minutes=1)),
                    3660,
                ]
            ],
        ),
    ]
    empty = v5plus_receive_delay_vs_time_since_sync(pl.DataFrame())
    assert receive_delay_vs_time_since_sync_option(empty, title="drift") is None
