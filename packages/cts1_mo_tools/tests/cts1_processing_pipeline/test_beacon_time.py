"""Beacon views are keyed on each beacon's own onboard clock (`utc_time`),
not on `received_at` -- see `data.BEACON_TIME_COLUMN`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.charts import (
    BEACON_CHART_GROUPS,
    chart_option,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.data import (
    ATTITUDE_COLUMNS,
    ATTITUDE_MODE_COLUMNS,
    BEACON_TIME_COLUMN,
    latest_beacons,
    latest_local_max_pending_tcmd_count,
    load_attitude_window,
    load_beacon_window,
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
        (_NOW - timedelta(minutes=1)).isoformat(),
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
