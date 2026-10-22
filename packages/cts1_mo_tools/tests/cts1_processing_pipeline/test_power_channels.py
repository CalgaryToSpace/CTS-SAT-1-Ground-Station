from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_decode_satnogs_packets import ADCS_ENABLED_BITS
from cts1_mo_tools.cts1_processing_pipeline.web_ui.charts import (
    channel_states_option,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.data import (
    ADCS_POWER_CONTROL_CHANNELS,
    adcs_channel_states,
    adcs_error_states,
    eps_channel_states,
)

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _beacons(bitfields: list[str | None], adcs_lists: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "received_at": [
                _BASE + timedelta(minutes=i) for i in range(len(bitfields))
            ],
            "eps_enabled_channels_bitfield": bitfields,
            "adcs_powered_list": adcs_lists,
        }
    )


def test_eps_channel_states_decodes_bitfield() -> None:
    # Stack channels (bits 0, 1, 5) + 3V3_GNSS (bit 8).
    states = eps_channel_states(_beacons(["0x00000123"], [None]))
    row = states.row(0, named=True)
    assert row["VBATT_STACK"]
    assert row["5V_STACK"]
    assert row["3V3_STACK"]
    assert row["3V3_GNSS"]
    assert not row["12V_BOOM"]
    assert not row["5V_MPI"]


def test_eps_channel_states_hides_unused_channels_unless_on() -> None:
    states = eps_channel_states(_beacons(["0x00000001", "0x00000002"], [None, None]))
    assert not any(c.endswith("_UNUSED") for c in states.columns)

    # Bit 2 is 5V_CH2_UNUSED.
    states = eps_channel_states(_beacons(["0x00000001", "0x00000004"], [None, None]))
    assert "5V_CH2_UNUSED" in states.columns
    assert "5V_CH3_UNUSED" not in states.columns


def test_eps_channel_states_skips_rows_without_bitfield() -> None:
    states = eps_channel_states(_beacons([None, "0x00000001"], [None, None]))
    assert states.height == 1
    assert states["received_at"][0] == _BASE + timedelta(minutes=1)


def test_adcs_channel_states_extended_rows_only() -> None:
    states = adcs_channel_states(
        _beacons(
            ["0x00000001", "0x00000001", "0x00000001"],
            [None, '["CUBEWHEEL1","GPS_RECEIVER"]', "[]"],
        )
    )
    assert states.height == 2
    first, second = states.iter_rows(named=True)
    assert first["CUBEWHEEL1"]
    assert first["GPS_RECEIVER"]
    assert not first["CUBEWHEEL2"]
    assert not any(v for k, v in second.items() if k != "received_at")


def test_channel_states_option_splits_on_and_off() -> None:
    states = eps_channel_states(_beacons(["0x00000001", "0x00000000"], [None, None]))
    option = channel_states_option(states, title="EPS")
    assert option is not None
    on, off = option["series"]
    channels = [c for c in states.columns if c != "received_at"]
    assert option["yAxis"]["data"] == channels[::-1]
    assert on["data"] == [[_BASE.isoformat(), "VBATT_STACK"]]
    assert len(off["data"]) == 2 * len(channels) - 1


def test_channel_states_option_empty() -> None:
    states = eps_channel_states(_beacons([None], [None]))
    assert channel_states_option(states, title="EPS") is None


def test_adcs_power_control_channels_are_decoded_names() -> None:
    decoded = {name for _, name in ADCS_ENABLED_BITS}
    assert set(ADCS_POWER_CONTROL_CHANNELS) <= decoded
    assert "GPS_LNA_POWER" not in ADCS_POWER_CONTROL_CHANNELS


def test_adcs_channel_states_in_telecommand_order() -> None:
    states = adcs_channel_states(_beacons(["0x00000001"], ['["GPS_LNA_POWER"]']))
    assert states.columns == ["received_at", *ADCS_POWER_CONTROL_CHANNELS]


def test_adcs_error_states_keeps_only_seen_in_bit_order() -> None:
    beacons = pl.DataFrame(
        {
            "received_at": [_BASE, _BASE + timedelta(minutes=1), _BASE],
            "adcs_errors": [
                '["COARSE_SUN_SENSOR_ERROR","CAM1_SENSOR_BUSY_ERROR"]',
                "[]",
                None,
            ],
            "adcs_flags": ['["CAM1_3V3_OVERCURRENT"]', "[]", None],
        }
    )
    states = adcs_error_states(beacons)
    # Bits 34, 35, 45: the overcurrent flag sorts in among the errors.
    assert states.columns == [
        "received_at",
        "CAM1_3V3_OVERCURRENT",
        "CAM1_SENSOR_BUSY_ERROR",
        "COARSE_SUN_SENSOR_ERROR",
    ]
    assert states.height == 2
    assert all(states.row(0)[1:])
    assert not any(states.row(1)[1:])


def test_channel_states_option_alert_uses_set_clear() -> None:
    states = eps_channel_states(_beacons(["0x00000001"], [None]))
    option = channel_states_option(states, title="x", alert=True)
    assert option is not None
    assert [s["name"] for s in option["series"]] == ["set", "clear"]
