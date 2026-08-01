import json
import struct
from typing import Any

import pytest
from cts1_mo_tools.cts1_decode_satnogs_packets import (
    ADCS_ENABLED_BITS,
    ADCS_ERROR_BITS,
    ADCS_FLAG_BITS,
    BEACON_EXTENDED_TOTAL_STRUCT_SIZE,
    BEACON_FIXED_SIZE,
    BEACON_TOTAL_STRUCT_SIZE,
    BULK_DOWNLINK_HEADER_FMT,
    BULK_DOWNLINK_HEADER_SIZE,
    CSP_HEADER_SIZE,
    END_MESSAGE_SIZE,
    EXTENDED_FMT,
    FIXED_FMT,
    FRIENDLY_MESSAGE_SIZE,
    PACKET_TYPE_MAP,
    PACKET_TYPE_MAP_INV,
    TCMD_RESPONSE_HEADER_FMT,
    TCMD_RESPONSE_HEADER_SIZE,
    decode_adcs_current_state_1,
    decode_beacon_basic_packet,
    decode_beacon_extended_packet,
    decode_beacon_peripheral_packet,
    decode_bulk_file_downlink_packet,
    decode_log_message_packet,
    decode_packet_safe,
    decode_tcmd_response_packet,
    e,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_CSP = b"\xaa\xbb\xcc\xdd"  # 4-byte CSP header (content irrelevant to tests)


def _make_beacon_payload(  # noqa: PLR0913
    *,
    packet_type: int = 0x01,
    satellite_name: bytes = b"CTS1",
    active_rf_switch_antenna: int = 1,
    active_rf_switch_control_mode: int = 0,
    uptime_ms: int = 5_000,
    duration_since_last_uplink_ms: int = 1_000,
    unix_epoch_time_ms: int = 1_700_000_000_000,
    last_time_sync_source_enum: int = 1,
    is_fs_mounted: int = 1,
    total_tcmd_queued_count: int = 10,
    pending_queued_tcmd_count: int = 2,
    total_beacon_count_since_boot: int = 42,
    eps_mode_enum: int = 1,
    eps_reset_cause_enum: int = 0,
    eps_uptime_sec: int = 4,
    eps_error_code: int = 0,
    eps_battery_voltage_mV: int = 4000,  # noqa: N803
    eps_battery_percent: int = 80,
    eps_battery_temperature_0_cC: int = 2500,  # noqa: N803
    eps_battery_temperature_1_cC: int = 2600,  # noqa: N803
    eps_total_fault_count: int = 0,
    eps_enabled_channels_bitfield: int = 0xFFFFFFFF,
    eps_total_pcu_power_input_cW: int = 500,  # noqa: N803
    eps_total_pcu_power_output_cW: int = 450,  # noqa: N803
    eps_total_avg_pcu_power_input_cW: int = 490,  # noqa: N803
    eps_total_avg_pcu_power_output_cW: int = 440,  # noqa: N803
    obc_temperature_cC: int = 2000,  # noqa: N803
    reboot_reason: int = 4,
    cts1_operation_state: int = 2,
    rbf_pin_state: int = 1,
    mpi_rx_mode_enum: int = 0,
    mpi_transceiver_state_enum: int = 0,
    mpi_last_reason_for_stopping_enum: int = 0,
    gnss_uart_interrupt_enabled: int = 1,
    gnss_rx_mode_enum: int = 0,
    friendly_message: bytes = b"Hello from CTS-SAT-1",
    end_sentinel: bytes = b"END\x00",
) -> bytes:
    """Build a valid BEACON_BASIC payload (no CSP header)."""
    fixed = struct.pack(
        FIXED_FMT,
        packet_type,
        satellite_name,
        active_rf_switch_antenna,
        active_rf_switch_control_mode,
        uptime_ms,
        duration_since_last_uplink_ms,
        unix_epoch_time_ms,
        last_time_sync_source_enum,
        is_fs_mounted,
        total_tcmd_queued_count,
        pending_queued_tcmd_count,
        total_beacon_count_since_boot,
        eps_mode_enum,
        eps_reset_cause_enum,
        eps_uptime_sec,
        eps_error_code,
        eps_battery_voltage_mV,
        eps_battery_percent,
        eps_battery_temperature_0_cC,
        eps_battery_temperature_1_cC,
        eps_total_fault_count,
        eps_enabled_channels_bitfield,
        eps_total_pcu_power_input_cW,
        eps_total_pcu_power_output_cW,
        eps_total_avg_pcu_power_input_cW,
        eps_total_avg_pcu_power_output_cW,
        obc_temperature_cC,
        reboot_reason,
        cts1_operation_state,
        rbf_pin_state,
        mpi_rx_mode_enum,
        mpi_transceiver_state_enum,
        mpi_last_reason_for_stopping_enum,
        gnss_uart_interrupt_enabled,
        gnss_rx_mode_enum,
    )
    fm = friendly_message.ljust(FRIENDLY_MESSAGE_SIZE, b"\x00")[:FRIENDLY_MESSAGE_SIZE]
    return fixed + fm + end_sentinel


def _make_extended_payload(  # noqa: PLR0913
    *,
    packet_type: int = 0x20,
    friendly_message: bytes = b"Hello from CTS-SAT-1",
    end_version_number: bytes = b" X2\x00",
    obc_active_oscillator_MHz: int = 25,  # noqa: N803
    obc_adc_battery_voltage_mV: int = 7400,  # noqa: N803
    mpi_last_temperature_C: int = -99,  # noqa: N803
    eps_pcu_ch0_volt_in_mppt_mV: int = 5000,  # noqa: N803
    eps_pcu_ch0_curr_in_mppt_mA: int = 100,  # noqa: N803
    eps_pcu_ch0_curr_ou_mppt_mA: int = 90,  # noqa: N803
    eps_pcu_ch1_volt_in_mppt_mV: int = 5000,  # noqa: N803
    eps_pcu_ch1_curr_in_mppt_mA: int = 100,  # noqa: N803
    eps_pcu_ch1_curr_ou_mppt_mA: int = 90,  # noqa: N803
    eps_pcu_ch2_volt_in_mppt_mV: int = 5000,  # noqa: N803
    eps_pcu_ch2_curr_in_mppt_mA: int = 100,  # noqa: N803
    eps_pcu_ch2_curr_ou_mppt_mA: int = 90,  # noqa: N803
    eps_pcu_ch3_volt_in_mppt_mV: int = 5000,  # noqa: N803
    eps_pcu_ch3_curr_in_mppt_mA: int = 100,  # noqa: N803
    eps_pcu_ch3_curr_ou_mppt_mA: int = 90,  # noqa: N803
    eps_battery_pack_status_bitfield: int = 0x8000,
    eps_total_avg_net_battery_power_cW: int = -50,  # noqa: N803
    eps_total_avg_power_distributed_cW: int = 300,  # noqa: N803
    adcs_current_state_1: bytes = b"\x01\x02\x03\x04\x05\x06",
    adcs_raw_css_1: int = 10,
    adcs_raw_css_2: int = 20,
    adcs_raw_css_3: int = 30,
    adcs_raw_css_4: int = 40,
    adcs_raw_css_5: int = 50,
    adcs_raw_css_6: int = 60,
    adcs_raw_css_7: int = 70,
    adcs_raw_css_9: int = 90,
    adcs_magnetic_field_x_T_en8: int = 100,  # noqa: N803
    adcs_magnetic_field_y_T_en8: int = -100,  # noqa: N803
    adcs_magnetic_field_z_T_en8: int = 200,  # noqa: N803
    adcs_angular_rate_norm_cdeg_per_sec: int = 150,
    adcs_estimated_rate_x_cdeg_per_sec: int = 10,
    adcs_estimated_rate_y_cdeg_per_sec: int = -10,
    adcs_estimated_rate_z_cdeg_per_sec: int = 20,
    adcs_estimated_roll_angle_cdeg: int = 500,
    adcs_estimated_pitch_angle_cdeg: int = -500,
    adcs_estimated_yaw_angle_cdeg: int = 1000,
    **beacon_kwargs: Any,
) -> bytes:
    """Build a valid BEACON_EXTENDED payload (no CSP header)."""
    basic = _make_beacon_payload(
        packet_type=packet_type,
        friendly_message=friendly_message,
        end_sentinel=end_version_number,
        **beacon_kwargs,
    )
    extended = struct.pack(
        EXTENDED_FMT,
        obc_active_oscillator_MHz,
        obc_adc_battery_voltage_mV,
        mpi_last_temperature_C,
        eps_pcu_ch0_volt_in_mppt_mV,
        eps_pcu_ch0_curr_in_mppt_mA,
        eps_pcu_ch0_curr_ou_mppt_mA,
        eps_pcu_ch1_volt_in_mppt_mV,
        eps_pcu_ch1_curr_in_mppt_mA,
        eps_pcu_ch1_curr_ou_mppt_mA,
        eps_pcu_ch2_volt_in_mppt_mV,
        eps_pcu_ch2_curr_in_mppt_mA,
        eps_pcu_ch2_curr_ou_mppt_mA,
        eps_pcu_ch3_volt_in_mppt_mV,
        eps_pcu_ch3_curr_in_mppt_mA,
        eps_pcu_ch3_curr_ou_mppt_mA,
        eps_battery_pack_status_bitfield,
        eps_total_avg_net_battery_power_cW,
        eps_total_avg_power_distributed_cW,
        adcs_current_state_1,
        adcs_raw_css_1,
        adcs_raw_css_2,
        adcs_raw_css_3,
        adcs_raw_css_4,
        adcs_raw_css_5,
        adcs_raw_css_6,
        adcs_raw_css_7,
        adcs_raw_css_9,
        adcs_magnetic_field_x_T_en8,
        adcs_magnetic_field_y_T_en8,
        adcs_magnetic_field_z_T_en8,
        adcs_angular_rate_norm_cdeg_per_sec,
        adcs_estimated_rate_x_cdeg_per_sec,
        adcs_estimated_rate_y_cdeg_per_sec,
        adcs_estimated_rate_z_cdeg_per_sec,
        adcs_estimated_roll_angle_cdeg,
        adcs_estimated_pitch_angle_cdeg,
        adcs_estimated_yaw_angle_cdeg,
    )
    return basic + extended


def _make_tcmd_payload(  # noqa: PLR0913
    *,
    packet_type: int = 0x04,
    ts_sent: int = 12345678,
    response_code: int = 0,
    duration_ms: int = 50,
    response_seq_num: int = 0,
    response_max_seq_num: int = 0,
    data: bytes = b"OK",
) -> bytes:
    header = struct.pack(
        TCMD_RESPONSE_HEADER_FMT,
        packet_type,
        ts_sent,
        response_code,
        duration_ms,
        response_seq_num,
        response_max_seq_num,
    )
    return header + data


def _make_bulk_payload(
    *,
    packet_type: int = 0x10,
    file_offset: int = 0,
    data: bytes = b"\xde\xad\xbe\xef",
) -> bytes:
    header = struct.pack(BULK_DOWNLINK_HEADER_FMT, packet_type, file_offset)
    return header + data


def _make_log_payload(*, message: bytes = b"System boot complete") -> bytes:
    return bytes([0x03]) + message + b"\x00"


# ---------------------------------------------------------------------------
# Tests for the `e()` helper
# ---------------------------------------------------------------------------


class TestEnumHelper:
    def test_known_value(self) -> None:
        assert e({1: "ONE"}, 1) == "ONE"

    def test_unknown_value(self) -> None:
        assert e({1: "ONE"}, 99) == "UNKNOWN(99)"

    def test_zero_known(self) -> None:
        assert e({0: "ZERO"}, 0) == "ZERO"


# ---------------------------------------------------------------------------
# Tests for decode_beacon_basic_packet
# ---------------------------------------------------------------------------


class TestDecodeBeaconBasic:
    def _valid(self, **kwargs: Any) -> dict[str, Any]:
        return decode_beacon_basic_packet(_make_beacon_payload(**kwargs))

    def test_packet_type_label(self) -> None:
        result = self._valid()
        assert result["packet_type"] == "BEACON_BASIC"

    def test_satellite_name(self) -> None:
        result = self._valid(satellite_name=b"CTS1")
        assert result["satellite_name"] == "CTS1"

    def test_uptime_conversion(self) -> None:
        result = self._valid(uptime_ms=90_000)
        assert result["uptime_ms"] == 90_000
        assert result["uptime_sec"] == 90.0

    def test_battery_voltage_conversion(self) -> None:
        result = self._valid(eps_battery_voltage_mV=3700)
        assert result["eps_battery_voltage_V"] == (3.7)

    def test_obc_temperature_conversion(self) -> None:
        result = self._valid(obc_temperature_cC=2550)
        assert result["obc_temperature_C"] == (25.5)

    def test_battery_temperature_signed_negative(self) -> None:
        # -10 °C → -1000 cC (signed 16-bit)
        result = self._valid(eps_battery_temperature_0_cC=-1000)
        assert result["eps_battery_temperature_0_C"] == (-10.0)

    def test_power_conversion(self) -> None:
        result = self._valid(eps_total_pcu_power_input_cW=1234)
        assert result["eps_total_pcu_power_input_W"] == (12.34)

    def test_eps_enabled_channels_hex_format(self) -> None:
        result = self._valid(eps_enabled_channels_bitfield=0xDEADBEEF)
        assert result["eps_enabled_channels_bitfield"] == "0xDEADBEEF"

    def test_is_fs_mounted_bool(self) -> None:
        assert self._valid(is_fs_mounted=1)["is_fs_mounted"] is True
        assert self._valid(is_fs_mounted=0)["is_fs_mounted"] is False

    def test_gnss_uart_interrupt_enabled_bool(self) -> None:
        assert (
            self._valid(gnss_uart_interrupt_enabled=1)["gnss_uart_interrupt_enabled"]
            is True
        )

    def test_friendly_message_decoded(self) -> None:
        result = self._valid(friendly_message=b"Hello world")
        assert result["friendly_message"] == "Hello world"

    def test_friendly_message_null_terminated(self) -> None:
        # Only the part before the first null byte should appear.
        result = self._valid(friendly_message=b"Hello\x00garbage")
        assert result["friendly_message"] == "Hello"

    def test_enum_fields_resolved(self) -> None:
        result = self._valid(
            active_rf_switch_control_mode=1,
            last_time_sync_source_enum=2,
            eps_mode_enum=1,
            eps_reset_cause_enum=0,
            reboot_reason=4,
            cts1_operation_state=2,
            rbf_pin_state=1,
            mpi_rx_mode_enum=0,
            mpi_transceiver_state_enum=0,
            mpi_last_reason_for_stopping_enum=0,
            gnss_rx_mode_enum=0,
        )
        assert result["active_rf_switch_control_mode"] == "FORCE_ANT1"
        assert result["last_time_sync_source"] == "GNSS_PPS"
        assert result["eps_mode"] == "NOMINAL"
        assert result["eps_reset_cause"] == "POWER_ON"
        assert result["reboot_reason"] == "SOFTWARE_RESET"
        assert result["cts1_operation_state"] == "NOMINAL_WITH_RADIO_TX"
        assert result["rbf_pin_state"] == "FLYING"
        assert result["mpi_rx_mode"] == "COMMAND_MODE"
        assert result["mpi_transceiver_state"] == "INACTIVE"
        assert result["mpi_last_reason_for_stopping"] == "NOT_SET"
        assert result["gnss_rx_mode"] == "COMMAND_MODE"

    def test_utc_time_non_zero(self) -> None:
        # 1_700_000_000_000 ms → a recognisable ISO timestamp
        result = self._valid(unix_epoch_time_ms=1_700_000_000_000)
        assert result["utc_time"] is not None
        assert "2023" in result["utc_time"]

    def test_utc_time_zero(self) -> None:
        result = self._valid(unix_epoch_time_ms=0)
        assert result["utc_time"] is None

    def test_too_short_raises(self) -> None:
        short = b"\x01" * (BEACON_TOTAL_STRUCT_SIZE - 1)
        with pytest.raises(ValueError, match="Too short"):
            decode_beacon_basic_packet(short)

    def test_wrong_packet_type_raises(self) -> None:
        payload = _make_beacon_payload(packet_type=0x02)
        with pytest.raises(ValueError, match="Unexpected packet_type"):
            decode_beacon_basic_packet(payload)

    def test_total_struct_size_constant(self) -> None:
        # Sanity: our builder should produce exactly BEACON_TOTAL_STRUCT_SIZE bytes.
        assert len(_make_beacon_payload()) == BEACON_TOTAL_STRUCT_SIZE

    def test_extra_bytes_ignored(self) -> None:
        # Extra trailing bytes should not cause an error.
        payload = _make_beacon_payload() + b"\xff" * 10
        result = decode_beacon_basic_packet(payload)
        assert result["packet_type"] == "BEACON_BASIC"

    def test_unknown_enum_values(self) -> None:
        result = self._valid(eps_mode_enum=99)
        assert "UNKNOWN" in result["eps_mode"]


# ---------------------------------------------------------------------------
# Tests for decode_beacon_peripheral_packet
# ---------------------------------------------------------------------------


class TestDecodeBeaconPeripheral:
    def test_returns_peripheral_type(self) -> None:
        result = decode_beacon_peripheral_packet(b"\x02\xab\xcd\xef")
        assert result["packet_type"] == "BEACON_PERIPHERAL"

    def test_raw_hex_preserved(self) -> None:
        payload = b"\x02\x01\x02\x03"
        result = decode_beacon_peripheral_packet(payload)
        assert result["raw_payload_hex"] == payload.hex()

    def test_note_present(self) -> None:
        result = decode_beacon_peripheral_packet(b"\x02")
        assert "_note" in result

    def test_empty_payload(self) -> None:
        result = decode_beacon_peripheral_packet(b"")
        assert result["packet_type"] == "BEACON_PERIPHERAL"
        assert result["raw_payload_hex"] == ""


# ---------------------------------------------------------------------------
# Tests for decode_adcs_current_state_1
# ---------------------------------------------------------------------------


def _make_adcs_state_bytes(
    *,
    estim_mode: int = 0,
    control_mode: int = 0,
    run_mode: int = 0,
    asgp4_mode: int = 0,
    bit_indices_set: tuple[int, ...] = (),
) -> bytes:
    """Build a 6-byte ADCS Current State frame from field values + bit offsets."""
    value = estim_mode | (control_mode << 4) | (run_mode << 8) | (asgp4_mode << 10)
    for i in bit_indices_set:
        value |= 1 << i
    return value.to_bytes(6, "little")


class TestDecodeAdcsCurrentState1:
    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 6 bytes"):
            decode_adcs_current_state_1(b"\x00" * 5)

    def test_all_zero_modes(self) -> None:
        result = decode_adcs_current_state_1(_make_adcs_state_bytes())
        assert result["adcs_attitude_estimation_mode"] == "0 - No attitude estimation"
        assert result["adcs_control_mode"] == "0 - No control"
        assert result["adcs_run_mode"] == "0 - Off"
        assert result["adcs_asgp4_mode"] == "0 - Off"

    def test_max_valid_modes(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(
                estim_mode=7, control_mode=15, run_mode=3, asgp4_mode=3
            )
        )
        assert (
            result["adcs_attitude_estimation_mode"]
            == "7 - User Coded Estimation Mode"
        )
        assert (
            result["adcs_control_mode"]
            == "15 - Target-tracking yaw-only wheel control mode"
        )
        assert result["adcs_run_mode"] == "3 - Simulation"
        assert result["adcs_asgp4_mode"] == "3 - Augment"

    def test_enabled_json_list_single(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(12,))
        )
        assert json.loads(result["adcs_enabled"]) == ["CUBECONTROL_SIGNAL"]

    def test_enabled_json_list_multiple_preserves_order(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(22, 12, 19))
        )
        assert json.loads(result["adcs_enabled"]) == [
            "CUBECONTROL_SIGNAL",
            "CUBESTAR",
            "MOTOR_DRIVER",
        ]

    def test_enabled_json_list_empty_by_default(self) -> None:
        result = decode_adcs_current_state_1(_make_adcs_state_bytes())
        assert json.loads(result["adcs_enabled"]) == []

    def test_sun_above_local_horizon_is_separate_bool(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(23,))
        )
        assert result["adcs_sun_above_local_horizon"] is True
        assert json.loads(result["adcs_enabled"]) == []

    def test_errors_json_list_single(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(24,))
        )
        assert json.loads(result["adcs_errors"]) == ["CUBESENSE1_COMMS_ERROR"]
        assert json.loads(result["adcs_flags"]) == []

    def test_errors_json_list_multiple_preserves_order(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(46, 24, 32))
        )
        assert json.loads(result["adcs_errors"]) == [
            "CUBESENSE1_COMMS_ERROR",
            "MAGNETOMETER_RANGE_ERROR",
            "STAR_TRACKER_MATCH_ERROR",
        ]

    def test_flags_json_list(self) -> None:
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(33, 34, 47))
        )
        assert json.loads(result["adcs_flags"]) == [
            "CAM1_SRAM_OVERCURRENT",
            "CAM1_3V3_OVERCURRENT",
            "STAR_TRACKER_OVERCURRENT",
        ]
        assert json.loads(result["adcs_errors"]) == []

    def test_errors_and_flags_independent_of_enabled_bits(self) -> None:
        # Setting an error/flag bit should not affect unrelated enabled bits.
        result = decode_adcs_current_state_1(
            _make_adcs_state_bytes(bit_indices_set=(24, 33))
        )
        assert json.loads(result["adcs_enabled"]) == []
        assert json.loads(result["adcs_errors"]) == ["CUBESENSE1_COMMS_ERROR"]
        assert json.loads(result["adcs_flags"]) == ["CAM1_SRAM_OVERCURRENT"]

    def test_no_overlap_between_enabled_error_and_flag_bits(self) -> None:
        enabled_offsets = {i for i, _ in ADCS_ENABLED_BITS}
        error_offsets = {i for i, _ in ADCS_ERROR_BITS}
        flag_offsets = {i for i, _ in ADCS_FLAG_BITS}
        assert enabled_offsets.isdisjoint(error_offsets)
        assert enabled_offsets.isdisjoint(flag_offsets)
        assert error_offsets.isdisjoint(flag_offsets)

    def test_all_36_bit_offsets_accounted_for(self) -> None:
        enabled_offsets = {i for i, _ in ADCS_ENABLED_BITS}
        error_offsets = {i for i, _ in ADCS_ERROR_BITS}
        flag_offsets = {i for i, _ in ADCS_FLAG_BITS}
        sun_above_horizon_offset = {23}
        all_offsets = (
            enabled_offsets | error_offsets | flag_offsets | sun_above_horizon_offset
        )
        assert all_offsets == set(range(12, 48))
        assert len(all_offsets) == 36


# ---------------------------------------------------------------------------
# Tests for decode_beacon_extended_packet
# ---------------------------------------------------------------------------


class TestDecodeBeaconExtended:
    def _valid(self, **kwargs: Any) -> dict[str, Any]:
        return decode_beacon_extended_packet(_make_extended_payload(**kwargs))

    def test_packet_type_label(self) -> None:
        result = self._valid()
        assert result["packet_type"] == "BEACON_EXTENDED"

    def test_shared_fields_still_decoded(self) -> None:
        # Sanity check that the shared basic-beacon fields still come through.
        result = self._valid(satellite_name=b"CTS1", uptime_ms=90_000)
        assert result["satellite_name"] == "CTS1"
        assert result["uptime_sec"] == 90.0

    def test_end_version_number(self) -> None:
        result = self._valid(end_version_number=b" X2\x00")
        assert result["end_version_number"] == " X2"

    def test_obc_fields(self) -> None:
        result = self._valid(
            obc_active_oscillator_MHz=16, obc_adc_battery_voltage_mV=7500
        )
        assert result["obc_active_oscillator_MHz"] == 16
        assert result["obc_adc_battery_voltage_V"] == 7.5

    def test_mpi_last_temperature_inactive_passthrough(self) -> None:
        result = self._valid(mpi_last_temperature_C=-99)
        assert result["mpi_last_temperature_C"] == -99

    def test_eps_pcu_channel_conversion(self) -> None:
        result = self._valid(
            eps_pcu_ch0_volt_in_mppt_mV=5250,
            eps_pcu_ch0_curr_in_mppt_mA=1200,
            eps_pcu_ch0_curr_ou_mppt_mA=-800,
        )
        assert result["eps_pcu_ch0_volt_in_mppt_V"] == 5.25
        assert result["eps_pcu_ch0_curr_in_mppt_A"] == 1.2
        assert result["eps_pcu_ch0_curr_ou_mppt_A"] == -0.8

    def test_eps_battery_pack_status_bitfield_hex(self) -> None:
        result = self._valid(eps_battery_pack_status_bitfield=0x1234)
        assert result["eps_battery_pack_status_bitfield"] == "0x1234"

    def test_eps_avg_power_conversion(self) -> None:
        result = self._valid(
            eps_total_avg_net_battery_power_cW=-150,
            eps_total_avg_power_distributed_cW=250,
        )
        assert result["eps_total_avg_net_battery_power_W"] == -1.5
        assert result["eps_total_avg_power_distributed_W"] == 2.5

    def test_adcs_current_state_1_decoded_fields_merged_in(self) -> None:
        # bits: estim_mode=1, control_mode=2 -> byte0 = (2<<4)|1 = 0x21
        # run_mode=3, asgp4_mode=0 -> byte1 low nibble = 0x03
        result = self._valid(adcs_current_state_1=bytes([0x21, 0x03, 0, 0, 0, 0]))
        assert result["adcs_attitude_estimation_mode"] == "1 - MEMS rate sensing"
        assert result["adcs_control_mode"] == "2 - Y-Thomson spin"
        assert result["adcs_run_mode"] == "3 - Simulation"
        assert result["adcs_errors"] == "[]"
        assert result["adcs_flags"] == "[]"

    def test_adcs_raw_css_passthrough(self) -> None:
        result = self._valid(adcs_raw_css_1=12, adcs_raw_css_9=99)
        assert result["adcs_raw_css_1"] == 12
        assert result["adcs_raw_css_9"] == 99

    def test_adcs_magnetic_field_conversion(self) -> None:
        result = self._valid(
            adcs_magnetic_field_x_T_en8=1000,
            adcs_magnetic_field_y_T_en8=-500,
            adcs_magnetic_field_z_T_en8=0,
        )
        assert result["adcs_magnetic_field_x_uT"] == 10.0
        assert result["adcs_magnetic_field_y_uT"] == -5.0
        assert result["adcs_magnetic_field_z_uT"] == 0.0

    def test_adcs_angular_rate_norm_conversion(self) -> None:
        result = self._valid(adcs_angular_rate_norm_cdeg_per_sec=250)
        assert result["adcs_angular_rate_norm_deg_per_sec"] == 2.5

    def test_adcs_estimated_rate_conversion(self) -> None:
        result = self._valid(
            adcs_estimated_rate_x_cdeg_per_sec=100,
            adcs_estimated_rate_y_cdeg_per_sec=-100,
            adcs_estimated_rate_z_cdeg_per_sec=0,
        )
        assert result["adcs_estimated_rate_x_deg_per_sec"] == 1.0
        assert result["adcs_estimated_rate_y_deg_per_sec"] == -1.0
        assert result["adcs_estimated_rate_z_deg_per_sec"] == 0.0

    def test_adcs_estimated_angle_conversion(self) -> None:
        result = self._valid(
            adcs_estimated_roll_angle_cdeg=9000,
            adcs_estimated_pitch_angle_cdeg=-4500,
            adcs_estimated_yaw_angle_cdeg=18000,
        )
        assert result["adcs_estimated_roll_angle_deg"] == 90.0
        assert result["adcs_estimated_pitch_angle_deg"] == -45.0
        assert result["adcs_estimated_yaw_angle_deg"] == 180.0

    def test_too_short_raises(self) -> None:
        short = b"\x20" * (BEACON_EXTENDED_TOTAL_STRUCT_SIZE - 1)
        with pytest.raises(ValueError, match="Too short"):
            decode_beacon_extended_packet(short)

    def test_wrong_packet_type_raises(self) -> None:
        payload = _make_extended_payload(packet_type=0x01)
        with pytest.raises(ValueError, match="Unexpected packet_type"):
            decode_beacon_extended_packet(payload)

    def test_total_struct_size_constant(self) -> None:
        assert len(_make_extended_payload()) == BEACON_EXTENDED_TOTAL_STRUCT_SIZE

    def test_total_struct_size_is_198(self) -> None:
        # Matches the firmware blob's _Static_assert(sizeof(...) == 198).
        assert BEACON_EXTENDED_TOTAL_STRUCT_SIZE == 198

    def test_extra_bytes_ignored(self) -> None:
        payload = _make_extended_payload() + b"\xff" * 10
        result = decode_beacon_extended_packet(payload)
        assert result["packet_type"] == "BEACON_EXTENDED"

    def test_dispatch_via_decode_packet_safe(self) -> None:
        hex_str = (DUMMY_CSP + _make_extended_payload()).hex()
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "BEACON_EXTENDED"


# ---------------------------------------------------------------------------
# Tests for decode_log_message_packet
# ---------------------------------------------------------------------------


class TestDecodeLogMessage:
    def test_basic_message(self) -> None:
        result = decode_log_message_packet(b"\x03Hello, satellite!\n")
        assert result["packet_type"] == "LOG_MESSAGE"
        assert result["log_message"] == "Hello, satellite!"

    def test_no_null_terminator(self) -> None:
        # Should still decode without crashing.
        payload = b"\x03" + b"A" * 10
        result = decode_log_message_packet(payload)
        assert result["log_message"] == "A" * 10

    def test_empty_message(self) -> None:
        result = decode_log_message_packet(b"\x03\n")
        assert result["log_message"] == ""

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="Too short"):
            decode_log_message_packet(b"")

    def test_wrong_packet_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unexpected packet_type"):
            decode_log_message_packet(b"\x01Hello\n")

    def test_utf8_replacement_on_bad_bytes(self) -> None:
        payload = b"\x03\xff\xfe\x00"
        result = decode_log_message_packet(payload)
        # Should not raise; replacement character expected.
        assert isinstance(result["log_message"], str)


# ---------------------------------------------------------------------------
# Tests for decode_tcmd_response_packet
# ---------------------------------------------------------------------------


class TestDecodeTcmdResponse:
    def test_basic_fields(self) -> None:
        payload = _make_tcmd_payload(
            ts_sent=999,
            response_code=0,
            duration_ms=100,
            response_seq_num=1,
            response_max_seq_num=3,
            data=b"result\x00",
        )
        result = decode_tcmd_response_packet(payload)
        assert result["packet_type"] == "TCMD_RESPONSE"
        assert result["tcmd_ts_sent"] == 999
        assert result["tcmd_response_code"] == 0
        assert result["tcmd_duration_ms"] == 100
        assert result["tcmd_response_seq_num"] == 1
        assert result["tcmd_response_max_seq_num"] == 3
        assert result["tcmd_response_text"] == "result"

    def test_non_zero_response_code(self) -> None:
        payload = _make_tcmd_payload(response_code=42)
        result = decode_tcmd_response_packet(payload)
        assert result["tcmd_response_code"] == 42

    def test_empty_data(self) -> None:
        payload = _make_tcmd_payload(data=b"")
        result = decode_tcmd_response_packet(payload)
        assert result["tcmd_response_text"] == ""

    def test_data_without_null(self) -> None:
        payload = _make_tcmd_payload(data=b"ABCDE")
        result = decode_tcmd_response_packet(payload)
        assert result["tcmd_response_text"] == "ABCDE"

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="Too short"):
            decode_tcmd_response_packet(b"\x04\x00")

    def test_wrong_packet_type_raises(self) -> None:
        payload = _make_tcmd_payload(packet_type=0x01)
        with pytest.raises(ValueError, match="Unexpected packet_type"):
            decode_tcmd_response_packet(payload)

    def test_multipart_sequence(self) -> None:
        p0 = _make_tcmd_payload(
            response_seq_num=0, response_max_seq_num=2, data=b"part0"
        )
        p1 = _make_tcmd_payload(
            response_seq_num=1, response_max_seq_num=2, data=b"part1"
        )
        r0 = decode_tcmd_response_packet(p0)
        r1 = decode_tcmd_response_packet(p1)
        assert r0["tcmd_response_seq_num"] == 0
        assert r1["tcmd_response_seq_num"] == 1
        assert r0["tcmd_response_max_seq_num"] == 2


# ---------------------------------------------------------------------------
# Tests for decode_bulk_file_downlink_packet
# ---------------------------------------------------------------------------


class TestDecodeBulkFileDownlink:
    def test_basic_fields(self) -> None:
        payload = _make_bulk_payload(file_offset=256, data=b"\x01\x02\x03\x04")
        result = decode_bulk_file_downlink_packet(payload)
        assert result["packet_type"] == "BULK_FILE_DOWNLINK"
        assert result["bulk_file_offset"] == 256
        assert result["bulk_data_len"] == 4
        assert result["bulk_data_hex"] == "01020304"

    def test_zero_offset(self) -> None:
        payload = _make_bulk_payload(file_offset=0)
        result = decode_bulk_file_downlink_packet(payload)
        assert result["bulk_file_offset"] == 0

    def test_large_offset(self) -> None:
        payload = _make_bulk_payload(file_offset=0xFFFFFFFF)
        result = decode_bulk_file_downlink_packet(payload)
        assert result["bulk_file_offset"] == 0xFFFFFFFF

    def test_empty_data(self) -> None:
        payload = _make_bulk_payload(data=b"")
        result = decode_bulk_file_downlink_packet(payload)
        assert result["bulk_data_len"] == 0
        assert result["bulk_data_hex"] == ""

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="Too short"):
            decode_bulk_file_downlink_packet(b"\x10")

    def test_wrong_packet_type_raises(self) -> None:
        payload = _make_bulk_payload(packet_type=0x01)
        with pytest.raises(ValueError, match="Unexpected packet_type"):
            decode_bulk_file_downlink_packet(payload)


# ---------------------------------------------------------------------------
# Tests for decode_packet_safe (top-level dispatcher)
# ---------------------------------------------------------------------------


class TestDecodePacketSafe:
    def _wrap(self, payload: bytes) -> str:
        """Wrap payload with a dummy CSP header and hex-encode."""
        return (DUMMY_CSP + payload).hex()

    def test_beacon_basic_dispatch(self) -> None:
        hex_str = self._wrap(_make_beacon_payload())
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "BEACON_BASIC"
        assert result["csp_header_hex"] == DUMMY_CSP.hex()

    def test_beacon_peripheral_dispatch(self) -> None:
        hex_str = self._wrap(b"\x02\x00\x01\x02\x03")
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "BEACON_PERIPHERAL"

    def test_log_message_dispatch(self) -> None:
        hex_str = self._wrap(_make_log_payload(message=b"test log\n"))
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "LOG_MESSAGE"
        assert result["log_message"] == "test log"

    def test_tcmd_response_dispatch(self) -> None:
        hex_str = self._wrap(_make_tcmd_payload(data=b"ok\n"))
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "TCMD_RESPONSE"

    def test_bulk_file_downlink_dispatch(self) -> None:
        hex_str = self._wrap(_make_bulk_payload())
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "BULK_FILE_DOWNLINK"

    def test_invalid_hex_returns_none(self) -> None:
        result = decode_packet_safe("ZZZZZZZZ")
        assert result is None

    def test_too_short_returns_none(self) -> None:
        # Only 4 bytes → stripped CSP leaves 0 payload bytes.
        result = decode_packet_safe(DUMMY_CSP.hex())
        assert result is None

    def test_unknown_packet_type_returns_unknown(self) -> None:
        # Packet type byte 0xFF is not in PACKET_TYPE_MAP.
        hex_str = self._wrap(b"\xff" + b"\x00" * 10)
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert result["packet_type"] == "UNKNOWN"

    def test_csp_header_always_present(self) -> None:
        hex_str = self._wrap(_make_beacon_payload())
        result = decode_packet_safe(hex_str)
        assert isinstance(result, dict)
        assert "csp_header_hex" in result

    def test_malformed_beacon_falls_back_gracefully(self) -> None:
        # Valid type byte but truncated body → decoder raises, safe wrapper
        # should return a partial result with the type name, not raise.
        bad_beacon = b"\x01" + b"\x00" * 5  # way too short
        hex_str = self._wrap(bad_beacon)
        result = decode_packet_safe(hex_str)
        assert result is not None
        assert "BEACON_BASIC" in result.get("packet_type", "")

    def test_empty_string_returns_none(self) -> None:
        result = decode_packet_safe("")
        assert result is None


# ---------------------------------------------------------------------------
# Tests for PACKET_TYPE_MAP consistency
# ---------------------------------------------------------------------------


class TestPacketTypeMaps:
    def test_inv_map_is_inverse(self) -> None:
        for k, v in PACKET_TYPE_MAP.items():
            assert PACKET_TYPE_MAP_INV[v] == k

    def test_all_expected_types_present(self) -> None:
        for name in (
            "BEACON_BASIC",
            "BEACON_PERIPHERAL",
            "LOG_MESSAGE",
            "TCMD_RESPONSE",
            "BULK_FILE_DOWNLINK",
            "BEACON_EXTENDED",
        ):
            assert name in PACKET_TYPE_MAP_INV


# ---------------------------------------------------------------------------
# Tests for struct size constants
# ---------------------------------------------------------------------------


class TestSizeConstants:
    def test_beacon_fixed_size_matches_format(self) -> None:
        assert struct.calcsize(FIXED_FMT) == BEACON_FIXED_SIZE

    def test_beacon_total_size(self) -> None:
        assert (
            BEACON_TOTAL_STRUCT_SIZE
            == BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE + END_MESSAGE_SIZE
        )

    def test_tcmd_header_size_matches_format(self) -> None:
        assert struct.calcsize(TCMD_RESPONSE_HEADER_FMT) == TCMD_RESPONSE_HEADER_SIZE

    def test_bulk_header_size_matches_format(self) -> None:
        assert struct.calcsize(BULK_DOWNLINK_HEADER_FMT) == BULK_DOWNLINK_HEADER_SIZE

    def test_csp_header_size(self) -> None:
        assert CSP_HEADER_SIZE == 4
