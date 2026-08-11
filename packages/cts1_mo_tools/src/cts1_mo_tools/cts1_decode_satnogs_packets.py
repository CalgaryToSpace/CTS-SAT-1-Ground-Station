"""
CTS-SAT-1 Packet Decoder (from SatNOGS data).

Decodes COMMS_*_packet_t structs from either a SatNOGS-style CSV export or a
SQLite database of received packets.

CSV format (pipe-delimited):
  timestamp | hex_payload | observation_id | ground_station

SQLite format: a "packet" table with (at least) "ts_received", "payload",
"rs_errs", and "session_dir" columns.
"""

import json
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, assert_never

import polars as pl
import tyro
from loguru import logger
from ordered_set import OrderedSet

# -- Constants ----------------------------------------------------------------

CSP_HEADER_SIZE = 4
FRIENDLY_MESSAGE_SIZE = 42  # COMMS_BEACON_FRIENDLY_MESSAGE_SIZE
END_MESSAGE_SIZE = 4  # "END\0"

AX100_DOWNLINK_MAX_BYTES_SIZE = 200

MAX_VALID_EPOCH_MS = int(datetime(2100, 1, 1, 1, 1, 1, tzinfo=UTC).timestamp() * 1000)

FIXED_FMT = (
    "<"
    "B"  # packet_type
    "4s"  # satellite_name
    "B"  # active_rf_switch_antenna
    "B"  # active_rf_switch_control_mode
    "I"  # uptime_ms
    "I"  # duration_since_last_uplink_ms
    "Q"  # unix_epoch_time_ms
    "B"  # last_time_sync_source_enum
    "B"  # is_fs_mounted
    "H"  # total_tcmd_queued_count
    "H"  # pending_queued_tcmd_count
    "I"  # total_beacon_count_since_boot
    "B"  # eps_mode_enum
    "B"  # eps_reset_cause_enum
    "I"  # eps_uptime_sec
    "H"  # eps_error_code
    "H"  # eps_battery_voltage_mV
    "B"  # eps_battery_percent
    "h"  # eps_battery_temperature_0_cC  (signed)
    "h"  # eps_battery_temperature_1_cC  (signed)
    "i"  # eps_total_fault_count          (signed)
    "I"  # eps_enabled_channels_bitfield
    "i"  # eps_total_pcu_power_input_cW   (signed)
    "i"  # eps_total_pcu_power_output_cW  (signed)
    "i"  # eps_total_avg_pcu_power_input_cW  (signed)
    "i"  # eps_total_avg_pcu_power_output_cW (signed)
    "i"  # obc_temperature_cC             (signed)
    "B"  # reboot_reason
    "B"  # cts1_operation_state
    "B"  # rbf_pin_state
    "B"  # mpi_rx_mode_enum
    "B"  # mpi_transceiver_state_enum
    "B"  # mpi_last_reason_for_stopping_enum
    "B"  # gnss_uart_interrupt_enabled
    "B"  # gnss_rx_mode_enum
)
BEACON_FIXED_SIZE = struct.calcsize(FIXED_FMT)
BEACON_TOTAL_STRUCT_SIZE = (
    BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE + END_MESSAGE_SIZE
)  # 130 bytes

BEACON_FIELD_NAMES = [
    "packet_type",
    "satellite_name",
    "active_rf_switch_antenna",
    "active_rf_switch_control_mode",
    "uptime_ms",
    "duration_since_last_uplink_ms",
    "unix_epoch_time_ms",
    "last_time_sync_source_enum",
    "is_fs_mounted",
    "total_tcmd_queued_count",
    "pending_queued_tcmd_count",
    "total_beacon_count_since_boot",
    "eps_mode_enum",
    "eps_reset_cause_enum",
    "eps_uptime_sec",
    "eps_error_code",
    "eps_battery_voltage_mV",
    "eps_battery_percent",
    "eps_battery_temperature_0_cC",
    "eps_battery_temperature_1_cC",
    "eps_total_fault_count",
    "eps_enabled_channels_bitfield",
    "eps_total_pcu_power_input_cW",
    "eps_total_pcu_power_output_cW",
    "eps_total_avg_pcu_power_input_cW",
    "eps_total_avg_pcu_power_output_cW",
    "obc_temperature_cC",
    "reboot_reason",
    "cts1_operation_state",
    "rbf_pin_state",
    "mpi_rx_mode_enum",
    "mpi_transceiver_state_enum",
    "mpi_last_reason_for_stopping_enum",
    "gnss_uart_interrupt_enabled",
    "gnss_rx_mode_enum",
]

# COMMS_beacon_extended_packet_t (beacon v2) adds these fields after the shared
# fixed part + friendly_message + end_message (here called end_version_number,
# e.g. " X2\0" for this packet version).
EXTENDED_FMT = (
    "<"
    "B"  # obc_active_oscillator_MHz
    "h"  # obc_adc_battery_voltage_mV      (signed)
    "b"  # mpi_last_temperature_C          (signed)
    "h"  # eps_pcu_ch0_volt_in_mppt_mV     (signed)
    "h"  # eps_pcu_ch0_curr_in_mppt_mA     (signed)
    "h"  # eps_pcu_ch0_curr_ou_mppt_mA     (signed)
    "h"  # eps_pcu_ch1_volt_in_mppt_mV     (signed)
    "h"  # eps_pcu_ch1_curr_in_mppt_mA     (signed)
    "h"  # eps_pcu_ch1_curr_ou_mppt_mA     (signed)
    "h"  # eps_pcu_ch2_volt_in_mppt_mV     (signed)
    "h"  # eps_pcu_ch2_curr_in_mppt_mA     (signed)
    "h"  # eps_pcu_ch2_curr_ou_mppt_mA     (signed)
    "h"  # eps_pcu_ch3_volt_in_mppt_mV     (signed)
    "h"  # eps_pcu_ch3_curr_in_mppt_mA     (signed)
    "h"  # eps_pcu_ch3_curr_ou_mppt_mA     (signed)
    "H"  # eps_battery_pack_status_bitfield
    "h"  # eps_total_avg_net_battery_power_cW  (signed)
    "h"  # eps_total_avg_power_distributed_cW  (signed)
    "6s"  # adcs_current_state_1 (raw bytes; contents not yet decoded)
    "B"  # adcs_raw_css_1
    "B"  # adcs_raw_css_2
    "B"  # adcs_raw_css_3
    "B"  # adcs_raw_css_4
    "B"  # adcs_raw_css_5
    "B"  # adcs_raw_css_6
    "B"  # adcs_raw_css_7
    "B"  # adcs_raw_css_9
    "h"  # adcs_magnetic_field_x_T_en8     (signed)
    "h"  # adcs_magnetic_field_y_T_en8     (signed)
    "h"  # adcs_magnetic_field_z_T_en8     (signed)
    "H"  # adcs_angular_rate_norm_cdeg_per_sec
    "h"  # adcs_estimated_rate_x_cdeg_per_sec  (signed)
    "h"  # adcs_estimated_rate_y_cdeg_per_sec  (signed)
    "h"  # adcs_estimated_rate_z_cdeg_per_sec  (signed)
    "h"  # adcs_estimated_roll_angle_cdeg  (signed)
    "h"  # adcs_estimated_pitch_angle_cdeg (signed)
    "h"  # adcs_estimated_yaw_angle_cdeg   (signed)
)
EXTENDED_FIXED_SIZE = struct.calcsize(EXTENDED_FMT)
BEACON_EXTENDED_TOTAL_STRUCT_SIZE = (
    BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE + END_MESSAGE_SIZE + EXTENDED_FIXED_SIZE
)  # 198 bytes

EXTENDED_FIELD_NAMES = [
    "obc_active_oscillator_MHz",
    "obc_adc_battery_voltage_mV",
    "mpi_last_temperature_C",
    "eps_pcu_ch0_volt_in_mppt_mV",
    "eps_pcu_ch0_curr_in_mppt_mA",
    "eps_pcu_ch0_curr_ou_mppt_mA",
    "eps_pcu_ch1_volt_in_mppt_mV",
    "eps_pcu_ch1_curr_in_mppt_mA",
    "eps_pcu_ch1_curr_ou_mppt_mA",
    "eps_pcu_ch2_volt_in_mppt_mV",
    "eps_pcu_ch2_curr_in_mppt_mA",
    "eps_pcu_ch2_curr_ou_mppt_mA",
    "eps_pcu_ch3_volt_in_mppt_mV",
    "eps_pcu_ch3_curr_in_mppt_mA",
    "eps_pcu_ch3_curr_ou_mppt_mA",
    "eps_battery_pack_status_bitfield",
    "eps_total_avg_net_battery_power_cW",
    "eps_total_avg_power_distributed_cW",
    "adcs_current_state_1",
    "adcs_raw_css_1",
    "adcs_raw_css_2",
    "adcs_raw_css_3",
    "adcs_raw_css_4",
    "adcs_raw_css_5",
    "adcs_raw_css_6",
    "adcs_raw_css_7",
    "adcs_raw_css_9",
    "adcs_magnetic_field_x_T_en8",
    "adcs_magnetic_field_y_T_en8",
    "adcs_magnetic_field_z_T_en8",
    "adcs_angular_rate_norm_cdeg_per_sec",
    "adcs_estimated_rate_x_cdeg_per_sec",
    "adcs_estimated_rate_y_cdeg_per_sec",
    "adcs_estimated_rate_z_cdeg_per_sec",
    "adcs_estimated_roll_angle_cdeg",
    "adcs_estimated_pitch_angle_cdeg",
    "adcs_estimated_yaw_angle_cdeg",
]

# COMMS_tcmd_response_packet_t layout (after the CSP header):
#   uint8_t  packet_type        1
#   uint64_t ts_sent            8
#   uint8_t  response_code      1
#   uint16_t duration_ms        2
#   uint8_t  response_seq_num   1
#   uint8_t  response_max_seq_num 1
#   uint8_t  data[186]
TCMD_RESPONSE_HEADER_FMT = "<B Q B H B B"
TCMD_RESPONSE_HEADER_SIZE = struct.calcsize(TCMD_RESPONSE_HEADER_FMT)
TCMD_RESPONSE_MAX_DATA = AX100_DOWNLINK_MAX_BYTES_SIZE - 1 - 8 - 1 - 2 - 1 - 1  # 186

# COMMS_bulk_file_downlink_packet_t layout (after the CSP header):
#   uint8_t  packet_type   1
#   uint32_t file_offset   4
#   uint8_t  data[195]
BULK_DOWNLINK_HEADER_FMT = "<B I"
BULK_DOWNLINK_HEADER_SIZE = struct.calcsize(BULK_DOWNLINK_HEADER_FMT)
BULK_DOWNLINK_MAX_DATA = AX100_DOWNLINK_MAX_BYTES_SIZE - 1 - 4  # 195

# COMMS_log_message_packet_t layout (after the CSP header):
#   uint8_t packet_type   1
#   uint8_t data[199]
LOG_MESSAGE_MAX_DATA = AX100_DOWNLINK_MAX_BYTES_SIZE - 1  # 199

# -- Enum maps ----------------------------------------------------------------

PACKET_TYPE_MAP = {
    0x01: "BEACON_BASIC",
    0x02: "BEACON_PERIPHERAL",
    0x03: "LOG_MESSAGE",
    0x04: "TCMD_RESPONSE",
    0x10: "BULK_FILE_DOWNLINK",
    0x20: "BEACON_EXTENDED",
}
PACKET_TYPE_MAP_INV = {v: k for k, v in PACKET_TYPE_MAP.items()}
RF_SWITCH_CONTROL_MODE_MAP = {
    0: "TOGGLE_BEFORE_EVERY_BEACON",
    1: "FORCE_ANT1",
    2: "FORCE_ANT2",
    3: "USE_ADCS_NORMAL",
    4: "USE_ADCS_FLIPPED",
    255: "UNKNOWN",
}
TIME_SYNC_SOURCE_MAP = {
    0: "NONE",
    1: "GNSS_UART",
    2: "GNSS_PPS",
    3: "TELECOMMAND_ABSOLUTE",
    4: "TELECOMMAND_CORRECTION",
    5: "EPS_RTC",
}
EPS_MODE_MAP = {0: "STARTUP", 1: "NOMINAL", 2: "SAFETY", 3: "EMERGENCY_LOW_POWER"}
EPS_RESET_CAUSE_MAP = {
    0: "POWER_ON",
    1: "WATCHDOG",
    2: "COMMANDED",
    3: "CONTROL_SYSTEM_RESET",
    4: "EMERGENCY_LOW_POWER",
}
STM32_RESET_CAUSE_MAP = {
    0: "UNKNOWN",
    1: "LOW_POWER_RESET",
    2: "WINDOW_WATCHDOG_RESET",
    3: "INDEPENDENT_WATCHDOG_RESET",
    4: "SOFTWARE_RESET",
    5: "EXTERNAL_RESET_PIN_RESET",
    6: "BROWNOUT_RESET",
    7: "OPTION_BYTE_LOADER_RESET",
    8: "FIREWALL_RESET",
}
CTS1_OPERATION_STATE_MAP = {
    0: "BOOTED_AND_WAITING",
    1: "DEPLOYING",
    2: "NOMINAL_WITH_RADIO_TX",
    3: "NOMINAL_WITHOUT_RADIO_TX",
}
RBF_STATE_MAP = {0: "BENCH", 1: "FLYING"}
MPI_RX_MODE_MAP = {0: "COMMAND_MODE", 1: "SENSING_MODE", 2: "NOT_LISTENING_TO_MPI"}
MPI_TRANSCEIVER_STATE_MAP = {0: "INACTIVE", 1: "MOSI", 2: "MISO", 3: "DUPLEX"}
MPI_STOP_REASON_MAP = {
    0: "NOT_SET",
    1: "TEMPERATURE_EXCEEDED",
    2: "TELECOMMAND",
    3: "MAX_TIME_EXCEEDED",
    4: "SELF_CHECK_DONE",
}
GNSS_RX_MODE_MAP = {0: "COMMAND_MODE", 1: "FIREHOSE_MODE", 2: "DISABLED"}

# ADCS Current State (Telemetry ID 132, frame 1) enum maps.
ADCS_ESTIM_MODE_MAP = {
    0: "No attitude estimation",
    1: "MEMS rate sensing",
    2: "Magnetometer rate filter",
    3: "Magnetometer rate filter with pitch estimation",
    4: "Magnetometer and Fine-sun TRIAD algorithm",
    5: "Full-state EKF",
    6: "MEMS gyro EKF",
    7: "User Coded Estimation Mode",
}
ADCS_CONTROL_MODE_MAP = {
    0: "No control",
    1: "Detumbling control",
    2: "Y-Thomson spin",
    3: "Y-Wheel momentum stabilized - Initial Pitch Acquisition",
    4: "Y-Wheel momentum stabilized - Steady State",
    5: "XYZ-Wheel control",
    6: "Rwheel sun tracking control",
    7: "Rwheel target tracking control",
    8: "Very Fast-spin Detumbling control (10Hz)",
    9: "Fast-spin Detumbling control",
    10: "User Specific Control Mode 1",
    11: "User Specific Control Mode 2",
    12: "Stop R-wheels",
    13: "User Coded Control Mode",
    14: "Sun-tracking yaw- or roll-only wheel control mode",
    15: "Target-tracking yaw-only wheel control mode",
}
ADCS_RUN_MODE_MAP = {0: "Off", 1: "Enabled", 2: "Triggered", 3: "Simulation"}
ADCS_ASGP4_MODE_MAP = {0: "Off", 1: "Trigger", 2: "Background", 3: "Augment"}

# ADCS Current State (Telemetry ID 132, frame 1), offset 12-22: single-bit BOOL
# "... Enabled" fields (hardware/electronics enabled statuses).
ADCS_ENABLED_BITS: list[tuple[int, str]] = [
    (12, "CUBECONTROL_SIGNAL"),
    (13, "CUBECONTROL_MOTOR"),
    (14, "CUBESENSE1"),
    (15, "CUBESENSE2"),
    (16, "CUBEWHEEL1"),
    (17, "CUBEWHEEL2"),
    (18, "CUBEWHEEL3"),
    (19, "CUBESTAR"),
    (20, "GPS_RECEIVER"),
    (21, "GPS_LNA_POWER"),
    (22, "MOTOR_DRIVER"),
]

# ADCS Current State (Telemetry ID 132, frame 1), offset 12-47: single-bit BOOL
# fields explicitly named "... Error" (comms errors, out-of-range detections).
ADCS_ERROR_BITS: list[tuple[int, str]] = [
    (24, "CUBESENSE1_COMMS_ERROR"),
    (25, "CUBESENSE2_COMMS_ERROR"),
    (26, "CUBECONTROL_SIGNAL_COMMS_ERROR"),
    (27, "CUBECONTROL_MOTOR_COMMS_ERROR"),
    (28, "CUBEWHEEL1_COMMS_ERROR"),
    (29, "CUBEWHEEL2_COMMS_ERROR"),
    (30, "CUBEWHEEL3_COMMS_ERROR"),
    (31, "CUBESTAR_COMMS_ERROR"),
    (32, "MAGNETOMETER_RANGE_ERROR"),
    (35, "CAM1_SENSOR_BUSY_ERROR"),
    (36, "CAM1_SENSOR_DETECTION_ERROR"),
    (37, "SUN_SENSOR_RANGE_ERROR"),
    (40, "CAM2_SENSOR_BUSY_ERROR"),
    (41, "CAM2_SENSOR_DETECTION_ERROR"),
    (42, "NADIR_SENSOR_RANGE_ERROR"),
    (43, "RATE_SENSOR_RANGE_ERROR"),
    (44, "WHEEL_SPEED_RANGE_ERROR"),
    (45, "COARSE_SUN_SENSOR_ERROR"),
    (46, "STAR_TRACKER_MATCH_ERROR"),
]

# Same offset range: single-bit BOOL fields for "overcurrent detected" conditions
# (not named "... Error" in the spec, kept as a separate "flags" bucket).
ADCS_FLAG_BITS: list[tuple[int, str]] = [
    (33, "CAM1_SRAM_OVERCURRENT"),
    (34, "CAM1_3V3_OVERCURRENT"),
    (38, "CAM2_SRAM_OVERCURRENT"),
    (39, "CAM2_3V3_OVERCURRENT"),
    (47, "STAR_TRACKER_OVERCURRENT"),
]


def e(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"UNKNOWN({value})")


def e_numbered(mapping: dict[int, str], value: int) -> str:
    """Like `e()`, but merges the number in, e.g. "2 - Y-Thomson spin"."""
    return f"{value} - {e(mapping, value)}"


def decode_adcs_current_state_1(raw: bytes) -> dict[str, Any]:
    """Decode the 6-byte ADCS Current State telemetry frame (ID 132, frame 1).

    Bit layout (48 bits total, byte order little-endian):
      bits 0-3:   Attitude Estimation Mode (ENUM, see ADCS_ESTIM_MODE_MAP)
      bits 4-7:   Control Mode (ENUM, see ADCS_CONTROL_MODE_MAP)
      bits 8-9:   ADCS Run Mode (ENUM, see ADCS_RUN_MODE_MAP)
      bits 10-11: ASGP4 Mode (ENUM, see ADCS_ASGP4_MODE_MAP)
      bits 12-47: single-bit BOOL flags (enabled statuses, comms/range errors,
                  overcurrent detections); see ADCS_ENABLED_BITS / ADCS_ERROR_BITS
                  / ADCS_FLAG_BITS.
    """
    if len(raw) != 6:  # noqa: PLR2004
        msg = f"adcs_current_state_1 must be 6 bytes, got {len(raw)}"
        raise ValueError(msg)

    value = int.from_bytes(raw, "little")

    def bit(i: int) -> bool:
        return bool((value >> i) & 1)

    estim_mode = value & 0xF
    control_mode = (value >> 4) & 0xF
    run_mode = (value >> 8) & 0x3
    asgp4_mode = (value >> 10) & 0x3

    enabled = [name for bit_num, name in ADCS_ENABLED_BITS if bit(bit_num)]
    errors = [name for bit_num, name in ADCS_ERROR_BITS if bit(bit_num)]
    flags = [name for bit_num, name in ADCS_FLAG_BITS if bit(bit_num)]

    return {
        "adcs_attitude_estimation_mode": e_numbered(ADCS_ESTIM_MODE_MAP, estim_mode),
        "adcs_control_mode": e_numbered(ADCS_CONTROL_MODE_MAP, control_mode),
        "adcs_run_mode": e_numbered(ADCS_RUN_MODE_MAP, run_mode),
        "adcs_asgp4_mode": e_numbered(ADCS_ASGP4_MODE_MAP, asgp4_mode),
        "adcs_enabled": json.dumps(enabled),
        "adcs_sun_above_local_horizon": bit(23),
        "adcs_errors": json.dumps(errors),
        "adcs_flags": json.dumps(flags),
    }


def epoch_ms_to_iso(epoch_ms: int | None) -> str | None:
    """Convert a Unix epoch (milliseconds) to an ISO-8601 UTC string, or None."""
    if epoch_ms is None or epoch_ms <= 0 or epoch_ms > MAX_VALID_EPOCH_MS:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, UTC).isoformat()


# -- Decoders -----------------------------------------------------------------


def decode_beacon_basic_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_beacon_basic_packet_t payload (CSP header already stripped)."""
    if len(payload) < BEACON_TOTAL_STRUCT_SIZE:
        msg = (
            f"Too short for BEACON_BASIC: {len(payload)} bytes "
            f"(need {BEACON_TOTAL_STRUCT_SIZE})"
        )
        raise ValueError(msg)

    vals = struct.unpack_from(FIXED_FMT, payload, 0)

    if vals[0] != PACKET_TYPE_MAP_INV["BEACON_BASIC"]:
        msg = f"Unexpected packet_type byte for BEACON_BASIC: {vals[0]:#04x}"
        raise ValueError(msg)

    rf = dict(zip(BEACON_FIELD_NAMES, vals, strict=True))
    fm_raw = payload[BEACON_FIXED_SIZE : BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE]
    friendly = fm_raw.split(b"\x00")[0].decode("utf-8", errors="replace")
    sat_name = rf["satellite_name"].decode("ascii", errors="replace").rstrip("\x00")
    epoch_ms = rf["unix_epoch_time_ms"]
    utc_time = epoch_ms_to_iso(epoch_ms)

    data = {
        "packet_type": e(PACKET_TYPE_MAP, rf["packet_type"]),
        # Identity
        "satellite_name": sat_name,
        # RF switch
        "active_rf_switch_antenna": rf["active_rf_switch_antenna"],
        "active_rf_switch_control_mode": e(
            RF_SWITCH_CONTROL_MODE_MAP, rf["active_rf_switch_control_mode"]
        ),
        # Timing
        "uptime_ms": rf["uptime_ms"],
        "uptime_sec": round(rf["uptime_ms"] / 1000, 3),
        "duration_since_last_uplink_ms": rf["duration_since_last_uplink_ms"],
        "unix_epoch_time_ms": epoch_ms,
        "utc_time": utc_time,
        "last_time_sync_source": e(
            TIME_SYNC_SOURCE_MAP, rf["last_time_sync_source_enum"]
        ),
        # OBC
        "is_fs_mounted": bool(rf["is_fs_mounted"]),
        "total_tcmd_queued_count": rf["total_tcmd_queued_count"],
        "pending_queued_tcmd_count": rf["pending_queued_tcmd_count"],
        "total_beacon_count_since_boot": rf["total_beacon_count_since_boot"],
        "reboot_reason": e(STM32_RESET_CAUSE_MAP, rf["reboot_reason"]),
        "obc_temperature_C": round(rf["obc_temperature_cC"] / 100.0, 2),
        # EPS
        "eps_mode": e(EPS_MODE_MAP, rf["eps_mode_enum"]),
        "eps_reset_cause": e(EPS_RESET_CAUSE_MAP, rf["eps_reset_cause_enum"]),
        "eps_uptime_sec": rf["eps_uptime_sec"],
        "eps_error_code": rf["eps_error_code"],
        "eps_battery_voltage_V": round(rf["eps_battery_voltage_mV"] / 1000.0, 3),
        "eps_battery_percent": rf["eps_battery_percent"],
        "eps_battery_temperature_0_C": round(
            rf["eps_battery_temperature_0_cC"] / 100.0, 2
        ),
        "eps_battery_temperature_1_C": round(
            rf["eps_battery_temperature_1_cC"] / 100.0, 2
        ),
        "eps_total_fault_count": rf["eps_total_fault_count"],
        "eps_enabled_channels_bitfield": f"0x{rf['eps_enabled_channels_bitfield']:08X}",
        "eps_total_pcu_power_input_W": round(
            rf["eps_total_pcu_power_input_cW"] / 100.0, 2
        ),
        "eps_total_pcu_power_output_W": round(
            rf["eps_total_pcu_power_output_cW"] / 100.0, 2
        ),
        "eps_total_avg_pcu_power_input_W": round(
            rf["eps_total_avg_pcu_power_input_cW"] / 100.0, 2
        ),
        "eps_total_avg_pcu_power_output_W": round(
            rf["eps_total_avg_pcu_power_output_cW"] / 100.0, 2
        ),
        # CTS1 state
        "cts1_operation_state": e(CTS1_OPERATION_STATE_MAP, rf["cts1_operation_state"]),
        "rbf_pin_state": e(RBF_STATE_MAP, rf["rbf_pin_state"]),
        # MPI
        "mpi_rx_mode": e(MPI_RX_MODE_MAP, rf["mpi_rx_mode_enum"]),
        "mpi_transceiver_state": e(
            MPI_TRANSCEIVER_STATE_MAP, rf["mpi_transceiver_state_enum"]
        ),
        "mpi_last_reason_for_stopping": e(
            MPI_STOP_REASON_MAP, rf["mpi_last_reason_for_stopping_enum"]
        ),
        # GNSS
        "gnss_uart_interrupt_enabled": bool(rf["gnss_uart_interrupt_enabled"]),
        "gnss_rx_mode": e(GNSS_RX_MODE_MAP, rf["gnss_rx_mode_enum"]),
        # Friendly
        "friendly_message": friendly,
    }

    return data  # noqa: RET504


def decode_beacon_extended_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_beacon_extended_packet_t payload (CSP header already stripped).

    Layout: the same fixed part + friendly_message + end_message as
    BEACON_BASIC, followed by additional beacon v2 fields (EPS per-channel
    solar data, and ADCS data). The ADCS state/status bitfield
    (``adcs_current_state_1``) is surfaced as raw hex; its internals are not
    decoded yet.
    """
    if len(payload) < BEACON_EXTENDED_TOTAL_STRUCT_SIZE:
        msg = (
            f"Too short for BEACON_EXTENDED: {len(payload)} bytes "
            f"(need {BEACON_EXTENDED_TOTAL_STRUCT_SIZE})"
        )
        raise ValueError(msg)

    vals = struct.unpack_from(FIXED_FMT, payload, 0)

    if vals[0] != PACKET_TYPE_MAP_INV["BEACON_EXTENDED"]:
        msg = f"Unexpected packet_type byte for BEACON_EXTENDED: {vals[0]:#04x}"
        raise ValueError(msg)

    rf = dict(zip(BEACON_FIELD_NAMES, vals, strict=True))

    fm_raw = payload[BEACON_FIXED_SIZE : BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE]
    friendly = fm_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    end_offset = BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE
    end_raw = payload[end_offset : end_offset + END_MESSAGE_SIZE]
    end_version_number = end_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    ext_offset = end_offset + END_MESSAGE_SIZE
    ext_vals = struct.unpack_from(EXTENDED_FMT, payload, ext_offset)
    ef = dict(zip(EXTENDED_FIELD_NAMES, ext_vals, strict=True))

    sat_name = rf["satellite_name"].decode("ascii", errors="replace").rstrip("\x00")
    epoch_ms = rf["unix_epoch_time_ms"]

    if epoch_ms <= 0 or epoch_ms > MAX_VALID_EPOCH_MS:
        utc_time = None
    else:
        utc_time = (
            datetime.fromtimestamp(
                epoch_ms / 1000.0,
                UTC,
            ).isoformat()
            + "Z"
        )

    data = {
        "packet_type": e(PACKET_TYPE_MAP, rf["packet_type"]),
        # Identity
        "satellite_name": sat_name,
        # RF switch
        "active_rf_switch_antenna": rf["active_rf_switch_antenna"],
        "active_rf_switch_control_mode": e(
            RF_SWITCH_CONTROL_MODE_MAP, rf["active_rf_switch_control_mode"]
        ),
        # Timing
        "uptime_ms": rf["uptime_ms"],
        "uptime_sec": round(rf["uptime_ms"] / 1000, 3),
        "duration_since_last_uplink_ms": rf["duration_since_last_uplink_ms"],
        "unix_epoch_time_ms": epoch_ms,
        "utc_time": utc_time,
        "last_time_sync_source": e(
            TIME_SYNC_SOURCE_MAP, rf["last_time_sync_source_enum"]
        ),
        # OBC
        "is_fs_mounted": bool(rf["is_fs_mounted"]),
        "total_tcmd_queued_count": rf["total_tcmd_queued_count"],
        "pending_queued_tcmd_count": rf["pending_queued_tcmd_count"],
        "total_beacon_count_since_boot": rf["total_beacon_count_since_boot"],
        "reboot_reason": e(STM32_RESET_CAUSE_MAP, rf["reboot_reason"]),
        "obc_temperature_C": round(rf["obc_temperature_cC"] / 100.0, 2),
        "obc_active_oscillator_MHz": ef["obc_active_oscillator_MHz"],
        "obc_adc_battery_voltage_V": round(
            ef["obc_adc_battery_voltage_mV"] / 1000.0, 3
        ),
        # EPS
        "eps_mode": e(EPS_MODE_MAP, rf["eps_mode_enum"]),
        "eps_reset_cause": e(EPS_RESET_CAUSE_MAP, rf["eps_reset_cause_enum"]),
        "eps_uptime_sec": rf["eps_uptime_sec"],
        "eps_error_code": rf["eps_error_code"],
        "eps_battery_voltage_V": round(rf["eps_battery_voltage_mV"] / 1000.0, 3),
        "eps_battery_percent": rf["eps_battery_percent"],
        "eps_battery_temperature_0_C": round(
            rf["eps_battery_temperature_0_cC"] / 100.0, 2
        ),
        "eps_battery_temperature_1_C": round(
            rf["eps_battery_temperature_1_cC"] / 100.0, 2
        ),
        "eps_total_fault_count": rf["eps_total_fault_count"],
        "eps_enabled_channels_bitfield": f"0x{rf['eps_enabled_channels_bitfield']:08X}",
        "eps_total_pcu_power_input_W": round(
            rf["eps_total_pcu_power_input_cW"] / 100.0, 2
        ),
        "eps_total_pcu_power_output_W": round(
            rf["eps_total_pcu_power_output_cW"] / 100.0, 2
        ),
        "eps_total_avg_pcu_power_input_W": round(
            rf["eps_total_avg_pcu_power_input_cW"] / 100.0, 2
        ),
        "eps_total_avg_pcu_power_output_W": round(
            rf["eps_total_avg_pcu_power_output_cW"] / 100.0, 2
        ),
        "eps_battery_pack_status_bitfield": (
            f"0x{ef['eps_battery_pack_status_bitfield']:04X}"
        ),
        "eps_total_avg_net_battery_power_W": round(
            ef["eps_total_avg_net_battery_power_cW"] / 100.0, 2
        ),
        "eps_total_avg_power_distributed_W": round(
            ef["eps_total_avg_power_distributed_cW"] / 100.0, 2
        ),
        # EPS per-channel PCU (instantaneous solar panel measurements)
        "eps_pcu_ch0_volt_in_mppt_V": round(
            ef["eps_pcu_ch0_volt_in_mppt_mV"] / 1000.0, 3
        ),
        "eps_pcu_ch0_curr_in_mppt_A": round(
            ef["eps_pcu_ch0_curr_in_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch0_curr_ou_mppt_A": round(
            ef["eps_pcu_ch0_curr_ou_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch1_volt_in_mppt_V": round(
            ef["eps_pcu_ch1_volt_in_mppt_mV"] / 1000.0, 3
        ),
        "eps_pcu_ch1_curr_in_mppt_A": round(
            ef["eps_pcu_ch1_curr_in_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch1_curr_ou_mppt_A": round(
            ef["eps_pcu_ch1_curr_ou_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch2_volt_in_mppt_V": round(
            ef["eps_pcu_ch2_volt_in_mppt_mV"] / 1000.0, 3
        ),
        "eps_pcu_ch2_curr_in_mppt_A": round(
            ef["eps_pcu_ch2_curr_in_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch2_curr_ou_mppt_A": round(
            ef["eps_pcu_ch2_curr_ou_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch3_volt_in_mppt_V": round(
            ef["eps_pcu_ch3_volt_in_mppt_mV"] / 1000.0, 3
        ),
        "eps_pcu_ch3_curr_in_mppt_A": round(
            ef["eps_pcu_ch3_curr_in_mppt_mA"] / 1000.0, 3
        ),
        "eps_pcu_ch3_curr_ou_mppt_A": round(
            ef["eps_pcu_ch3_curr_ou_mppt_mA"] / 1000.0, 3
        ),
        # CTS1 state
        "cts1_operation_state": e(CTS1_OPERATION_STATE_MAP, rf["cts1_operation_state"]),
        "rbf_pin_state": e(RBF_STATE_MAP, rf["rbf_pin_state"]),
        # MPI
        "mpi_rx_mode": e(MPI_RX_MODE_MAP, rf["mpi_rx_mode_enum"]),
        "mpi_transceiver_state": e(
            MPI_TRANSCEIVER_STATE_MAP, rf["mpi_transceiver_state_enum"]
        ),
        "mpi_last_reason_for_stopping": e(
            MPI_STOP_REASON_MAP, rf["mpi_last_reason_for_stopping_enum"]
        ),
        "mpi_last_temperature_C": ef["mpi_last_temperature_C"],
        # GNSS
        "gnss_uart_interrupt_enabled": bool(rf["gnss_uart_interrupt_enabled"]),
        "gnss_rx_mode": e(GNSS_RX_MODE_MAP, rf["gnss_rx_mode_enum"]),
        # ADCS
        **decode_adcs_current_state_1(ef["adcs_current_state_1"]),
        "adcs_raw_css_1": ef["adcs_raw_css_1"],
        "adcs_raw_css_2": ef["adcs_raw_css_2"],
        "adcs_raw_css_3": ef["adcs_raw_css_3"],
        "adcs_raw_css_4": ef["adcs_raw_css_4"],
        "adcs_raw_css_5": ef["adcs_raw_css_5"],
        "adcs_raw_css_6": ef["adcs_raw_css_6"],
        "adcs_raw_css_7": ef["adcs_raw_css_7"],
        "adcs_raw_css_9": ef["adcs_raw_css_9"],
        "adcs_magnetic_field_x_uT": round(ef["adcs_magnetic_field_x_T_en8"] * 0.01, 2),
        "adcs_magnetic_field_y_uT": round(ef["adcs_magnetic_field_y_T_en8"] * 0.01, 2),
        "adcs_magnetic_field_z_uT": round(ef["adcs_magnetic_field_z_T_en8"] * 0.01, 2),
        "adcs_angular_rate_norm_deg_per_sec": round(
            ef["adcs_angular_rate_norm_cdeg_per_sec"] * 0.01, 2
        ),
        "adcs_estimated_rate_x_deg_per_sec": round(
            ef["adcs_estimated_rate_x_cdeg_per_sec"] * 0.01, 2
        ),
        "adcs_estimated_rate_y_deg_per_sec": round(
            ef["adcs_estimated_rate_y_cdeg_per_sec"] * 0.01, 2
        ),
        "adcs_estimated_rate_z_deg_per_sec": round(
            ef["adcs_estimated_rate_z_cdeg_per_sec"] * 0.01, 2
        ),
        "adcs_estimated_roll_angle_deg": round(
            ef["adcs_estimated_roll_angle_cdeg"] * 0.01, 2
        ),
        "adcs_estimated_pitch_angle_deg": round(
            ef["adcs_estimated_pitch_angle_cdeg"] * 0.01, 2
        ),
        "adcs_estimated_yaw_angle_deg": round(
            ef["adcs_estimated_yaw_angle_cdeg"] * 0.01, 2
        ),
        # Friendly
        "friendly_message": friendly,
        "end_version_number": end_version_number,
    }

    return data  # noqa: RET504


def decode_beacon_peripheral_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_PACKET_TYPE_BEACON_PERIPHERAL payload.

    No struct is defined in the header for this type yet; we surface the raw
    bytes so the row is still tagged correctly rather than silently dropped.
    """
    return {
        "packet_type": "BEACON_PERIPHERAL",
        "raw_payload_hex": payload.hex(),
        "_note": "BEACON_PERIPHERAL struct not yet defined; raw bytes preserved",
    }


def decode_log_message_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_log_message_packet_t payload (CSP header already stripped).

    Layout:
        uint8_t  packet_type   (1 byte, always 0x03)
        uint8_t  data[199]     (null-terminated UTF-8 log string)
    """
    if len(payload) < 1:
        msg = "Too short for LOG_MESSAGE: 0 bytes"
        raise ValueError(msg)

    if payload[0] != PACKET_TYPE_MAP_INV["LOG_MESSAGE"]:
        msg = f"Unexpected packet_type byte for LOG_MESSAGE: {payload[0]:#04x}"
        raise ValueError(msg)

    data_bytes = payload[1 : 1 + LOG_MESSAGE_MAX_DATA]

    # Treat as null-terminated string; preserve anything after the first null
    # as a hex dump for forensic purposes.
    last_newline_pos = data_bytes.rfind(b"\n")

    # Slice off the trailing bytes, if present.
    if last_newline_pos >= 0:
        message = data_bytes[:last_newline_pos].decode("utf-8", errors="replace")
    else:
        message = data_bytes.decode("utf-8", errors="replace")

    return {
        "packet_type": "LOG_MESSAGE",
        "log_message": message,
        # Not actually useful - "log_trailing_data_hex": trailing_hex,
    }


def decode_tcmd_response_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_tcmd_response_packet_t payload (CSP header already stripped).

    Layout:
        uint8_t  packet_type          (1 byte, always 0x04)
        uint64_t ts_sent              (8 bytes)
        uint8_t  response_code        (1 byte)
        uint16_t duration_ms          (2 bytes)
        uint8_t  response_seq_num     (1 byte)
        uint8_t  response_max_seq_num (1 byte)
        uint8_t  data[186]
    """
    if len(payload) < TCMD_RESPONSE_HEADER_SIZE:
        msg = (
            f"Too short for TCMD_RESPONSE: {len(payload)} bytes "
            f"(need at least {TCMD_RESPONSE_HEADER_SIZE})"
        )
        raise ValueError(msg)

    (
        packet_type,
        ts_sent,
        response_code,
        duration_ms,
        response_seq_num,
        response_max_seq_num,
    ) = struct.unpack_from(TCMD_RESPONSE_HEADER_FMT, payload, 0)

    if packet_type != PACKET_TYPE_MAP_INV["TCMD_RESPONSE"]:
        msg = f"Unexpected packet_type byte for TCMD_RESPONSE: {packet_type:#04x}"
        raise ValueError(msg)

    data_bytes = payload[
        TCMD_RESPONSE_HEADER_SIZE : TCMD_RESPONSE_HEADER_SIZE + TCMD_RESPONSE_MAX_DATA
    ]
    # Treat the response data as a null-terminated string if it looks like text.
    null_pos = data_bytes.find(b"\x00")
    if null_pos >= 0:
        response_text = data_bytes[:null_pos].decode("utf-8", errors="replace")
    else:
        response_text = data_bytes.decode("utf-8", errors="replace")

    return {
        "packet_type": "TCMD_RESPONSE",
        "tcmd_ts_sent": ts_sent,
        "tcmd_response_code": response_code,
        "tcmd_duration_ms": duration_ms,
        "tcmd_response_seq_num": response_seq_num,
        "tcmd_response_max_seq_num": response_max_seq_num,
        "tcmd_response_text": response_text,
    }


def decode_bulk_file_downlink_packet(payload: bytes) -> dict[str, Any]:
    """Decode a COMMS_bulk_file_downlink_packet_t payload (CSP header already stripped).

    Layout:
        uint8_t  packet_type  (1 byte, always 0x10)
        uint32_t file_offset  (4 bytes)
        uint8_t  data[195]
    """
    if len(payload) < BULK_DOWNLINK_HEADER_SIZE:
        msg = (
            f"Too short for BULK_FILE_DOWNLINK: {len(payload)} bytes "
            f"(need at least {BULK_DOWNLINK_HEADER_SIZE})"
        )
        raise ValueError(msg)

    packet_type, file_offset = struct.unpack_from(BULK_DOWNLINK_HEADER_FMT, payload, 0)

    if packet_type != PACKET_TYPE_MAP_INV["BULK_FILE_DOWNLINK"]:
        msg = f"Unexpected packet_type byte for BULK_FILE_DOWNLINK: {packet_type:#04x}"
        raise ValueError(msg)

    data_bytes = payload[
        BULK_DOWNLINK_HEADER_SIZE : BULK_DOWNLINK_HEADER_SIZE + BULK_DOWNLINK_MAX_DATA
    ]

    return {
        "packet_type": "BULK_FILE_DOWNLINK",
        "bulk_file_offset": file_offset,
        "bulk_data_len": len(data_bytes),
        "bulk_data_hex": data_bytes.hex(),
    }


# Map packet_type byte → decoder function (payload = post-CSP bytes).
_PACKET_DECODERS = {
    0x01: decode_beacon_basic_packet,
    0x02: decode_beacon_peripheral_packet,
    0x03: decode_log_message_packet,
    0x04: decode_tcmd_response_packet,
    0x10: decode_bulk_file_downlink_packet,
    0x20: decode_beacon_extended_packet,
}


def decode_packet_safe(hex_str: str) -> dict[str, Any] | None:
    """Attempt to decode any supported packet type.

    Returns a dict with at least ``packet_type`` set, or None if the bytes
    cannot be interpreted at all.
    """
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError:
        return None

    if len(raw) <= CSP_HEADER_SIZE:
        return None

    csp = raw[:CSP_HEADER_SIZE]
    payload = raw[CSP_HEADER_SIZE:]
    packet_type_byte = payload[0]

    base = {"csp_header_hex": csp.hex()}

    decoder = _PACKET_DECODERS.get(packet_type_byte)
    if decoder is not None:
        try:
            decoded = decoder(payload)
        except (ValueError, struct.error) as exc:
            logger.warning(
                f"Failed to decode packet type {packet_type_byte:#04x}: {exc}"
            )
            # Fall through to the partial decode below.
        else:
            return {**base, **decoded}

    # Unknown or malformed: at least tag the packet type name.
    packet_type_name = e(PACKET_TYPE_MAP, packet_type_byte)
    if "UNKNOWN" in packet_type_name:
        packet_type_name = "UNKNOWN"
    return {**base, "packet_type": packet_type_name}


# -- Main ---------------------------------------------------------------------


def load_packets_from_csv(input_csv: Path) -> pl.DataFrame:
    """Load received packets from a SatNOGS-style pipe-delimited CSV."""
    logger.info(f"Reading: {input_csv}")

    df = pl.read_csv(
        input_csv,
        separator="|",
        has_header=False,
        new_columns=[
            "received_timestamp",
            "hex_payload",
            "observation_id",
            "ground_station",
        ],
    )

    logger.info(f"Read: {len(df)} rows")
    return df


SQLITE_PACKETS_QUERY = """
    SELECT
        ts_received,
        lower(hex(payload)) AS payload_hex,
        session_dir,
        csp_src,
        csp_dst,
        csp_dport,
        csp_sport,
        csp_prio,
        csp_flags
    FROM packet
    WHERE rs_errs >= 0
"""

SQLITE_SENT_TCMD_QUERY = """
    SELECT
        id,
        ts_sent_ms,
        tsexec_ms,
        command_text,
        tx_freq_hz,
        tx_gain_db,
        source_tool,
        source_run,
        ts_transmitted
    FROM sent_tcmd
"""


def encode_csp_header(  # noqa: PLR0913
    *, prio: int, src: int, dst: int, dport: int, sport: int, flags: int
) -> bytes:
    """Encode a CSP 1.x header (4 bytes, network/big-endian byte order).

    Bit layout (MSB to LSB): prio(2) src(5) dst(5) dport(6) sport(6) flags(8),
    where "flags" is the reserved/hmac/xtea/rdp/crc bits, packed as one byte.
    """
    value = (
        (prio & 0x3) << 30
        | (src & 0x1F) << 25
        | (dst & 0x1F) << 20
        | (dport & 0x3F) << 14
        | (sport & 0x3F) << 8
        | (flags & 0xFF)
    )
    return struct.pack(">I", value)


def load_packets_from_sqlite(input_sqlite: Path) -> pl.DataFrame:
    """Load received packets from a SQLite database's "packet" table.

    The stored ``payload`` blob excludes the CSP header, so the header is
    re-encoded from the ``csp_*`` columns and prepended to each row's hex.

    ``session_dir`` looks like "/FrontierSat/satnogs_archive/14391497"; the
    trailing integer is the SatNOGS observation ID.
    """
    logger.info(f"Reading: {input_sqlite}")

    with sqlite3.connect(input_sqlite) as conn:
        rows = conn.execute(SQLITE_PACKETS_QUERY).fetchall()

    records = [
        {
            "received_timestamp": ts_received,
            "hex_payload": (
                encode_csp_header(
                    prio=csp_prio,
                    src=csp_src,
                    dst=csp_dst,
                    dport=csp_dport,
                    sport=csp_sport,
                    flags=csp_flags,
                ).hex()
                + payload_hex
            ),
            "session_dir": session_dir,
        }
        for (
            ts_received,
            payload_hex,
            session_dir,
            csp_src,
            csp_dst,
            csp_dport,
            csp_sport,
            csp_prio,
            csp_flags,
        ) in rows
    ]

    df = pl.DataFrame(
        records,
        schema=["received_timestamp", "hex_payload", "session_dir"],
    ).with_columns(
        observation_id=(
            pl.col("session_dir")
            .str.extract(r"(\d+)/?$", 1)
            .cast(pl.Int64, strict=False)
        ),
        ground_station=pl.lit(None, dtype=pl.String),
    )
    df = df.drop("session_dir")

    logger.info(f"Read: {len(df)} rows")
    return df


def _bulk_data_hex_to_general_message(hex_str: str) -> str:
    """Render bulk downlink data as text, or a placeholder if it looks like binary data.

    Bulk file downlinks (eg. images, firmware) are not text, so naively decoding them as
    UTF-8 produces a wall of replacement characters in the CSV. Detect that case with a
    printable-character heuristic and substitute a short placeholder instead.
    """
    data_bytes = bytes.fromhex(hex_str)
    if not data_bytes:
        return ""

    try:
        text = data_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return f"<BINARY DATA: {len(data_bytes)} bytes>"

    printable_count = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    if printable_count / len(text) < 0.85:  # noqa: PLR2004
        return f"<BINARY DATA: {len(data_bytes)} bytes>"

    return text


def load_sent_tcmd_from_sqlite(input_sqlite: Path) -> pl.DataFrame:
    """Load uplinked telecommands from a SQLite database's "sent_tcmd" table.

    These rows aren't decoded from a received hex payload -- they record
    telecommands as they were sent -- so they're tagged with packet_type
    "TCMD_UPLINK" and merged in alongside the decoded downlink packets.

    Additionally, for every row with a known execution time (``tsexec_ms``),
    a second virtual "TCMD_UPLINK_EXEC" row is emitted at that execution
    timestamp. Identical commands that were uplinked (and executed) more than
    once collapse to a single TCMD_UPLINK_EXEC row, keyed on
    (command text, execution timestamp).
    """
    logger.info(f"Reading sent_tcmd from: {input_sqlite}")

    with sqlite3.connect(input_sqlite) as conn:
        rows = conn.execute(SQLITE_SENT_TCMD_QUERY).fetchall()

    df = pl.DataFrame(
        rows,
        schema=[
            "tcmd_uplink_id",
            "tcmd_uplink_ts_sent_ms",
            "tcmd_uplink_tsexec_ms",
            "tcmd_uplink_command_text",
            "tcmd_uplink_tx_freq_hz",
            "tcmd_uplink_tx_gain_db",
            "tcmd_uplink_source_tool",
            "tcmd_uplink_source_run",
            "tcmd_uplink_ts_transmitted",
        ],
        orient="row",
    ).with_columns(
        packet_type=pl.lit("TCMD_UPLINK"),
        received_timestamp=pl.col("tcmd_uplink_ts_transmitted"),
        observation_id=pl.lit(None, dtype=pl.Int64),
        ground_station=pl.lit(None, dtype=pl.String),
    )

    df_exec = (
        df.filter(pl.col("tcmd_uplink_tsexec_ms").is_not_null())
        .unique(
            subset=["tcmd_uplink_command_text", "tcmd_uplink_tsexec_ms"],
            keep="first",
        )
        .with_columns(
            packet_type=pl.lit("TCMD_UPLINK_EXEC"),
            # If the reported exec time precedes the send time (clock skew /
            # bad onboard clock at exec time), fall back to the send time.
            received_timestamp=(
                pl.when(
                    pl.col("tcmd_uplink_tsexec_ms") < pl.col("tcmd_uplink_ts_sent_ms")
                )
                .then(pl.col("tcmd_uplink_ts_transmitted"))
                .otherwise(
                    pl.col("tcmd_uplink_tsexec_ms").map_elements(
                        epoch_ms_to_iso, return_dtype=pl.String
                    )
                )
            ),
        )
    )

    df = pl.concat([df, df_exec], how="vertical")

    logger.info(
        f"Read: {len(df)} sent_tcmd rows (incl. {len(df_exec)} TCMD_UPLINK_EXEC rows)"
    )
    return df


def decode_to_csv(
    df: pl.DataFrame,
    output_csv: Path,
    *,
    sort_setting: Literal["no_sort", "by_timestamp"],
    df_extra: pl.DataFrame | None = None,
) -> None:
    """Decode CTS-SAT-1 packets already loaded into a dataframe."""
    # Create a separate dataframe of decoded packets.
    decoded_packets: dict[str, dict[str, Any]] = {
        # Keys: hex_str, Vals: decoded packet
    }
    for hex_val in df["hex_payload"].unique():
        decoded = decode_packet_safe(hex_val)
        if decoded:
            decoded_packets[hex_val] = decoded

    # Build a dataframe from the decoded results and join back.
    df_decoded = pl.DataFrame(
        [
            {"hex_payload": hex_val, **decoded_packet}
            for hex_val, decoded_packet in decoded_packets.items()
        ],
        infer_schema_length=None,  # Use all rows.
    )

    df = df.join(
        df_decoded,
        on="hex_payload",
        how="left",
        validate="m:1",  # Same payload can be received many times.
        maintain_order="left_right",  # Preserve the order of the original CSV.
    )
    del df_decoded

    if df_extra is not None and len(df_extra) > 0:
        df = pl.concat([df, df_extra], how="diagonal_relaxed")

    # Add a general "as decoded message" column for logs, telecommand responses,
    # bulk file transfers, and uplinked telecommands.
    df = df.with_columns(
        general_message=pl.coalesce(
            pl.col("log_message"),
            pl.col("tcmd_response_text"),
            pl.col("bulk_data_hex").map_elements(
                _bulk_data_hex_to_general_message,
                return_dtype=pl.String,
            ),
            pl.col("tcmd_uplink_command_text"),
        )
    )

    # Hard-code the column order here.
    force_start_col_names = [
        "received_timestamp",
        "observation_id",
        "packet_type",
        "general_message",
    ]

    end_cols = ["log_message", "tcmd_response_text", "bulk_data_hex", "hex_payload"]
    tcmd_col_names = [
        col for col in df.columns if col.startswith("tcmd_") and (col not in end_cols)
    ]
    bulk_col_names = [
        col for col in df.columns if col.startswith("bulk_") and (col not in end_cols)
    ]

    df = df.select(
        *force_start_col_names,
        *tcmd_col_names,
        *bulk_col_names,
        *(
            # All the general columns (includes the beacons).
            OrderedSet(df.columns)
            - set(force_start_col_names)
            - set(tcmd_col_names)
            - set(bulk_col_names)
            - set(end_cols)
        ),
        *end_cols,
    )

    if sort_setting == "by_timestamp":
        df = df.sort("received_timestamp", descending=True)
        logger.debug("Sorted by received_timestamp")
    elif sort_setting == "no_sort":
        logger.debug("Not sorted")
    else:
        assert_never(sort_setting)

    if output_csv:
        df.write_csv(output_csv)
        logger.info(f"  CSV  → {output_csv}")

    # Pretty-print the most recent BEACON_BASIC packet.
    df_beacons = df.filter(pl.col("packet_type") == pl.lit("BEACON_BASIC")).drop(
        col for col in df.columns if col.startswith(("tcmd_", "bulk_", "log_"))
    )
    if len(df_beacons) > 0:
        print("\n\n-- Last BEACON_BASIC packet -------------------------------------")  # noqa: T201
        for k, v in df_beacons.sort("received_timestamp").tail(1).to_dicts()[0].items():
            print(f"  {k:<44} {v}")  # noqa: T201

    # Summary counts by packet type.
    print("\n\n-- Packet type summary ------------------------------------------")  # noqa: T201
    df_summary = (
        df.group_by("packet_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    logger.info(f"Packet type summary: {df_summary}")


def run(
    input_csv: Path | None = None,
    input_sqlite: Path | None = None,
    output_csv: Path | None = None,
) -> None:
    """Decode CTS-SAT-1 packets from a SatNOGS-style pipe-delimited CSV or a
    SQLite database of received packets.

    Provide exactly one of ``input_csv`` or ``input_sqlite``.

    If ``output_csv`` is not given, the decoded packets will be written to a new
    file with the same stem as the input file but with "-decoded" appended.
    """
    if (input_csv is None) == (input_sqlite is None):
        msg = "Provide exactly one of --input-csv or --input-sqlite."
        raise ValueError(msg)

    df_extra = None
    if input_csv is not None:
        input_path = input_csv
        df = load_packets_from_csv(input_csv)
    else:
        assert input_sqlite is not None
        input_path = input_sqlite
        df = load_packets_from_sqlite(input_sqlite)
        df_extra = load_sent_tcmd_from_sqlite(input_sqlite)

    if output_csv is not None:
        output_csv_path = output_csv
    else:
        output_csv_path = input_path.with_stem(
            input_path.stem + "-decoded"
        ).with_suffix(".csv")

    decode_to_csv(
        df,
        output_csv=output_csv_path,
        sort_setting="by_timestamp" if input_sqlite else "no_sort",
        df_extra=df_extra,
    )


def main() -> None:
    """Entry point."""
    tyro.cli(run)


if __name__ == "__main__":
    main()
