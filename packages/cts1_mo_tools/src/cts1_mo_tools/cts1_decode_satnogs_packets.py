"""
CTS-SAT-1 Packet Decoder (from SatNOGS data).

Decodes COMMS_*_packet_t structs from either a SatNOGS-style CSV export or a
SQLite database of received packets.

CSV format (pipe-delimited):
  timestamp | hex_payload | observation_id | ground_station

SQLite format: a "packet" table with (at least) "ts_received", "payload",
"rs_errs", and "session_dir" columns.
"""

import math
import sqlite3
import struct
import zlib
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal, assert_never

import orjson
import polars as pl
import polars_hash
import tyro
from loguru import logger
from ordered_set import OrderedSet

# -- Constants ----------------------------------------------------------------

CSP_HEADER_SIZE = 4
CSP_CRC32C_SIZE = 4  # trailing CRC-32C (Castagnoli) appended by libcsp, if present
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

# GNSS_bestxyzb_downlink_packet_t layout (after the CSP header):
#   uint8_t  packet_type        1
#   uint16_t downlink_seq_num   2
#   uint16_t ring_position      2
#   uint8_t  bestxyzb_data[GNSS_SAMPLE_SIZE]  (raw NovAtel OEM7 BESTXYZB log)
GNSS_DOWNLINK_HEADER_FMT = "<B H H"
GNSS_DOWNLINK_HEADER_SIZE = struct.calcsize(GNSS_DOWNLINK_HEADER_FMT)

# NovAtel OEM7 binary message header ("Binary Message Header" in the OEM7
# Commands and Logs Reference Manual). All fields are little-endian.
NOVATEL_SYNC_BYTES = b"\xaa\x44\x12"
NOVATEL_HEADER_FMT = (
    "<"
    "3s"  # sync (0xAA 0x44 0x12)
    "B"  # header_length (normally 28)
    "H"  # message_id (241 for BESTXYZ)
    "B"  # message_type (bits 0-4: source; bits 5-6: format; bit 7: response)
    "B"  # port_address
    "H"  # message_length (body only; excludes header and CRC)
    "H"  # sequence (counts down from N-1 to 0 for related logs)
    "B"  # idle_time (divide by 2 for percent)
    "B"  # time_status (GPS Reference Time Status enum)
    "H"  # gps_week
    "I"  # gps_week_ms (milliseconds from the start of the GPS week)
    "I"  # receiver_status (bitfield)
    "H"  # reserved
    "H"  # receiver_sw_version (build number)
)
NOVATEL_HEADER_SIZE = struct.calcsize(NOVATEL_HEADER_FMT)  # 28 bytes
NOVATEL_CRC_SIZE = 4
NOVATEL_BESTXYZ_MESSAGE_ID = 241

# BESTXYZ log body (message ID 241). Offsets relative to the end of the header.
# Note: the manual lists the "P-sol status" offset as "H+3", which is a typo -
# it's a 4-byte enum at H+0 (the next field, "pos type", is at H+4).
BESTXYZ_BODY_FMT = (
    "<"
    "I"  # H+0   p_sol_status (Solution Status enum)
    "I"  # H+4   pos_type (Position or Velocity Type enum)
    "d"  # H+8   p_x_m (ECEF)
    "d"  # H+16  p_y_m (ECEF)
    "d"  # H+24  p_z_m (ECEF)
    "f"  # H+32  p_x_stddev_m
    "f"  # H+36  p_y_stddev_m
    "f"  # H+40  p_z_stddev_m
    "I"  # H+44  v_sol_status (Solution Status enum)
    "I"  # H+48  vel_type (Position or Velocity Type enum)
    "d"  # H+52  v_x_m_per_s (ECEF)
    "d"  # H+60  v_y_m_per_s (ECEF)
    "d"  # H+68  v_z_m_per_s (ECEF)
    "f"  # H+76  v_x_stddev_m_per_s
    "f"  # H+80  v_y_stddev_m_per_s
    "f"  # H+84  v_z_stddev_m_per_s
    "4s"  # H+88  stn_id (base station ID)
    "f"  # H+92  v_latency_sec
    "f"  # H+96  diff_age_sec
    "f"  # H+100 sol_age_sec
    "B"  # H+104 num_svs (tracked)
    "B"  # H+105 num_soln_svs (used in solution)
    "B"  # H+106 num_gg_l1 (with L1/E1/B1 signals used in solution)
    "B"  # H+107 num_soln_multi_svs (with multi-frequency signals used in solution)
    "B"  # H+108 reserved
    "B"  # H+109 ext_sol_stat (Extended Solution Status bitfield)
    "B"  # H+110 galileo_beidou_sig_mask
    "B"  # H+111 gps_glonass_sig_mask
)
BESTXYZ_BODY_SIZE = struct.calcsize(BESTXYZ_BODY_FMT)  # 112 bytes

BESTXYZ_BODY_FIELD_NAMES = [
    "p_sol_status",
    "pos_type",
    "p_x_m",
    "p_y_m",
    "p_z_m",
    "p_x_stddev_m",
    "p_y_stddev_m",
    "p_z_stddev_m",
    "v_sol_status",
    "vel_type",
    "v_x_m_per_s",
    "v_y_m_per_s",
    "v_z_m_per_s",
    "v_x_stddev_m_per_s",
    "v_y_stddev_m_per_s",
    "v_z_stddev_m_per_s",
    "stn_id",
    "v_latency_sec",
    "diff_age_sec",
    "sol_age_sec",
    "num_svs",
    "num_soln_svs",
    "num_gg_l1",
    "num_soln_multi_svs",
    "reserved",
    "ext_sol_stat",
    "galileo_beidou_sig_mask",
    "gps_glonass_sig_mask",
]

# GPS time -> UTC. GPS time is ahead of UTC by the accumulated leap seconds
# (18 s since 2017-01-01; no further leap seconds have been scheduled since).
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
GPS_UTC_LEAP_SECONDS = 18

# WGS84 ellipsoid, for converting the ECEF position to latitude/longitude/height.
WGS84_A_M = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

# Heights above the WGS84 ellipsoid outside this range can't be CTS-SAT-1 (LEO,
# ~500 km). With no valid fix, the receiver reports a placeholder position
# (seen in flight as a fixed point ~107,800 km up), sometimes even with a
# SINGLE position type; this keeps it from passing as a real lat/lon.
GNSS_PLAUSIBLE_HEIGHT_RANGE_KM = (0.0, 2_000.0)

# -- Enum maps ----------------------------------------------------------------

PACKET_TYPE_MAP = {
    0x01: "BEACON_BASIC",
    0x03: "LOG_MESSAGE",
    0x04: "TCMD_RESPONSE",
    0x10: "BULK_FILE_DOWNLINK",
    0x20: "BEACON_EXTENDED",
    0x30: "GNSS_BESTXYZB_SAMPLE",
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
# Bit index in `eps_enabled_channels_bitfield` -> channel name. Indices match
# `EPS_CHANNEL_enum_t`, and names match `EPS_channel_to_str()` in the flight software.
EPS_CHANNEL_MAP = {
    0: "VBATT_STACK",
    1: "5V_STACK",
    2: "5V_CH2_UNUSED",
    3: "5V_CH3_UNUSED",
    4: "5V_MPI",
    5: "3V3_STACK",
    6: "3V3_CAMERA",
    7: "3V3_UHF_ANTENNA_DEPLOY",
    8: "3V3_GNSS",
    9: "VBATT_CH9_UNUSED",
    10: "VBATT_CH10_UNUSED",
    11: "VBATT_CH11_UNUSED",
    12: "12V_MPI",
    13: "12V_BOOM",
    14: "3V3_CH14_UNUSED",
    15: "3V3_CH15_UNUSED",
    16: "28V6_CH16_UNUSED",
}
# Channels powering the main stack; normally always on together.
EPS_STACK_CHANNELS = frozenset({"VBATT_STACK", "5V_STACK", "3V3_STACK"})
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

# NovAtel OEM7 enums/bitfields, from the OEM7 Commands and Logs Reference Manual.
# "GPS Reference Time Status" (binary header `time_status` field).
GNSS_TIME_STATUS_MAP = {
    20: "UNKNOWN",
    60: "APPROXIMATE",
    80: "COARSEADJUSTING",
    100: "COARSE",
    120: "COARSESTEERING",
    130: "FREEWHEELING",
    140: "FINEADJUSTING",
    160: "FINE",
    170: "FINEBACKUPSTEERING",
    180: "FINESTEERING",
    200: "SATTIME",
}
# "Solution Status" (BESTPOS table; used by BESTXYZ `P-sol status`/`V-sol status`).
GNSS_SOLUTION_STATUS_MAP = {
    0: "SOL_COMPUTED",
    1: "INSUFFICIENT_OBS",
    2: "NO_CONVERGENCE",
    3: "SINGULARITY",
    4: "COV_TRACE",
    5: "TEST_DIST",
    6: "COLD_START",
    7: "V_H_LIMIT",
    8: "VARIANCE",
    9: "RESIDUALS",
    13: "INTEGRITY_WARNING",
    18: "PENDING",
    19: "INVALID_FIX",
    20: "UNAUTHORIZED",
    22: "INVALID_RATE",
}
# "Position or Velocity Type" (BESTPOS table; used by `pos type`/`vel type`).
GNSS_POSITION_VELOCITY_TYPE_MAP = {
    0: "NONE",
    1: "FIXEDPOS",
    2: "FIXEDHEIGHT",
    8: "DOPPLER_VELOCITY",
    16: "SINGLE",
    17: "PSRDIFF",
    18: "WAAS",
    19: "PROPAGATED",
    32: "L1_FLOAT",
    34: "NARROW_FLOAT",
    48: "L1_INT",
    49: "WIDE_INT",
    50: "NARROW_INT",
    52: "INS_SBAS",
    53: "INS_PSRSP",
    54: "INS_PSRDIFF",
    55: "INS_RTKFLOAT",
    56: "INS_RTKFIXED",
    67: "EXT_CONSTRAINED",
    68: "PPP_CONVERGING",
    69: "PPP",
    70: "OPERATIONAL",
    71: "WARNING",
    72: "OUT_OF_BOUNDS",
    73: "INS_PPP_CONVERGING",
    74: "INS_PPP",
    77: "PPP_BASIC_CONVERGING",
    78: "PPP_BASIC",
    79: "INS_PPP_BASIC_CONVERGING",
    80: "INS_PPP_BASIC",
}
GNSS_POSITION_VELOCITY_TYPE_MAP_INV = {
    v: k for k, v in GNSS_POSITION_VELOCITY_TYPE_MAP.items()
}
# "Receiver Status" (RXSTATUS table; binary header `receiver_status` field).
# Names describe the bit=1 meaning. Bits 25-26 are the status word's version
# number (not a flag), so they're omitted and never listed.
GNSS_RECEIVER_STATUS_BITS: list[tuple[int, str]] = [
    (0, "ERROR"),
    (1, "TEMPERATURE_WARNING"),
    (2, "VOLTAGE_SUPPLY_WARNING"),
    (3, "ANTENNA_NOT_POWERED"),
    (4, "LNA_FAILURE"),
    (5, "ANTENNA_OPEN_CIRCUIT"),
    (6, "ANTENNA_SHORT_CIRCUIT"),
    (7, "CPU_OVERLOAD"),
    (8, "COM_BUFFER_OVERRUN"),
    (9, "SPOOFING_DETECTED"),
    (10, "RESERVED_BIT_10"),
    (11, "LINK_OVERRUN"),
    (12, "INPUT_OVERRUN"),
    (13, "AUX_TRANSMIT_OVERRUN"),
    (14, "ANTENNA_GAIN_OUT_OF_RANGE"),
    (15, "JAMMER_DETECTED"),
    (16, "INS_RESET"),
    (17, "IMU_COMMS_FAILURE"),
    (18, "ALMANAC_OR_UTC_INVALID"),
    (19, "POSITION_SOLUTION_INVALID"),
    (20, "POSITION_FIXED"),
    (21, "CLOCK_STEERING_DISABLED"),
    (22, "CLOCK_MODEL_INVALID"),
    (23, "EXTERNAL_OSCILLATOR_LOCKED"),
    (24, "SOFTWARE_RESOURCE_WARNING"),
    (27, "HDR_TRACKING"),
    (28, "DIGITAL_FILTERING_ENABLED"),
    (29, "AUX3_EVENT"),
    (30, "AUX2_EVENT"),
    (31, "AUX1_EVENT"),
]
GNSS_RECEIVER_STATUS_VERSION_SHIFT = 25
GNSS_RECEIVER_STATUS_VERSION_MASK = 0x3
# "Extended Solution Status" (BESTPOS table). Bits 1-3 are the pseudorange
# iono correction enum (below); the rest are single-bit flags.
GNSS_EXT_SOL_STAT_BITS: list[tuple[int, str]] = [
    (0, "RTK_VERIFIED_OR_GLIDE"),
    (4, "RTK_ASSIST_ACTIVE"),
    (5, "ANTENNA_INFO_MISSING"),
    (6, "RESERVED_BIT_6"),
    (7, "TERRAIN_COMPENSATION_USED"),
]
GNSS_IONO_CORRECTION_MAP = {
    0: "UNKNOWN_OR_DEFAULT_KLOBUCHAR",
    1: "KLOBUCHAR_BROADCAST",
    2: "SBAS_BROADCAST",
    3: "MULTI_FREQUENCY_COMPUTED",
    4: "PSRDIFF_CORRECTION",
    5: "NOVATEL_BLENDED_IONO",
}
# "Galileo and BeiDou Signal-Used Mask" (BESTPOS table).
GNSS_GALILEO_BEIDOU_SIGNAL_BITS: list[tuple[int, str]] = [
    (0, "GALILEO_E1"),
    (1, "GALILEO_E5A"),
    (2, "GALILEO_E5B"),
    (3, "GALILEO_ALTBOC"),
    (4, "BEIDOU_B1"),
    (5, "BEIDOU_B2"),
    (6, "BEIDOU_B3"),
    (7, "GALILEO_E6"),
]
# "GPS and GLONASS Signal-Used Mask" (BESTPOS table). Bits 3 and 7 are reserved.
GNSS_GPS_GLONASS_SIGNAL_BITS: list[tuple[int, str]] = [
    (0, "GPS_L1"),
    (1, "GPS_L2"),
    (2, "GPS_L5"),
    (3, "RESERVED_BIT_3"),
    (4, "GLONASS_L1"),
    (5, "GLONASS_L2"),
    (6, "GLONASS_L3"),
    (7, "RESERVED_BIT_7"),
]

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


# -- OBC ADC battery percentage -----------------------------------------------

# The OBC's own ADC reads the battery voltage slightly differently than the EPS
# does, so the OBC reading is calibrated onto the EPS scale before it's turned
# into a percentage. Regression of the EPS battery ADC (y) on the OBC ADC (x),
# over the 7,358 beacons where both readings were non-zero:
#     y = 0.988x + 0.372   (R^2 = 0.9855)
# Source: https://github.com/CalgaryToSpace/CTS-SAT-1-Ground-Station/issues/48
OBC_ADC_BATTERY_CALIBRATION_SLOPE = 0.988
OBC_ADC_BATTERY_CALIBRATION_INTERCEPT_V = 0.372

# Battery voltage endpoints, matching `EPS_convert_battery_voltage_to_percent()`
# in the flight software:
#   Source (low side) - 12.4V: SAFETY_VOLT_LOTHR on Page 93 of the EPS Software ICD.
#   EMLOPO_VOLT_HITHR on Page 99 of the EPS Software ICD.
BATTERY_MIN_TOTAL_VOLTAGE_V = 12.4
BATTERY_MAX_TOTAL_VOLTAGE_V = 16.0


def convert_obc_adc_battery_voltage_to_percent(
    obc_adc_battery_voltage_volts: float,
) -> float | None:
    """Convert an OBC ADC battery voltage reading to a battery percentage.

    The reading is first calibrated onto the EPS battery ADC's scale, then
    converted using the same linear percentage logic as the flight software.

    Args:
        obc_adc_battery_voltage_volts: OBC ADC battery voltage, in volts.

    Returns:
        Battery percentage. Nominally between 0 and 100, but can exceed 100% if
        the battery voltage is above the maximum voltage, and can be less than
        0% if it's below the minimum voltage. `None` if the OBC ADC reading is
        non-positive, which means the OBC has no valid reading.
    """
    if obc_adc_battery_voltage_volts <= 0:
        return None

    calibrated_voltage_volts = (
        OBC_ADC_BATTERY_CALIBRATION_SLOPE * obc_adc_battery_voltage_volts
        + OBC_ADC_BATTERY_CALIBRATION_INTERCEPT_V
    )

    calc = (calibrated_voltage_volts - BATTERY_MIN_TOTAL_VOLTAGE_V) / (
        BATTERY_MAX_TOTAL_VOLTAGE_V - BATTERY_MIN_TOTAL_VOLTAGE_V
    )
    return round(calc * 100.0, 2)


def e(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"UNKNOWN({value})")


def e_numbered(mapping: dict[int, str], value: int) -> str:
    """Like `e()`, but merges the number in, e.g. "2 - Y-Thomson spin"."""
    return f"{value} - {e(mapping, value)}"


def decode_eps_enabled_channels(bitfield: int) -> str:
    """Decode the 32-bit EPS enabled channels bitfield to a JSON list of names.

    Set bits with no entry in EPS_CHANNEL_MAP are listed as "INVALID_CHANNEL(<n>)",
    after the fallback in `EPS_channel_to_str()`.

    The stack channels are normally always on, so when all of them are enabled
    they're replaced by a single "STACK_X3" entry at the end of the list.
    """
    enabled = [
        EPS_CHANNEL_MAP.get(bit_num, f"INVALID_CHANNEL({bit_num})")
        for bit_num in range(32)
        if (bitfield >> bit_num) & 1
    ]
    if EPS_STACK_CHANNELS.issubset(enabled):
        enabled = [name for name in enabled if name not in EPS_STACK_CHANNELS]
        enabled.append("STACK_X3")
    return orjson.dumps(enabled).decode()


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
        "adcs_powered_list": orjson.dumps(enabled).decode(),
        "adcs_sun_above_local_horizon": bit(23),
        "adcs_errors": orjson.dumps(errors).decode(),
        "adcs_flags": orjson.dumps(flags).decode(),
    }


# -- CRC ------------------------------------------------------------------


def crc32c(data: bytes, crc: int = 0xFFFFFFFF) -> int:
    """CRC-32C (Castagnoli) — same variant used by iSCSI/SCTP and by libcsp."""
    poly = 0x82F63B78  # reflected form of 0x1EDC6F41
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1)
            crc = (crc >> 1) ^ (poly & mask)
    return crc ^ 0xFFFFFFFF


def verify_csp_packet_crc32c(packet: bytes) -> tuple[bool, int, int]:
    """Split `packet` into payload + trailing 4-byte CRC-32C, and verify it.

    Splits the packet into payload + trailing 4-byte CRC, recomputes CRC-32C
    over the payload, and compares. Returns (is_valid, computed_crc, received_crc).
    """
    if len(packet) <= CSP_CRC32C_SIZE:
        return False, 0, 0

    payload, received = packet[:-CSP_CRC32C_SIZE], packet[-CSP_CRC32C_SIZE:]
    computed = crc32c(payload)
    received_int = int.from_bytes(received, "big")
    return computed == received_int, computed, received_int


# -- Decoders -----------------------------------------------------------------


def epoch_ms_to_utc_isoformat(epoch_ms: int) -> str | None:
    """`epoch_ms` as a plain ISO 8601 UTC timestamp with millisecond
    precision (e.g. `2023-11-14T22:13:20.000+00:00`), or None if it's
    unset/invalid (not yet time-synced -- see `MAX_VALID_EPOCH_MS`).
    """
    if epoch_ms <= 0 or epoch_ms > MAX_VALID_EPOCH_MS:
        return None
    # Integer timedelta math, not `fromtimestamp(ms / 1000)`: avoids float
    # rounding turning e.g. `.123` into `.122999`.
    dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=epoch_ms)
    return dt.isoformat(timespec="milliseconds")


def decode_beacon_basic_packet(
    payload: bytes, _full_payload: bytes | None = None
) -> dict[str, Any]:
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

    rf: dict[str, Any] = dict(zip(BEACON_FIELD_NAMES, vals, strict=True))
    fm_raw = payload[BEACON_FIXED_SIZE : BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE]
    friendly = fm_raw.split(b"\x00")[0].decode("utf-8", errors="replace")
    sat_name = rf["satellite_name"].decode("ascii", errors="replace").rstrip("\x00")
    epoch_ms = rf["unix_epoch_time_ms"]

    utc_time = epoch_ms_to_utc_isoformat(epoch_ms)

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
        "eps_enabled_channels_list": decode_eps_enabled_channels(
            rf["eps_enabled_channels_bitfield"]
        ),
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


def decode_beacon_extended_packet(
    payload: bytes, _full_payload: bytes | None = None
) -> dict[str, Any]:
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

    rf: dict[str, Any] = dict(zip(BEACON_FIELD_NAMES, vals, strict=True))

    fm_raw = payload[BEACON_FIXED_SIZE : BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE]
    friendly = fm_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    end_offset = BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE
    end_raw = payload[end_offset : end_offset + END_MESSAGE_SIZE]
    end_version_number = end_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    ext_offset = end_offset + END_MESSAGE_SIZE
    ext_vals = struct.unpack_from(EXTENDED_FMT, payload, ext_offset)
    ef: dict[str, Any] = dict(zip(EXTENDED_FIELD_NAMES, ext_vals, strict=True))

    sat_name = rf["satellite_name"].decode("ascii", errors="replace").rstrip("\x00")
    epoch_ms = rf["unix_epoch_time_ms"]

    utc_time = epoch_ms_to_utc_isoformat(epoch_ms)

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
        "obc_adc_battery_percent": convert_obc_adc_battery_voltage_to_percent(
            ef["obc_adc_battery_voltage_mV"] / 1000.0
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
        "eps_enabled_channels_list": decode_eps_enabled_channels(
            rf["eps_enabled_channels_bitfield"]
        ),
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


def decode_log_message_packet(
    payload: bytes, _full_payload: bytes | None = None
) -> dict[str, Any]:
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


def decode_tcmd_response_packet(
    payload: bytes, _full_payload: bytes | None = None
) -> dict[str, Any]:
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


def decode_bulk_file_downlink_packet(
    payload: bytes, full_payload: bytes, *, crc_valid: bool | None = None
) -> dict[str, Any]:
    """Decode a COMMS_bulk_file_downlink_packet_t payload (CSP header already stripped).

    `crc_valid` is whether `full_payload` ends in a valid CSP CRC-32C, if the
    caller already knows (computed here otherwise).

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

    data_bytes = payload[BULK_DOWNLINK_HEADER_SIZE:]

    # If the full_payload has a CRC32C on its end, we must chop it off. Critical for the
    # final packet in any bulk downlink series.
    # This branch is always true in the nominal re-demodulating pipeline, but SatNOGS
    # stations are inconsistent whether they've pre-chopped the CRC. Thus, we must
    # check and chop here.
    if crc_valid is None:
        crc_valid, _computed, _received = verify_csp_packet_crc32c(full_payload)
    if crc_valid:
        data_bytes = data_bytes[:-4]

    return {
        "packet_type": "BULK_FILE_DOWNLINK",
        "bulk_file_offset": file_offset,
        "bulk_data_len": len(data_bytes),
        "bulk_data_hex": data_bytes.hex(),
    }


def novatel_crc32(data: bytes) -> int:
    """NovAtel's 32-bit CRC (`CalculateBlockCRC32` in the OEM7 manual).

    Same reflected polynomial (0xEDB88320) as zlib's CRC-32, but with an
    initial value of 0 and no final XOR. Expressed via zlib by pre-seeding it
    so that its internal register starts at 0, then undoing its final XOR.
    """
    return zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF


def _bits_to_list(value: int, bits: list[tuple[int, str]]) -> list[str]:
    return [name for bit_num, name in bits if (value >> bit_num) & 1]


def _bits_to_json_list(value: int, bits: list[tuple[int, str]]) -> str:
    return orjson.dumps(_bits_to_list(value, bits)).decode()


def gps_time_to_utc_isoformat(
    gps_week: int, gps_week_ms: int, *, minus_sec: float = 0.0
) -> str:
    """GPS week + milliseconds-of-week (minus `minus_sec`) as an ISO 8601 UTC
    timestamp, with millisecond precision.
    """
    dt = GPS_EPOCH + timedelta(
        weeks=gps_week,
        milliseconds=gps_week_ms - round(minus_sec * 1000),
        seconds=-GPS_UTC_LEAP_SECONDS,
    )
    return dt.isoformat(timespec="milliseconds")


def ecef_to_geodetic(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    """Convert WGS84 ECEF coordinates to (latitude_deg, longitude_deg, height_m).

    Height is above the WGS84 ellipsoid. Uses the standard fixed-point
    iteration on latitude, which converges to sub-mm within a few iterations
    for any point near the Earth (including LEO).
    """
    lon = math.atan2(y_m, x_m)
    p = math.hypot(x_m, y_m)
    lat = math.atan2(z_m, p * (1 - WGS84_E2))
    height = 0.0
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A_M / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
        # Stable at all latitudes (unlike `p / cos(lat) - n`, which fails at poles).
        height = p * math.cos(lat) + z_m * sin_lat - WGS84_A_M**2 / n
        lat = math.atan2(z_m, p * (1 - WGS84_E2 * n / (n + height)))
    return math.degrees(lat), math.degrees(lon), height


def decode_bestxyzb(raw: bytes) -> dict[str, Any]:
    """Decode a raw NovAtel OEM7 BESTXYZB binary log (header + body [+ CRC]).

    See "BESTXYZ" and "Binary Message Header" in the OEM7 Commands and Logs
    Reference Manual. All returned keys are prefixed with "gnss_". Rarely-useful
    fields are packed into the `gnss_misc_json` column rather than getting their own.

    `gnss_crc_valid` is None if the 4-byte trailing CRC wasn't included.
    """
    if len(raw) < NOVATEL_HEADER_SIZE:
        msg = (
            f"Too short for a NovAtel binary header: {len(raw)} bytes "
            f"(need {NOVATEL_HEADER_SIZE})"
        )
        raise ValueError(msg)

    (
        sync,
        header_length,
        message_id,
        message_type,
        port_address,
        message_length,
        sequence,
        idle_time,
        time_status,
        gps_week,
        gps_week_ms,
        receiver_status,
        _reserved,
        receiver_sw_version,
    ) = struct.unpack_from(NOVATEL_HEADER_FMT, raw, 0)

    if sync != NOVATEL_SYNC_BYTES:
        expected = NOVATEL_SYNC_BYTES.hex()
        msg = f"Bad NovAtel sync bytes: {sync.hex()} (expected {expected})"
        raise ValueError(msg)
    if message_id != NOVATEL_BESTXYZ_MESSAGE_ID:
        msg = f"Not a BESTXYZ log: message ID {message_id}"
        raise ValueError(msg)
    if header_length < NOVATEL_HEADER_SIZE:
        msg = f"NovAtel header_length too small: {header_length}"
        raise ValueError(msg)

    # Honour the header's own length field, in case a future header is longer.
    body_end = header_length + BESTXYZ_BODY_SIZE
    if len(raw) < body_end:
        msg = f"Too short for BESTXYZB: {len(raw)} bytes (need {body_end})"
        raise ValueError(msg)

    body_vals = struct.unpack_from(BESTXYZ_BODY_FMT, raw, header_length)
    b: dict[str, Any] = dict(zip(BESTXYZ_BODY_FIELD_NAMES, body_vals, strict=True))

    crc_valid: bool | None = None
    if len(raw) >= body_end + NOVATEL_CRC_SIZE:
        received_crc = int.from_bytes(
            raw[body_end : body_end + NOVATEL_CRC_SIZE], "little"
        )
        crc_valid = novatel_crc32(raw[:body_end]) == received_crc

    # The header time is when the log was generated. The reported position may
    # be an older solution (e.g. the last good fix, held after losing lock), so
    # back-date by the solution age to get when the position was actually solved.
    solution_utc_time = (
        gps_time_to_utc_isoformat(gps_week, gps_week_ms, minus_sec=b["sol_age_sec"])
        if gps_week > 0 and GNSS_TIME_STATUS_MAP.get(time_status) != "UNKNOWN"
        else None
    )

    # Derive geodetic coordinates for any real fix, including less-reliable ones
    # (e.g. a SINGLE fix held with INSUFFICIENT_OBS), but not the placeholder
    # position reported without one (see `GNSS_PLAUSIBLE_HEIGHT_RANGE_KM`).
    lat_deg = lon_deg = height_km = None
    if b["pos_type"] != GNSS_POSITION_VELOCITY_TYPE_MAP_INV["NONE"]:
        lat, lon, height_m = ecef_to_geodetic(b["p_x_m"], b["p_y_m"], b["p_z_m"])
        min_km, max_km = GNSS_PLAUSIBLE_HEIGHT_RANGE_KM
        if min_km <= height_m / 1000 <= max_km:
            lat_deg, lon_deg, height_km = (
                round(lat, 6),
                round(lon, 6),
                round(height_m / 1000, 3),
            )

    ext_sol_stat = b["ext_sol_stat"]

    # Fields that are constant, always zero, or redundant with another column in
    # practice. Packed into one JSON column to keep the table narrow.
    misc = {
        "message_id": message_id,
        "message_type": f"0x{message_type:02X}",
        "port_address": port_address,
        "message_length": message_length,
        "sequence": sequence,
        "receiver_status_bitfield": f"0x{receiver_status:08X}",
        "receiver_status_version": (
            (receiver_status >> GNSS_RECEIVER_STATUS_VERSION_SHIFT)
            & GNSS_RECEIVER_STATUS_VERSION_MASK
        ),
        "receiver_sw_version": receiver_sw_version,
        "velocity_latency_sec": round(b["v_latency_sec"], 3),
        "base_station_id": b["stn_id"]
        .split(b"\x00")[0]
        .decode("ascii", errors="replace"),
        "differential_age_sec": round(b["diff_age_sec"], 3),
        "num_svs_l1_in_solution": b["num_gg_l1"],
        "extended_solution_status": f"0x{ext_sol_stat:02X}",
        "extended_solution_flags": _bits_to_list(ext_sol_stat, GNSS_EXT_SOL_STAT_BITS),
    }

    return {
        # Header
        "gnss_crc_valid": crc_valid,
        "gnss_idle_time_percent": idle_time / 2,
        "gnss_time_status": e(GNSS_TIME_STATUS_MAP, time_status),
        "gnss_gps_week": gps_week,
        "gnss_gps_week_ms": gps_week_ms,
        "gnss_solution_utc_time": solution_utc_time,
        "gnss_receiver_status_flags": _bits_to_json_list(
            receiver_status, GNSS_RECEIVER_STATUS_BITS
        ),
        # Position, in ECEF coordinates
        "gnss_position_solution_status": e(GNSS_SOLUTION_STATUS_MAP, b["p_sol_status"]),
        "gnss_position_type": e(GNSS_POSITION_VELOCITY_TYPE_MAP, b["pos_type"]),
        "gnss_position_x_m": round(b["p_x_m"], 3),
        "gnss_position_y_m": round(b["p_y_m"], 3),
        "gnss_position_z_m": round(b["p_z_m"], 3),
        "gnss_position_x_stddev_m": round(b["p_x_stddev_m"], 3),
        "gnss_position_y_stddev_m": round(b["p_y_stddev_m"], 3),
        "gnss_position_z_stddev_m": round(b["p_z_stddev_m"], 3),
        # Position (derived: WGS84 geodetic)
        "gnss_latitude_deg": lat_deg,
        "gnss_longitude_deg": lon_deg,
        "gnss_height_above_ellipsoid_km": height_km,
        # Velocity, in ECEF coordinates
        "gnss_velocity_solution_status": e(GNSS_SOLUTION_STATUS_MAP, b["v_sol_status"]),
        "gnss_velocity_type": e(GNSS_POSITION_VELOCITY_TYPE_MAP, b["vel_type"]),
        "gnss_velocity_x_m_per_s": round(b["v_x_m_per_s"], 4),
        "gnss_velocity_y_m_per_s": round(b["v_y_m_per_s"], 4),
        "gnss_velocity_z_m_per_s": round(b["v_z_m_per_s"], 4),
        "gnss_velocity_x_stddev_m_per_s": round(b["v_x_stddev_m_per_s"], 4),
        "gnss_velocity_y_stddev_m_per_s": round(b["v_y_stddev_m_per_s"], 4),
        "gnss_velocity_z_stddev_m_per_s": round(b["v_z_stddev_m_per_s"], 4),
        # Derived: magnitude of the ECEF velocity (relative to the rotating Earth).
        "gnss_ecef_speed_m_per_s": round(
            math.sqrt(
                b["v_x_m_per_s"] ** 2 + b["v_y_m_per_s"] ** 2 + b["v_z_m_per_s"] ** 2
            ),
            4,
        ),
        # Solution details
        "gnss_solution_age_sec": round(b["sol_age_sec"], 3),
        "gnss_num_svs_tracked": b["num_svs"],
        "gnss_num_svs_in_solution": b["num_soln_svs"],
        "gnss_num_svs_multi_freq_in_solution": b["num_soln_multi_svs"],
        "gnss_pseudorange_iono_correction": e(
            GNSS_IONO_CORRECTION_MAP, (ext_sol_stat >> 1) & 0x7
        ),
        "gnss_galileo_beidou_signals_used": _bits_to_json_list(
            b["galileo_beidou_sig_mask"], GNSS_GALILEO_BEIDOU_SIGNAL_BITS
        ),
        "gnss_gps_glonass_signals_used": _bits_to_json_list(
            b["gps_glonass_sig_mask"], GNSS_GPS_GLONASS_SIGNAL_BITS
        ),
        "gnss_misc_json": orjson.dumps(misc).decode(),
    }


def decode_gnss_bestxyzb_sample_packet(
    payload: bytes, _full_payload: bytes | None = None
) -> dict[str, Any]:
    """Decode a GNSS_bestxyzb_downlink_packet_t payload (CSP header already stripped).

    Layout:
        uint8_t  packet_type        (1 byte, always 0x30)
        uint16_t downlink_seq_num   (2 bytes)
        uint16_t ring_position      (2 bytes)
        uint8_t  bestxyzb_data[GNSS_SAMPLE_SIZE]  (raw log; see `decode_bestxyzb`)
    """
    if len(payload) < GNSS_DOWNLINK_HEADER_SIZE:
        msg = (
            f"Too short for GNSS_BESTXYZB_SAMPLE: {len(payload)} bytes "
            f"(need at least {GNSS_DOWNLINK_HEADER_SIZE})"
        )
        raise ValueError(msg)

    packet_type, downlink_seq_num, ring_position = struct.unpack_from(
        GNSS_DOWNLINK_HEADER_FMT, payload, 0
    )

    if packet_type != PACKET_TYPE_MAP_INV["GNSS_BESTXYZB_SAMPLE"]:
        msg = (
            f"Unexpected packet_type byte for GNSS_BESTXYZB_SAMPLE: {packet_type:#04x}"
        )
        raise ValueError(msg)

    return {
        "packet_type": "GNSS_BESTXYZB_SAMPLE",
        "gnss_downlink_seq_num": downlink_seq_num,
        "gnss_ring_position": ring_position,
        **decode_bestxyzb(payload[GNSS_DOWNLINK_HEADER_SIZE:]),
    }


# Map packet_type byte → decoder function (payload = post-CSP bytes).
_PACKET_DECODERS = {
    0x01: decode_beacon_basic_packet,
    0x03: decode_log_message_packet,
    0x04: decode_tcmd_response_packet,
    0x10: decode_bulk_file_downlink_packet,
    0x20: decode_beacon_extended_packet,
    0x30: decode_gnss_bestxyzb_sample_packet,
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
    return decode_raw_packet_safe(raw)


def decode_raw_packet_safe(
    raw: bytes, *, crc_valid: bool | None = None
) -> dict[str, Any] | None:
    """`decode_packet_safe`, for a packet already parsed to bytes.

    `crc_valid` is whether `raw` ends in a valid CSP CRC-32C, if the caller
    already knows -- e.g. `decode_to_df`, which checks every packet's CRC
    at once with polars rather than one at a time in (slow) pure Python.
    """
    if len(raw) <= CSP_HEADER_SIZE:
        return None

    csp = raw[:CSP_HEADER_SIZE]
    payload = raw[CSP_HEADER_SIZE:]
    packet_type_byte = payload[0]

    if crc_valid is None:
        crc_valid, _crc_computed, _crc_received = verify_csp_packet_crc32c(raw)
    base = {"csp_header_hex": csp.hex(), "csp_crc_valid": crc_valid}

    decoder = _PACKET_DECODERS.get(packet_type_byte)
    if decoder is decode_bulk_file_downlink_packet:
        # The only decoder that needs the CRC verdict itself.
        decoder = partial(decode_bulk_file_downlink_packet, crc_valid=crc_valid)
    if decoder is not None:
        try:
            decoded = decoder(payload, raw)
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

    binary_string_msg = f"BINARY DATA: {len(data_bytes)} bytes"

    # Attempt to identify the type of binary data (can be many):
    if "0cffff0c" in hex_str.lower():  # MPI sync word.
        binary_string_msg += ", maybe MPI data"
    # MPI data could also match like: "uptime_ms":252000748,"timestamp":"1786257055000+3594260776_E","datetime":"2026-08-09T063612.182Z_E","timestamp_ms":1786257  # noqa: E501

    if "aa4412" in hex_str.lower():  # GNSS binary data sync word.
        binary_string_msg += ", maybe binary GNSS data"

    binary_string_msg = f"<{binary_string_msg}>"

    try:
        text = data_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return binary_string_msg

    printable_count = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    if printable_count / len(text) < 0.85:  # noqa: PLR2004
        return binary_string_msg

    return text


# Payloads decoded per chunk in `_decode_unique_payloads`: bounds how many
# per-packet Python dicts are alive at once, which otherwise dwarf the
# dataframe they end up in.
_DECODE_CHUNK_SIZE = 20_000


def _decode_unique_payloads(hex_payloads: pl.Series) -> pl.DataFrame:
    """One row per distinct, decodable `hex_payload`, with its decoded fields.

    The hex parsing and CSP CRC-32C check run vectorized in polars, for every
    payload at once; only the struct-unpacking itself runs per packet.
    """
    raw = pl.col("raw")
    payloads = (
        hex_payloads.unique(maintain_order=True)
        .to_frame("hex_payload")
        .with_columns(raw=pl.col("hex_payload").str.decode("hex", strict=False))
        # Not valid hex at all (`decode_packet_safe` returns None for those).
        .filter(raw.is_not_null())
        # Same verdict as `verify_csp_packet_crc32c`: CRC-32C over all but the
        # last 4 bytes must equal those last 4 bytes (big-endian).
        .with_columns(crc_body=raw.bin.head(-CSP_CRC32C_SIZE))
        .with_columns(
            crc_valid=(raw.bin.size() > CSP_CRC32C_SIZE)
            & (
                polars_hash.col("crc_body").nchash.crc32c(
                    return_binary=True, byte_order="big"
                )
                == raw.bin.tail(CSP_CRC32C_SIZE)
            )
        )
        .select("hex_payload", "raw", "crc_valid")
    )

    chunks: list[pl.DataFrame] = []
    for chunk in payloads.iter_slices(_DECODE_CHUNK_SIZE):
        rows: list[dict[str, Any]] = []
        for hex_val, raw_bytes, crc_valid in chunk.iter_rows():
            decoded = decode_raw_packet_safe(raw_bytes, crc_valid=crc_valid)
            if decoded:
                rows.append({"hex_payload": hex_val, **decoded})
        if rows:
            chunks.append(pl.DataFrame(rows, infer_schema_length=None))

    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks, how="diagonal_relaxed")


def decode_to_df(
    df: pl.DataFrame,
    *,
    sort_setting: Literal["no_sort", "by_timestamp"],
) -> pl.DataFrame:
    """Decode CTS-SAT-1 packets already loaded into a dataframe."""
    input_columns_at_start = df.columns

    # Create a separate dataframe of decoded packets, then join back.
    df_decoded = _decode_unique_payloads(df["hex_payload"])

    # `general_message` below unconditionally references these three --
    # guarantee they all exist even if this particular batch didn't happen
    # to contain any packet of that type (plausible, e.g. early in a
    # mission before any bulk file downlinks have occurred), and even if
    # nothing in the batch decoded at all (which would otherwise leave
    # `df_decoded` with no columns whatsoever, not even `hex_payload`,
    # since `pl.DataFrame([])` on an empty list of dicts has none).
    for required_col in (
        "hex_payload",
        "log_message",
        "tcmd_response_text",
        "bulk_data_hex",
    ):
        if required_col not in df_decoded.columns:
            df_decoded = df_decoded.with_columns(
                pl.lit(None, dtype=pl.String).alias(required_col)
            )

    df = df.join(
        df_decoded,
        on="hex_payload",
        how="left",
        validate="m:1",  # Same payload can be received many times.
        maintain_order="left_right",  # Preserve the order of the original CSV.
    )
    del df_decoded

    # Add a general "as decoded message" column for logs, telecommand responses, and
    # bulk file transfers.
    df = df.with_columns(
        general_message=pl.coalesce(
            pl.col("log_message"),
            pl.col("tcmd_response_text"),
            pl.col("bulk_data_hex").map_elements(
                _bulk_data_hex_to_general_message,
                return_dtype=pl.String,
            ),
        )
    )

    # Hard-code the column order here.
    force_start_col_names = ["packet_type", "general_message"]
    end_cols = ["log_message", "tcmd_response_text", "bulk_data_hex", "hex_payload"]
    tcmd_col_names = [
        col for col in df.columns if col.startswith("tcmd_") and (col not in end_cols)
    ]
    bulk_col_names = [
        col for col in df.columns if col.startswith("bulk_") and (col not in end_cols)
    ]
    force_positioned_columns = OrderedSet(
        force_start_col_names + tcmd_col_names + bulk_col_names + end_cols
    )

    df = df.select(
        *(OrderedSet(input_columns_at_start) - force_positioned_columns),
        *force_start_col_names,
        *tcmd_col_names,
        *bulk_col_names,
        *(
            OrderedSet(df.columns)
            - force_positioned_columns
            - set(input_columns_at_start)
        ),
        *end_cols,
    )

    assert set(df.columns) >= set(input_columns_at_start), "Columns were removed - bug?"

    if sort_setting == "by_timestamp":
        df = df.sort("received_timestamp", descending=True)
        logger.debug("Sorted by received_timestamp")
    elif sort_setting == "no_sort":
        logger.debug("Not sorted")
    else:
        assert_never(sort_setting)

    return df


def decode_to_csv(
    df: pl.DataFrame,
    output_csv: Path,
    *,
    sort_setting: Literal["no_sort", "by_timestamp"],
) -> None:
    """Decode CTS-SAT-1 packets already loaded into a dataframe."""
    df = decode_to_df(df, sort_setting=sort_setting)

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

    if input_csv is not None:
        input_path = input_csv
        df = load_packets_from_csv(input_csv)
    else:
        assert input_sqlite is not None
        input_path = input_sqlite
        df = load_packets_from_sqlite(input_sqlite)

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
    )


def main() -> None:
    """Entry point."""
    tyro.cli(run)


if __name__ == "__main__":
    main()
