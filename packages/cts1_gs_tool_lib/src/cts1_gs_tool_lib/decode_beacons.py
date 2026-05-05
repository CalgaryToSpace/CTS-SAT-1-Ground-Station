"""
CTS-SAT-1 Beacon Packet Decoder.

Decodes COMMS_beacon_basic_packet_t structs from the SatNOGS-style CSV export.

CSV format (pipe-delimited):
  timestamp | hex_payload | (empty) | ground_station

Packet layout (all little-endian, #pragma pack(1)):
  4  bytes  CSP header         (stripped, not decoded)
  130 bytes  COMMS_beacon_basic_packet_t
  0-4 bytes  optional trailing bytes appended by ground station (CRC/metadata)
"""

import datetime
import json
import struct
import sys
from pathlib import Path
from typing import Any

import polars as pl
import tyro
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────

CSP_HEADER_SIZE = 4
FRIENDLY_MESSAGE_SIZE = 42  # COMMS_BEACON_FRIENDLY_MESSAGE_SIZE
END_MESSAGE_SIZE = 4  # "END\0"

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
FIXED_SIZE = struct.calcsize(FIXED_FMT)
TOTAL_STRUCT_SIZE = FIXED_SIZE + FRIENDLY_MESSAGE_SIZE + END_MESSAGE_SIZE  # 130 bytes

FIELD_NAMES = [
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

# ── Enum maps ────────────────────────────────────────────────────────────────

PACKET_TYPE_MAP = {
    0x01: "BEACON_BASIC",
    0x02: "BEACON_PERIPHERAL",
    0x03: "LOG_MESSAGE",
    0x04: "TCMD_RESPONSE",
    0x10: "BULK_FILE_DOWNLINK",
}
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


def e(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"UNKNOWN({value})")


# ── Decoder ──────────────────────────────────────────────────────────────────


def decode_packet(
    hex_str: str, received_timestamp: str, ground_station: str
) -> dict[str, Any]:
    raw = bytes.fromhex(hex_str)
    if len(raw) < CSP_HEADER_SIZE + TOTAL_STRUCT_SIZE:
        msg = f"Too short: {len(raw)} bytes"
        raise ValueError(msg)
    csp = raw[:CSP_HEADER_SIZE]
    payload = raw[CSP_HEADER_SIZE:]

    vals = struct.unpack_from(FIXED_FMT, payload, 0)
    rf = dict(zip(FIELD_NAMES, vals, strict=True))
    fm_raw = payload[FIXED_SIZE : FIXED_SIZE + FRIENDLY_MESSAGE_SIZE]
    friendly = fm_raw.split(b"\x00")[0].decode("utf-8", errors="replace")
    end_raw = payload[
        FIXED_SIZE + FRIENDLY_MESSAGE_SIZE : FIXED_SIZE
        + FRIENDLY_MESSAGE_SIZE
        + END_MESSAGE_SIZE
    ]
    end_ok = end_raw == b"END\x00"
    sat_name = rf["satellite_name"].decode("ascii", errors="replace").rstrip("\x00")
    epoch_ms = rf["unix_epoch_time_ms"]
    utc_time = (
        datetime.datetime.fromtimestamp(epoch_ms / 1000.0, datetime.UTC).isoformat()
        + "Z"
        if epoch_ms
        else None
    )

    return {
        # Meta
        "received_timestamp": received_timestamp,
        "ground_station": ground_station,
        "csp_header_hex": csp.hex(),
        "end_sentinel_ok": end_ok,
        # Identity
        "packet_type": e(PACKET_TYPE_MAP, rf["packet_type"]),
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


# ── CSV parsing ──────────────────────────────────────────────────────────────


def parse_input_csv(path: Path) -> tuple[list[dict[str, Any]], int]:
    packets: list[dict[str, Any]] = []
    skipped_count = 0

    with path.open(newline="", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:  # noqa: PLR2004
                continue
            timestamp = parts[0].strip()
            hex_payload = parts[1].strip()
            ground_station = parts[3].strip() if len(parts) > 3 else ""  # noqa: PLR2004
            if not hex_payload:
                skipped_count += 1
                continue
            try:
                raw = bytes.fromhex(hex_payload)
            except ValueError:
                logger.warning(
                    f"  [line {lineno}] Invalid hex, skipping.", file=sys.stderr
                )
                skipped_count += 1
                continue
            if len(raw) < CSP_HEADER_SIZE + 1:
                skipped_count += 1
                continue
            ptype = raw[CSP_HEADER_SIZE]
            if ptype != 0x01:
                skipped_count += 1
                continue
            try:
                packets.append(decode_packet(hex_payload, timestamp, ground_station))
            except Exception as ex:  # noqa: BLE001
                logger.info(f"  [line {lineno}] Decode error: {ex}", file=sys.stderr)
                skipped_count += 1

    return packets, skipped_count


# ── Main ─────────────────────────────────────────────────────────────────────


def main(input_csv: Path, output_csv: Path, output_json: Path | None = None) -> None:
    """Decode CTS1 beacon packets from a SatNOGS-style pipe-delimited CSV.

    Also writes a sidecar JSON file alongside the output CSV
    (same path, .json extension).
    """
    logger.info(f"Reading: {input_csv}")
    packets: list[dict[str, Any]]
    packets, skipped = parse_input_csv(input_csv)
    logger.info(f"Decoded: {len(packets)} beacon packets  |  Skipped: {skipped}")

    df_packets = pl.DataFrame(packets)
    del packets

    df_packets = df_packets.sort("uptime_ms", "received_timestamp").unique(
        "uptime_ms", keep="first", maintain_order=True
    )

    if output_json:
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(df_packets.to_dict(), f, indent=2, default=str)
        logger.info(f"  JSON → {output_json}")

    if output_csv:
        df_packets.write_csv(output_csv)
        logger.info(f"  CSV  → {output_csv}")

    # Pretty-print the most recent packet.
    print("\n\n── Last decoded packet ──────────────────────────────────────────")  # noqa: T201
    for k, v in df_packets.tail(1).to_dicts()[0].items():
        print(f"  {k:<44} {v}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
