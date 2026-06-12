"""
CTS-SAT-1 Packet Decoder (from SatNOGS data).

Decodes COMMS_*_packet_t structs from the SatNOGS-style CSV export.

CSV format (pipe-delimited):
  timestamp | hex_payload | observation_id | ground_station
"""

import datetime
import struct
from pathlib import Path
from typing import Any

import polars as pl
import tyro
from loguru import logger
from ordered_set import OrderedSet

# -- Constants ----------------------------------------------------------------

CSP_HEADER_SIZE = 4
FRIENDLY_MESSAGE_SIZE = 42  # COMMS_BEACON_FRIENDLY_MESSAGE_SIZE
END_MESSAGE_SIZE = 4  # "END\0"

AX100_DOWNLINK_MAX_BYTES_SIZE = 200

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


def e(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"UNKNOWN({value})")


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
    end_raw = payload[
        BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE : BEACON_FIXED_SIZE
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
        "end_sentinel_ok": end_ok,
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
    null_pos = data_bytes.find(b"\x00")
    if null_pos >= 0:
        message = data_bytes[:null_pos].decode("utf-8", errors="replace")
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


def decode_to_csv(input_csv: Path, output_csv: Path) -> None:
    """Decode CTS-SAT-1 packets from a SatNOGS-style pipe-delimited CSV."""
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

    # Add a general "as decoded message" column for logs, telecommand responses, and
    # bulk file transfers.
    df = df.with_columns(
        general_message=pl.coalesce(
            pl.col("log_message"),
            pl.col("tcmd_response_text"),
            pl.col("bulk_data_hex").map_elements(
                lambda hex_str: bytes.fromhex(hex_str).decode(
                    "utf-8", errors="replace"
                ),
                return_dtype=pl.String,
            ),
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


def run(input_csv: Path, output_csv: Path | None = None) -> None:
    """Decode CTS-SAT-1 packets from a SatNOGS-style pipe-delimited CSV.

    If ``output_csv`` is not given, the decoded packets will be written to a new
    file with the same stem as ``input_csv`` but with "-decoded" appended.
    """

    if output_csv is not None:
        output_csv_path = output_csv
    else:
        output_csv_path = input_csv.with_stem(input_csv.stem + "-decoded")

    decode_to_csv(input_csv, output_csv=output_csv_path)


def main() -> None:
    """Entry point."""
    tyro.cli(run)


if __name__ == "__main__":
    main()
