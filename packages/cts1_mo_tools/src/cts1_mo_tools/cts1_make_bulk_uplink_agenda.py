"""Bulk uplink a file by writing telecommands to an output file."""

import base64
import hashlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, assert_never

import tyro
from loguru import logger


def _parse_datetime_argument(dt_arg: int | str) -> datetime:
    if isinstance(dt_arg, int):
        return datetime.fromtimestamp(dt_arg, tz=UTC)

    # Else, it's a string.
    val = datetime.fromisoformat(dt_arg)
    if val.tzinfo is None:
        msg = f"Please specify a timezone offset in the timestamp string: {dt_arg}"
        raise ValueError(msg)

    return val


def send_file_to_tcmd_file(  # noqa: C901, PLR0913, PLR0915
    input_file: Path,
    *,
    satellite_file: str,
    telecommand_output_file: Path,
    chunk_size: int = 96,
    tssent_start_val: int | str | None = None,
    tssent_interval_ms: int = 1000,
    tsexec_start_val: int | str | None = None,
    tsexec_interval_ms: int = 30_000,
    mode: Literal[
        "bulk_uplink_b64", "bulk_uplink_hex", "write_file_hex"
    ] = "bulk_uplink_b64",
) -> None:
    """Send a file by writing CTS1 telecommands to an output file.

    Args:
        input_file: Path to the input file to send.
        satellite_file: Destination filename/path on the satellite filesystem.
        telecommand_output_file: Path to write the telecommand sequence to.
        chunk_size: Chunk size in bytes before base64 encoding. Best if
            divisible by 3 (and by a power of 2).
        tssent_start_val: Timestamp to use for the first tssent telecommand.
            If not provided, no tssent suffix tags will be added.
            E.g., "2027-01-01T00:00:00-06:00"
        tssent_interval_ms: Interval in milliseconds between tssent telecommands.
        tsexec_start_val: Timestamp to use for the first tsexec telecommand.
            If not provided, no tsexec suffix tags will be added.
            E.g., "2027-01-01T00:00:00-06:00"
        tsexec_interval_ms: Interval in milliseconds between tsexec telecommands.
        mode: "bulk_uplink_b64" (use `comms_bulk_uplink_*` commands, more efficient)
            or "bulk_uplink_hex" (use `comms_bulk_uplink_*` commands, less efficient)
            or "write_file_hex" (use `fs_write_file_hex` command).
            As of 2026-06-28, FrontierSat doesn't appear to work with "bulk_uplink_b64".
    """
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)

    lines: list[str] = []

    current_tssent: datetime | None = (
        _parse_datetime_argument(tssent_start_val)
        if tssent_start_val is not None
        else None
    )
    current_tsexec: datetime | None = (
        _parse_datetime_argument(tsexec_start_val) if tsexec_start_val else None
    )

    def emit(command: str, *, immediate: bool = False) -> None:
        nonlocal current_tssent, current_tsexec

        command_out = command.rstrip("!")

        if current_tssent is not None:
            tssent_int = int(current_tssent.timestamp() * 1000)
            command_out += f"@tssent={tssent_int}"

        if current_tsexec is not None and (immediate is False):
            tsexec_int = int(current_tsexec.timestamp() * 1000)
            command_out += f"@tsexec={tsexec_int}"

        command_out += "!"

        lines.append(command_out)
        logger.debug(f"Emitting: {command_out}")

        if current_tssent is not None:
            current_tssent += timedelta(milliseconds=tssent_interval_ms)

        if current_tsexec is not None and (immediate is False):
            current_tsexec += timedelta(milliseconds=tsexec_interval_ms)

    use_bulk_uplink: bool = mode in {"bulk_uplink_b64", "bulk_uplink_hex"}

    if use_bulk_uplink:
        emit("CTS1+comms_bulk_uplink_close_file()")  # Safety measure.
    emit("CTS1+config_set_int_var(TCMD_require_unique_tssent,1)", immediate=True)
    if use_bulk_uplink:
        emit(f"CTS1+comms_bulk_uplink_open_file({satellite_file},truncate)")

    file_bytes = input_file.read_bytes()
    total_size = len(file_bytes)
    chunk_index = 0

    logger.info(f"Encoding {input_file} ({total_size:,} bytes) into telecommands...")

    offset = 0
    while offset < total_size:
        chunk = file_bytes[offset : offset + chunk_size]
        if mode == "bulk_uplink_hex":
            hex_data = chunk.hex()
            emit(f"CTS1+bulkup16({hex_data})")
        elif mode == "write_file_hex":
            hex_data = chunk.hex()
            emit(f"CTS1+fs_write_file_hex({satellite_file},{offset},{hex_data})")
        elif mode == "bulk_uplink_b64":
            b64_data = base64.b64encode(chunk).decode("ascii")
            emit(f"CTS1+bulkup64({b64_data})")
        else:
            assert_never(mode)
        offset += len(chunk)
        chunk_index += 1

    if use_bulk_uplink:
        emit("CTS1+comms_bulk_uplink_close_file()")

    # Repeat this 5 times for a better chance of data transfer.
    for _ in range(5):
        emit(f"CTS1+fs_read_file_sha256_hash_json({satellite_file},0,0)")

    hash_on_disk = hashlib.sha256(file_bytes).hexdigest()

    # Add a comment with the hash of the input file.
    lines.append(f"# SHA256 of input file: {hash_on_disk} ({total_size:,} bytes)")
    lines.extend(
        [
            "# Generated with arguments:",
            f"#   input_file.name={input_file.name}",
            f"#   satellite_file={satellite_file}",
            f"#   chunk_size={chunk_size}",
            f"#   tssent_start_val={tssent_start_val}",
            f"#   tssent_interval_ms={tssent_interval_ms}",
            f"#   tsexec_start_val={tsexec_start_val}",
            f"#   tsexec_interval_ms={tsexec_interval_ms}",
            f"#   mode={mode}",
        ]
    )

    telecommand_output_file.write_text("\n".join(lines) + "\n")

    logger.success(
        f"Wrote {len(lines)} telecommands ({chunk_index} data chunks) to "
        f"{telecommand_output_file}"
    )
    logger.info(
        f"SHA256 of input file (computer-side): {hash_on_disk} ({total_size:,} bytes)"
    )


def main() -> None:
    tyro.cli(send_file_to_tcmd_file)


if __name__ == "__main__":
    main()
