"""Bulk uplink a file by writing telecommands to an output file."""

import base64
import hashlib
import sys
from pathlib import Path

import tyro
from loguru import logger


def send_file_to_tcmd_file(
    input_file: Path,
    *,
    satellite_file: str,
    telecommand_output_file: Path,
    chunk_size: int = 168,
) -> None:
    """Send a file by writing CTS1 telecommands to an output file.

    Args:
        input_file: Path to the input file to send.
        satellite_file: Destination filename/path on the satellite filesystem.
        telecommand_output_file: Path to write the telecommand sequence to.
        chunk_size: Chunk size in bytes before base64 encoding. Best if
            divisible by 3 (and by a power of 2). Defaults to 168.
    """
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)

    lines: list[str] = []

    def emit(command: str) -> None:
        lines.append(command)
        logger.debug(f"Emitting: {command}")

    emit("CTS1+comms_bulk_uplink_close_file()")  # Safety measure.
    emit("CTS1+config_set_int_var(TCMD_require_unique_tssent,1)")
    emit(f"CTS1+comms_bulk_uplink_open_file({satellite_file},truncate)")

    file_bytes = input_file.read_bytes()
    total_size = len(file_bytes)
    chunk_index = 0

    logger.info(f"Encoding {input_file} ({total_size:,} bytes) into telecommands...")

    offset = 0
    while offset < total_size:
        chunk = file_bytes[offset : offset + chunk_size]
        b64_data = base64.b64encode(chunk).decode("ascii")
        emit(f"CTS1+bulkup64({b64_data})")
        offset += len(chunk)
        chunk_index += 1

    emit("CTS1+comms_bulk_uplink_close_file()")
    emit(f"CTS1+fs_read_file_sha256_hash_json({satellite_file},0,0)")

    hash_on_disk = hashlib.sha256(file_bytes).hexdigest()

    # Add a comment with the hash of the input file.
    lines.append(f"# SHA256 of input file: {hash_on_disk} ({total_size:,} bytes)")

    telecommand_output_file.write_text("\n".join(lines) + "\n")

    logger.success(
        f"Wrote {len(lines)} telecommands ({chunk_index} data chunks) to "
        f"{telecommand_output_file}"
    )
    logger.info(
        f"SHA256 of input file (computer-side): {hash_on_disk} ({total_size:,} bytes)"
    )


if __name__ == "__main__":
    tyro.cli(send_file_to_tcmd_file)
