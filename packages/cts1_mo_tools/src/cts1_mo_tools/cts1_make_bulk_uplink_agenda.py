"""Bulk uplink a file by writing telecommands to an output file."""

import base64
import hashlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, assert_never

import tyro
from loguru import logger

MAX_TELECOMMAND_LENGTH = 210


def _parse_datetime_argument(dt_arg: int | str) -> datetime:
    if isinstance(dt_arg, int):
        return datetime.fromtimestamp(dt_arg, tz=UTC)

    # Else, it's a string.
    val = datetime.fromisoformat(dt_arg)
    if val.tzinfo is None:
        msg = f"Please specify a timezone offset in the timestamp string: {dt_arg}"
        raise ValueError(msg)

    return val


def _determine_max_optimal_chunk_size(  # noqa: PLR0913
    *,
    encoding: Literal["base64", "hex"],
    tcmd_name_len: int,
    len_of_other_args: int,
    overall_max_command_length: int = MAX_TELECOMMAND_LENGTH,
    # Defaults lengths:
    cts1_prefix_len: int = 4,
    tssent_tsexec_len: int = len("@tssent=1783064603123@tsexec=1783152000000"),
    parens_and_exclamation_len: int = 3,
) -> int:

    len_for_encoded_data = (
        overall_max_command_length
        - cts1_prefix_len
        - tcmd_name_len
        - len_of_other_args
        - tssent_tsexec_len
        - parens_and_exclamation_len
    )

    if encoding == "base64":
        # 4 characters per 3 bytes!
        raw_bytes_capacity = int(len_for_encoded_data / (4 / 3))

        # Best if divisible by 3 (so there's no base64 padding) *and* by a power of 2.
        # Round down to nearest multiple of 12.
        return int((raw_bytes_capacity // 12) * 12)

    elif encoding == "hex":  # noqa: RET505
        # 2 characters per byte.
        return len_for_encoded_data // 2
    else:
        assert_never(encoding)


def send_file_to_tcmd_file(  # noqa: C901, PLR0913, PLR0915
    input_file: Path,
    *,
    satellite_file: str,
    telecommand_output_file: Path,
    chunk_size: int | None = None,
    tssent_start_val: int | str | None = None,
    tssent_interval_ms: int = 1000,
    tsexec_start_val: int | str | None = None,
    tsexec_interval_ms: int = 30_000,
    hash_count: int = 30,
    mode: Literal[
        "bulk_uplink_b64", "bulk_uplink_hex", "write_file_hex"
    ] = "bulk_uplink_b64",
) -> None:
    """Send a file by writing CTS1 telecommands to an output file.

    Args:
        input_file: Path to the input file to send.
        satellite_file: Destination filename/path on the satellite filesystem.
        telecommand_output_file: Path to write the telecommand sequence to.
        chunk_size: Chunk size in bytes before base64/hex encoding. Best if
            divisible by 3 (and by a power of 2) for base64.
            If None, it will be determined automatically.
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

        assert len(command_out) <= MAX_TELECOMMAND_LENGTH, (
            f"Telecommand too long ({len(command_out)} chars): {command_out}"
        )

        lines.append(command_out)
        logger.debug(f"Emitting: {command_out}")

        if current_tssent is not None:
            current_tssent += timedelta(milliseconds=tssent_interval_ms)

        if current_tsexec is not None and (immediate is False):
            current_tsexec += timedelta(milliseconds=tsexec_interval_ms)

    file_bytes = input_file.read_bytes()
    total_size = len(file_bytes)
    chunk_index = 0

    use_bulk_uplink: bool
    encoding: Literal["base64", "hex"]

    if mode == "bulk_uplink_hex":
        encoding = "hex"
        use_bulk_uplink = True
        tcmd_name_len = len("bulkup16")
        len_of_other_args = 0
        command_format_string = "CTS1+bulkup16({hex_data})"
    elif mode == "write_file_hex":
        encoding = "hex"
        use_bulk_uplink = False
        tcmd_name_len = len("fs_write_file_hex")
        len_of_other_args = len(f"{satellite_file},{total_size},")
        command_format_string = (
            "CTS1+fs_write_file_hex({satellite_file},{offset},{hex_data})"
        )
    elif mode == "bulk_uplink_b64":
        encoding = "base64"
        use_bulk_uplink = True
        tcmd_name_len = len("bulkup64")
        len_of_other_args = 0
        command_format_string = "CTS1+bulkup64({b64_data})"

    else:
        assert_never(mode)

    if chunk_size is None:
        real_chunk_size: int = _determine_max_optimal_chunk_size(
            encoding=encoding,
            tcmd_name_len=tcmd_name_len,
            len_of_other_args=len_of_other_args,
        )
    else:
        real_chunk_size = chunk_size

    if use_bulk_uplink:
        emit("CTS1+comms_bulk_uplink_close_file()")  # Safety measure.
    emit("CTS1+config_set_int_var(TCMD_require_unique_tssent,1)", immediate=True)
    if use_bulk_uplink:
        emit(f"CTS1+comms_bulk_uplink_open_file({satellite_file},truncate)")

    logger.info(f"Encoding {input_file} ({total_size:,} bytes) into telecommands...")

    offset = 0
    while offset < total_size:
        chunk = file_bytes[offset : offset + real_chunk_size]
        command = command_format_string.format(
            hex_data=chunk.hex(),
            b64_data=base64.b64encode(chunk).decode("ascii"),
            satellite_file=satellite_file,
            offset=offset,
        )
        emit(command)
        offset += len(chunk)
        chunk_index += 1

    if use_bulk_uplink:
        emit("CTS1+comms_bulk_uplink_close_file()")

    # Repeat this `hash_count` times for a better chance of data transfer.
    for _ in range(hash_count):
        emit(f"CTS1+fs_read_file_sha256_hash_json({satellite_file},0,0)")

    hash_on_disk = hashlib.sha256(file_bytes).hexdigest()

    # Add a comment with the hash of the input file.
    cli_command = " ".join(
        [
            arg
            if "cts1_make_bulk_uplink_agenda" not in str(arg)
            else "cts1_make_bulk_uplink_agenda"
            for arg in sys.argv
        ]
    )
    commands_count = len([line for line in lines if line.startswith("CTS1+")])
    footer_comments = [
        f"# SHA256 of input file: {hash_on_disk} ({total_size:,} bytes)",
        f"# Data chunk count: {chunk_index}",
        f"# Total commands generated: {commands_count}",
        "# Generated with arguments:",
        f"#   input_file.name={input_file.name}",
        f"#   satellite_file={satellite_file}",
        (
            f"#   chunk_size_bytes={real_chunk_size}"
            + (" (auto determined)" if chunk_size is None else "")
        ),
        f"#   tssent_start_val={tssent_start_val}",
        f"#   tssent_interval_ms={tssent_interval_ms}",
        f"#   tsexec_start_val={tsexec_start_val}",
        f"#   tsexec_interval_ms={tsexec_interval_ms}",
        f"#   mode={mode}",
        f"#   command: {cli_command}",
    ]
    logger.debug("Footer comments:\n" + "\n".join(footer_comments))

    lines.extend(footer_comments)
    telecommand_output_file.write_text("\n".join(lines) + "\n")

    logger.success(
        f"Wrote {len(lines)} lines "
        f"({commands_count} commands, {chunk_index} data chunks) "
        f"to {telecommand_output_file}"
    )
    logger.info(
        f"SHA256 of input file (computer-side): {hash_on_disk} ({total_size:,} bytes)"
    )


def main() -> None:
    tyro.cli(send_file_to_tcmd_file)


if __name__ == "__main__":
    main()
