"""Step 3: decode every distinct packet's payload.

Runs step 2's output (`distinct_packets_over_time.parquet` -- one row per
distinct received packet, `data_hex` already CRC-complete) through the
CTS-SAT-1 packet decoder in `cts1_mo_tools.cts1_decode_satnogs_packets`.
That decoder is a plain Python module (struct-unpacking `COMMS_*_packet_t`
layouts), not a CLI tool like the audio decoders step 1 shells out to, so
it's called directly in-process here -- no subprocess involved, just an
ordinary Python function call per distinct packet.

The result is an "everything decoded" table: every column step 2 already
produced (trust/traceability columns, `received_at`, ...) plus every field
the decoder could pull out of the packet -- beacon telemetry, log messages,
telecommand responses, bulk file chunks, whichever fields apply to that
packet's type. A packet whose bytes don't decode at all (too short, unknown
type) still gets a row, just with the decoded columns left null rather than
being dropped.

Like step 2, this is parquet-in-parquet-out: no database involved, and the
whole dataset is cheap enough to reprocess from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_FILENAME",
    "compute_decoded_packets",
    "run",
]

from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import decode_packet_safe
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    pipeline as step_2_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_OUTPUT_DIR = step_2_pipeline.DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME = "everything_decoded.parquet"


def _decode_unique_payloads(data_hex_values: list[str]) -> pl.DataFrame:
    """Run `decode_packet_safe` once per unique `data_hex`, in-process.

    A hex value that doesn't decode at all (too short, malformed) is simply
    absent from the result rather than included with all-null fields -- the
    left join in `compute_decoded_packets` produces the same effect (null
    decoded columns) without this function needing to know that table's
    schema.
    """
    decoded: dict[str, dict[str, Any]] = {}
    for data_hex in data_hex_values:
        fields = decode_packet_safe(data_hex)
        if fields is None:
            continue
        # Already have a verified `csp_crc_valid` from step 2, computed over
        # these same CRC-completed bytes; drop the decoder's redundant copy
        # so it doesn't collide on the join below.
        fields.pop("csp_crc_valid", None)
        decoded[data_hex] = fields

    if not decoded:
        return pl.DataFrame(schema={"data_hex": pl.String})

    return pl.DataFrame(
        [{"data_hex": data_hex, **fields} for data_hex, fields in decoded.items()],
        infer_schema_length=None,
    )


def compute_decoded_packets(distinct_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: distinct packets -> distinct packets + decoded fields.

    Args:
        distinct_packets: `distinct_packets_over_time.parquet`, as written by
            step 2.

    Returns:
        The same rows, left-joined with every field `decode_packet_safe`
        could pull out of each row's `data_hex` (null where decoding failed
        or the packet type isn't one the decoder knows the layout of).
    """
    decoded = _decode_unique_payloads(distinct_packets["data_hex"].unique().to_list())
    return distinct_packets.join(decoded, on="data_hex", how="left")


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Recompute `everything_decoded.parquet` from step 2's output.

    Reads `distinct_packets_over_time.parquet` from `output_dir` (where step
    2 writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    distinct_packets_path = output_dir / step_2_pipeline.OUTPUT_FILENAME
    if not distinct_packets_path.exists():
        msg = f"{distinct_packets_path} not found -- run step_2 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {distinct_packets_path}")
    distinct_packets = pl.read_parquet(distinct_packets_path)

    result = compute_decoded_packets(distinct_packets)

    out_path = output_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result, out_path)

    decoded_count = (
        result.height - result["packet_type"].null_count()
        if "packet_type" in result.columns
        else 0
    )
    logger.info(
        f"Done. {decoded_count:,}/{result.height:,} packet(s) decoded, "
        f"written to {out_path}."
    )
