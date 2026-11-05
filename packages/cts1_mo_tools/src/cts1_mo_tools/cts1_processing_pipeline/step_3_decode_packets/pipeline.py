"""Step 3: decode every distinct packet's payload.

Runs step 2's output (`distinct_packets_over_time.parquet` -- one row per
distinct received packet, `data_hex` already CRC-complete) through
`decode_to_df()`, the same CTS-SAT-1 packet decoder the standalone
`cts1_decode_satnogs_packets` CLI uses. That decoder is a plain Python
module (struct-unpacking `COMMS_*_packet_t` layouts), not a CLI tool like
the audio decoders step 1 shells out to, so it's called directly in-process
here -- no subprocess involved.

The result is an "everything decoded" table: every column step 2 already
produced (trust/traceability columns, `received_at`, ...) plus every field
the decoder could pull out of the packet -- beacon telemetry, log messages,
telecommand responses, bulk file chunks, whichever fields apply to that
packet's type. A packet whose bytes don't decode at all (too short, unknown
type) still gets a row, just with the decoded columns left null rather than
being dropped.

Like step 2, this is parquet-in-parquet-out (no database involved), month
partitioned (see `partitions`), and runs either in full or incrementally:
a packet's decoded fields depend on nothing but its own row, so an
incremental run (`run(incremental=True)`) only re-decodes the months of
step 2's output that changed since the last run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "OUTPUT_FILENAME",
    "compute_decoded_packets",
    "run",
]

from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import decode_to_df
from cts1_mo_tools.cts1_processing_pipeline import partitions
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    pipeline as step_2_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets.schema import (
    DECODED_SCHEMA,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_2_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "everything_decoded.parquet"

# polars' default would make this a handful of ~200MB-uncompressed row
# groups (step 2's `sources` JSON is most of that), and a row group is the
# smallest unit a reader can skip or stream. The web UI reads this file
# constantly -- the newest few rows, a time window, a filtered null-count
# pass -- so small row groups keep each of those to the rows they touch,
# for a few percent more file size.
ROW_GROUP_SIZE = 10_000


def compute_decoded_packets(distinct_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: distinct packets -> distinct packets + decoded fields.

    Args:
        distinct_packets: `distinct_packets_over_time.parquet`, as written by
            step 2.

    Returns:
        The same rows, plus every field `decode_to_df` could pull out of
        each row's `data_hex` (null where decoding failed or the packet
        type isn't one the decoder knows the layout of) -- always all of
        `DECODED_SCHEMA`'s columns, in its order and dtypes, whatever
        packet types `distinct_packets` holds.
    """
    if distinct_packets.is_empty():
        return distinct_packets

    # `decode_to_df` expects a few different column names. Rename to its dialect,
    # then rename back at the end.
    renamed = distinct_packets.rename(
        {
            "data_hex": "hex_payload",
            "received_at": "received_timestamp",
            "csp_crc_valid": "csp_crc_valid_from_caller",
        }
    )

    decoded = decode_to_df(renamed, sort_setting="no_sort")

    decoded = decoded.drop(
        # Drop the one from within the decoder pipeline.
        "csp_crc_valid",
    ).rename(
        {
            "hex_payload": "data_hex",
            "received_timestamp": "received_at",
            "csp_crc_valid_from_caller": "csp_crc_valid",
        }
    )

    inputs = [col for col in distinct_packets.columns if col not in DECODED_SCHEMA]
    unlisted = [
        col
        for col in decoded.columns
        if col not in inputs and col not in DECODED_SCHEMA
    ]
    if unlisted:
        logger.warning(
            f"Decoded column(s) {unlisted} aren't in step 3's DECODED_SCHEMA -- "
            "kept, but add them there."
        )
    *listed, data_hex = DECODED_SCHEMA.items()
    return decoded.select(
        *inputs,
        *(
            pl.col(name).cast(dtype)
            if name in decoded.columns
            else pl.lit(None, dtype).alias(name)
            for name, dtype in listed
        ),
        *unlisted,
        pl.col(data_hex[0]).cast(data_hex[1]),
    )


def _run_incremental(in_path: Path, out_path: Path, changed_months: set[str]) -> bool:
    """Re-decode just `changed_months`, returning False (having written
    nothing) if they don't come out with the rest of the output's schema --
    only possible when `decode_to_df` has grown a column `DECODED_SCHEMA`
    doesn't list yet. Every month's file must share one schema, so then the
    whole output is rebuilt."""
    out_files = partitions.month_files(out_path)
    if not out_files:
        return False
    schema = pl.Schema(pl.read_parquet_schema(next(iter(out_files.values()))))
    in_files = partitions.month_files(in_path)
    months: list[pl.DataFrame] = []
    for month in sorted(changed_months & set(in_files)):
        decoded = compute_decoded_packets(pl.read_parquet(in_files[month]))
        if decoded.schema != schema:
            logger.info(f"Month {month} doesn't fit the existing schema")
            return False
        months.append(decoded)
    partitions.write_frame(
        pl.concat(months) if months else pl.DataFrame(schema=schema),
        out_path,
        months=changed_months,
        row_group_size=ROW_GROUP_SIZE,
    )
    rows = sum(month.height for month in months)
    logger.info(
        f"Done. Re-decoded {rows:,} packet(s) of changed month(s) "
        f"{sorted(changed_months)}, written to {out_path}."
    )
    return True


def run(*, data_dir: Path = DEFAULT_DATA_DIR, incremental: bool = False) -> None:
    """Recompute `everything_decoded.parquet` from step 2's output.

    Reads `distinct_packets_over_time.parquet` from `data_dir` (where step
    2 writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.

    With `incremental`, only the months step 2 changed since the last run
    are re-decoded -- falling back to a full run when that's not possible
    (see `partitions.changed_input_months` and `_run_incremental`).
    """
    distinct_packets_path = data_dir / step_2_pipeline.OUTPUT_FILENAME
    if not distinct_packets_path.exists():
        msg = f"{distinct_packets_path} not found -- run step_2 first."
        raise FileNotFoundError(msg)

    out_path = data_dir / OUTPUT_FILENAME
    input_signatures = partitions.signatures(distinct_packets_path)
    changed_months = (
        partitions.changed_input_months(out_path, input_signatures)
        if incremental
        else None
    )
    if changed_months is not None and not changed_months:
        logger.info("Done. No distinct packets changed since the last run.")
    elif changed_months is None or not _run_incremental(
        distinct_packets_path, out_path, changed_months
    ):
        _run_full(distinct_packets_path, out_path)
    partitions.write_state(out_path, input_signatures)


def _run_full(distinct_packets_path: Path, out_path: Path) -> None:
    logger.info(f"Reading {distinct_packets_path}")
    distinct_packets = pl.read_parquet(distinct_packets_path)

    result = compute_decoded_packets(distinct_packets)
    partitions.write_frame(result, out_path, row_group_size=ROW_GROUP_SIZE)

    decoded_count = (
        result.height - result["packet_type"].null_count()
        if "packet_type" in result.columns
        else 0
    )
    logger.info(
        f"Done. {decoded_count:,}/{result.height:,} packet(s) decoded, "
        f"written to {out_path}."
    )
