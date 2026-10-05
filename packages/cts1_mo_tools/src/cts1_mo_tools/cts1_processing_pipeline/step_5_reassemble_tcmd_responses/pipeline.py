"""Step 5: reassemble multi-packet telecommand responses.

A telecommand response longer than one downlink frame's worth of data
(`TCMD_RESPONSE_MAX_DATA` bytes) is split by the firmware across several
`TCMD_RESPONSE` packets, numbered `tcmd_response_seq_num`
1..`tcmd_response_max_seq_num` (1-based -- real firmware never sends a 0),
all sharing the same `tcmd_ts_sent` (the timestamp of the telecommand that
produced the response) and the same `tcmd_response_code`/`tcmd_duration_ms`.

This step groups every `TCMD_RESPONSE` packet by those shared header
fields, orders each group's packets by sequence number, and joins their
`tcmd_response_text` back together into one row per response. A
single-packet response comes through as-is, as a group of one.

Any sequence number that was never received is filled with
`TCMD_RESPONSE_MAX_DATA` `?` characters in its place -- a full frame's
worth, which is exactly right for every part but the last (only the last
part can be short), and a reasonable stand-in for the last part, whose
real length is unknowable. `missing_seq_nums`/`is_complete` record which
parts, if any, were filled in this way.

The group key includes `tcmd_response_code`/`tcmd_duration_ms`/
`tcmd_response_max_seq_num` on top of `tcmd_ts_sent` -- every part of one
response carries identical values for all of them, so this never splits a
genuine response, but it does keep two different responses apart should
they ever share a `tcmd_ts_sent` (e.g. both sent before the satellite's
clock was time-synced, when `tcmd_ts_sent` is 0).

If the same part was received more than once (e.g. across two separate
passes, which step 2 doesn't merge), the earliest-received copy is used.
Like step 4, only packets with a confirmed-valid CSP CRC (`csp_crc_valid`)
are considered, so a bit-flipped header can't mis-file a part into the
wrong response.

This step only depends on step 3's output, so it's independent of step 4
(the two could run in either order, or concurrently). Like steps 2-4, it's
parquet-in-parquet-out: no database involved, and cheap enough to
reprocess from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "MISSING_PART_FILL",
    "OUTPUT_FILENAME",
    "compute_reassembled_tcmd_responses",
    "run",
]

from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    MAX_VALID_EPOCH_MS,
    TCMD_RESPONSE_MAX_DATA,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "reassembled_tcmd_responses.parquet"

# What a never-received part is replaced with -- see the module docstring.
MISSING_PART_FILL = "?" * TCMD_RESPONSE_MAX_DATA

# Every part of one response carries identical values for all of these --
# see the module docstring for why it's more than just `tcmd_ts_sent`.
_GROUP_KEY = (
    "tcmd_ts_sent",
    "tcmd_response_code",
    "tcmd_duration_ms",
    "tcmd_response_max_seq_num",
)

# The output schema, in column order -- also what an empty result (no
# telecommand responses decoded yet) is built with, so downstream readers
# always see the same columns regardless of how much history exists. The
# list-valued columns are JSON-array strings, same as step 2's
# `observation_ids`/`decoders`, so every export format (CSV, Excel, SQLite)
# can hold them.
_OUTPUT_SCHEMA = {
    "first_received_at": pl.Datetime("us", "UTC"),
    "last_received_at": pl.Datetime("us", "UTC"),
    "tcmd_sent_at": pl.Datetime("us", "UTC"),
    "tcmd_ts_sent": pl.Int64,
    "tcmd_response_code": pl.Int64,
    "tcmd_duration_ms": pl.Int64,
    "part_count": pl.Int64,
    "received_part_count": pl.Int64,
    "missing_seq_nums": pl.String,  # JSON array, e.g. "[2,3]"
    "is_complete": pl.Boolean,
    "packet_ids": pl.String,  # JSON array, e.g. '["ab12","cd34"]'
    "tcmd_response_text": pl.String,
}


def _json_array(elements: pl.Expr) -> pl.Expr:
    """Aggregate already-JSON-encoded `elements` into one JSON-array string."""
    return pl.format("[{}]", elements.str.join(","))


def _reassemble_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The filter/dedupe/fill/join query plan itself, kept lazy end-to-end
    so it runs as one optimized pass straight off disk.
    """
    seq = pl.col("tcmd_response_seq_num")
    max_seq = pl.col("tcmd_response_max_seq_num")

    parts = (
        lf.filter(
            (pl.col("packet_type") == "TCMD_RESPONSE")
            & pl.col("csp_crc_valid")
            & (seq >= 1)
            & (seq <= max_seq)
        )
        .select(
            *_GROUP_KEY,
            "tcmd_response_seq_num",
            "received_at",
            "packet_id",
            "tcmd_response_text",
        )
        # Earliest-received copy of each duplicated part wins.
        .sort("received_at", nulls_last=True)
        .unique(subset=[*_GROUP_KEY, "tcmd_response_seq_num"], keep="first")
    )

    # One row per sequence number every response *should* have, received or
    # not -- left-joining the received parts onto this is what exposes the
    # gaps that need filling.
    expected = (
        parts.select(_GROUP_KEY)
        .unique()
        .with_columns(tcmd_response_seq_num=pl.int_ranges(1, max_seq + 1))
        .explode("tcmd_response_seq_num", empty_as_null=True)
    )

    is_received = pl.col("packet_id").is_not_null()
    ts_sent = pl.col("tcmd_ts_sent")
    return (
        expected.join(parts, on=[*_GROUP_KEY, "tcmd_response_seq_num"], how="left")
        .sort("tcmd_response_seq_num")
        .group_by(_GROUP_KEY)
        .agg(
            first_received_at=pl.col("received_at").min(),
            last_received_at=pl.col("received_at").max(),
            received_part_count=is_received.sum().cast(pl.Int64),
            missing_seq_nums=_json_array(seq.filter(~is_received).cast(pl.String)),
            packet_ids=_json_array(pl.format('"{}"', pl.col("packet_id").drop_nulls())),
            tcmd_response_text=(
                pl.when(is_received)
                .then(pl.col("tcmd_response_text").fill_null(""))
                .otherwise(pl.lit(MISSING_PART_FILL))
                .str.join("")
            ),
        )
        .with_columns(
            part_count=max_seq,
            is_complete=pl.col("received_part_count") == max_seq,
            tcmd_sent_at=pl.when((ts_sent > 0) & (ts_sent <= MAX_VALID_EPOCH_MS)).then(
                pl.from_epoch(ts_sent, time_unit="ms").dt.replace_time_zone("UTC")
            ),
        )
        .select(pl.col(name).cast(dtype) for name, dtype in _OUTPUT_SCHEMA.items())
        .sort("first_received_at", descending=True, nulls_last=True)
    )


def compute_reassembled_tcmd_responses(decoded_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: `everything_decoded` -> one row per telecommand
    response, with multi-packet responses joined back together.

    Args:
        decoded_packets: `everything_decoded.parquet`, as written by step 3.

    Returns:
        One row per telecommand response, newest first -- see the module
        docstring for the reassembly rule and `_OUTPUT_SCHEMA` for the
        columns.
    """
    if decoded_packets.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    return _reassemble_lazy(decoded_packets.lazy()).collect()


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `reassembled_tcmd_responses.parquet` from step 3's output.

    Reads `everything_decoded.parquet` from `data_dir` (where step 3
    writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    if not decoded_path.exists():
        msg = f"{decoded_path} not found -- run step_3 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {decoded_path}")
    result_df = _reassemble_lazy(pl.scan_parquet(decoded_path)).collect()

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)

    multi_part = result_df.filter(pl.col("part_count") > 1)
    incomplete = result_df.filter(~pl.col("is_complete"))
    logger.info(
        f"Done. {result_df.height:,} telecommand response(s) written to "
        f"{out_path}: {multi_part.height:,} multi-packet, "
        f"{incomplete.height:,} with missing part(s)."
    )
