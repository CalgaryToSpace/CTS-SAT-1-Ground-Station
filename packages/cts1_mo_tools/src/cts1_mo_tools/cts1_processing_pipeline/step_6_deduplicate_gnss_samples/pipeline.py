"""Step 6: de-duplicate GNSS samples by their content.

The satellite keeps its GNSS (BESTXYZB) samples in a ring buffer, and can
downlink the same sample many times over -- every `GNSS_BESTXYZB_SAMPLE`
packet carries a `downlink_seq_num` counter that ticks up on every downlink,
so two downlinks of the very same sample never have byte-identical packets
(and step 2, which de-duplicates by whole-packet content, keeps them apart).

This step groups every `GNSS_BESTXYZB_SAMPLE` packet by its content with the
downlink counter cut out -- i.e. the `ring_position` field plus the raw
BESTXYZB log (`gnss_sample_hex`) -- so each distinct sample gets one row,
along with how many times the ground segment received it:

  - `receive_count`: how many of step 3's rows (distinct received packets,
    i.e. separate downlinks) carried this sample.
  - `decode_count`: the sum of those rows' `packet_count` -- every decode
    of every ground station/decoder that contributed to them.

Each row's decoded `gnss_*` fields come from the earliest-received copy
(they're identical across copies anyway, being decoded from identical
bytes). Like steps 4/5, only packets with a confirmed-valid CSP CRC
(`csp_crc_valid`) are considered, so a bit-flipped copy can't show up as a
bogus "distinct" sample.

This step only depends on step 3's output (independent of steps 4 and 5).
Like steps 2-5, it's parquet-in-parquet-out: no database involved, and
cheap enough to reprocess from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "OUTPUT_FILENAME",
    "compute_distinct_gnss_samples",
    "run",
]

from typing import TYPE_CHECKING

import polars as pl
import polars_hash
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    CSP_CRC32C_SIZE,
    CSP_HEADER_SIZE,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "distinct_gnss_samples.parquet"

# Byte offset (into the whole CSP packet) where the sample content starts:
# past the CSP header, then the `packet_type` (1 byte) and `downlink_seq_num`
# (2 bytes) fields -- see `GNSS_DOWNLINK_HEADER_FMT`. The trailing CSP CRC is
# cut off too, since it covers the downlink counter.
_SAMPLE_START = CSP_HEADER_SIZE + 1 + 2

# The two `gnss_*` columns on beacon packets (not GNSS samples), which are
# null on every GNSS sample row -- not carried through.
_BEACON_GNSS_COLUMNS = frozenset({"gnss_uart_interrupt_enabled", "gnss_rx_mode"})

# Varies per copy of the sample, so it's aggregated rather than carried through.
_DOWNLINK_SEQ_NUM_COLUMN = "gnss_downlink_seq_num"

# The bookkeeping columns, in column order, ahead of the decoded `gnss_*`
# columns -- also what an empty result (no GNSS samples decoded yet) is built
# with. The list-valued columns are JSON-array strings, same as step 2's
# `observation_ids`/`decoders`, so every export format can hold them.
_OUTPUT_SCHEMA = {
    "gnss_sample_id": pl.String,
    "first_received_at": pl.Datetime("us", "UTC"),
    "last_received_at": pl.Datetime("us", "UTC"),
    "receive_count": pl.Int64,
    "decode_count": pl.Int64,
    "downlink_seq_nums": pl.String,  # JSON array, e.g. "[12,40]"
    "packet_ids": pl.String,  # JSON array, e.g. '["ab12","cd34"]'
    "gnss_sample_hex": pl.String,
}


def _json_array(elements: pl.Expr) -> pl.Expr:
    """Aggregate already-JSON-encoded `elements` into one JSON-array string."""
    return pl.format("[{}]", elements.str.join(","))


def _dedupe_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The filter/group-by query plan itself, kept lazy end-to-end so it runs
    as one optimized pass straight off disk.
    """
    names = lf.collect_schema().names()
    gnss_columns = [
        name
        for name in names
        if name.startswith("gnss_")
        and name not in _BEACON_GNSS_COLUMNS
        and name != _DOWNLINK_SEQ_NUM_COLUMN
    ]
    seq_num = (
        pl.col(_DOWNLINK_SEQ_NUM_COLUMN)
        if _DOWNLINK_SEQ_NUM_COLUMN in names
        else pl.lit(None)
    )

    raw = pl.col("data_hex").str.decode("hex", strict=False)
    sample = pl.col("_sample")
    return (
        lf.filter(
            (pl.col("packet_type") == "GNSS_BESTXYZB_SAMPLE") & pl.col("csp_crc_valid")
        )
        .with_columns(
            _sample=raw.bin.slice(_SAMPLE_START).bin.head(-CSP_CRC32C_SIZE),
            _seq_num=seq_num.cast(pl.Int64),
        )
        .filter(sample.is_not_null() & (sample.bin.size() > 0))
        # Earliest-received copy first, so `.first()` below picks it.
        .sort("received_at", nulls_last=True)
        .group_by("_sample")
        .agg(
            *(pl.col(name).first() for name in gnss_columns),
            first_received_at=pl.col("received_at").min(),
            last_received_at=pl.col("received_at").max(),
            receive_count=pl.len().cast(pl.Int64),
            decode_count=pl.col("packet_count").sum().cast(pl.Int64),
            downlink_seq_nums=_json_array(
                pl.col("_seq_num").drop_nulls().unique().sort().cast(pl.String)
            ),
            packet_ids=_json_array(pl.format('"{}"', pl.col("packet_id"))),
        )
        .with_columns(gnss_sample_hex=sample.bin.encode("hex"))
        .with_columns(
            gnss_sample_id=(
                polars_hash.col("gnss_sample_hex").chash.sha2_256().str.slice(0, 16)
            )
        )
        .select(
            *(pl.col(name).cast(dtype) for name, dtype in _OUTPUT_SCHEMA.items()),
            *gnss_columns,
        )
        .sort("first_received_at", descending=True, nulls_last=True)
    )


def compute_distinct_gnss_samples(decoded_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: `everything_decoded` -> one row per distinct GNSS
    sample, with how many times it was received.

    Args:
        decoded_packets: `everything_decoded.parquet`, as written by step 3.

    Returns:
        One row per distinct GNSS sample, newest first -- see the module
        docstring for the de-duplication rule, and `_OUTPUT_SCHEMA` for the
        leading columns (followed by every decoded `gnss_*` field).
    """
    return _compute(decoded_packets.lazy())


def _compute(lf: pl.LazyFrame) -> pl.DataFrame:
    if "packet_type" not in lf.collect_schema().names():
        # Step 3 found nothing to decode at all.
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    return _dedupe_lazy(lf).collect()


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `distinct_gnss_samples.parquet` from step 3's output.

    Reads `everything_decoded.parquet` from `data_dir` (where step 3
    writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    if not decoded_path.exists():
        msg = f"{decoded_path} not found -- run step_3 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {decoded_path}")
    result_df = _compute(pl.scan_parquet(decoded_path))

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)

    total_received = result_df["receive_count"].sum()
    logger.info(
        f"Done. {result_df.height:,} distinct GNSS sample(s) "
        f"(from {total_received:,} received GNSS packet(s)) written to {out_path}."
    )
