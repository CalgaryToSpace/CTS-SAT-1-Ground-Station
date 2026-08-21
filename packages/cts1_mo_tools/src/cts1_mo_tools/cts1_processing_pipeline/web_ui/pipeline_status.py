"""Loading + shaping pipeline-health data for the web UI's Pipeline Status
page: row counts and freshness timestamps at every stage (step 1's raw
SatNOGS observations/packets/decoder runs, step 2's deduplicated distinct
packets, step 3's fully decoded packets), plus a per-decoder-tool scoreboard.

Pure data layer: no NiceGUI/rendering concerns live here, just polars -- see
`data.py` for the analogous loader over step 3's final
`everything_decoded.parquet` alone.
"""

from __future__ import annotations

__all__ = [
    "DECODER_RUNS_FILENAME",
    "DEFAULT_OUTPUT_DIR",
    "DISTINCT_PACKETS_FILENAME",
    "RAW_OBSERVATIONS_FILENAME",
    "RAW_PACKETS_FILENAME",
    "DecoderToolStats",
    "PacketCompletenessSummary",
    "PipelineCounts",
    "decoder_tool_stats",
    "latest_packets",
    "packet_completeness_summary",
    "pipeline_counts",
]

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    db as step_1_db,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    pipeline as step_2_pipeline,
)

if TYPE_CHECKING:
    from datetime import timedelta
    from pathlib import Path

# Every one of these lives in the same output directory as step 1's DuckDB
# export -- see `step_1_download_and_demodulate.db.export_parquets` and
# `step_2_deduplicate_packets.pipeline` for where each is written.
DEFAULT_OUTPUT_DIR = step_1_pipeline.DEFAULT_DB_PATH.parent
RAW_OBSERVATIONS_FILENAME = f"{step_1_db.RAW_OBSERVATIONS_TABLE}.parquet"
RAW_PACKETS_FILENAME = f"{step_1_db.RAW_PACKETS_TABLE}.parquet"
DECODER_RUNS_FILENAME = f"{step_1_db.DECODER_RUNS_TABLE}.parquet"
DISTINCT_PACKETS_FILENAME = step_2_pipeline.OUTPUT_FILENAME


def _scan(path: Path) -> pl.LazyFrame | None:
    """Lazily open `path`, or None if that pipeline stage hasn't produced it
    yet -- every function below treats a missing file as "zero rows, no
    freshness timestamp" rather than raising, since a fresh checkout or a
    pipeline that hasn't reached that stage yet is a normal state to render.
    """
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def _row_count(lf: pl.LazyFrame | None) -> int:
    if lf is None:
        return 0
    return lf.select(pl.len()).collect().item()


def _max_datetime(lf: pl.LazyFrame | None, column: str) -> datetime | None:
    if lf is None:
        return None
    return lf.select(pl.col(column).max()).collect().item()


def _file_mtime(path: Path) -> datetime | None:
    """When `path` was last written, or None if it doesn't exist yet.

    Step 3 rewrites `everything_decoded.parquet` from scratch on every run
    (no per-row "decoded at" column -- see its pipeline module), so the
    file's own mtime is the only freshness signal for "when did step 3 last
    run" available without changing that file's schema.
    """
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


@dataclass(frozen=True, slots=True)
class PipelineCounts:
    """At-a-glance row counts + freshness timestamps for every pipeline
    stage, so a stalled or broken step shows up as a stale timestamp or a
    zero count instead of silently vanishing into "everything looks fine"
    charts.

    `total_distinct_packets` and `total_decoded_packets` should normally be
    equal (step 3 just adds decoded fields to every one of step 2's rows,
    dropping none) -- a gap between them means step 3 hasn't caught up to
    step 2's latest output yet.
    """

    total_observations: int
    total_raw_decodes: int
    total_distinct_packets: int
    total_decoded_packets: int
    latest_observation_end: datetime | None
    latest_packet_received_at: datetime | None
    latest_packet_ingested_at: datetime | None
    latest_decoder_run_at: datetime | None
    latest_decoded_at: datetime | None

    @property
    def decode_backlog(self) -> int:
        """How many of step 2's distinct packets step 3 hasn't decoded yet."""
        return max(0, self.total_distinct_packets - self.total_decoded_packets)


def pipeline_counts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    decoded_packets_path: Path,
) -> PipelineCounts:
    """Row counts + freshness timestamps for every pipeline stage.

    `output_dir` is where step 1/2's raw/intermediate parquet files live;
    `decoded_packets_path` is passed separately since the web UI already
    lets the user point `--parquet-path` at a non-default
    `everything_decoded.parquet`.
    """
    raw_observations_lf = _scan(output_dir / RAW_OBSERVATIONS_FILENAME)
    raw_packets_lf = _scan(output_dir / RAW_PACKETS_FILENAME)
    distinct_packets_lf = _scan(output_dir / DISTINCT_PACKETS_FILENAME)
    decoder_runs_lf = _scan(output_dir / DECODER_RUNS_FILENAME)
    decoded_packets_lf = _scan(decoded_packets_path)

    return PipelineCounts(
        total_observations=_row_count(raw_observations_lf),
        total_raw_decodes=_row_count(raw_packets_lf),
        total_distinct_packets=_row_count(distinct_packets_lf),
        total_decoded_packets=_row_count(decoded_packets_lf),
        latest_observation_end=_max_datetime(raw_observations_lf, "end"),
        latest_packet_received_at=_max_datetime(raw_packets_lf, "received_at"),
        latest_packet_ingested_at=_max_datetime(raw_packets_lf, "ingested_at"),
        latest_decoder_run_at=_max_datetime(decoder_runs_lf, "run_at"),
        latest_decoded_at=_file_mtime(decoded_packets_path),
    )


@dataclass(frozen=True, slots=True)
class DecoderToolStats:
    """Per-decoder-tool scoring: how many raw decode attempts it produced,
    how many of the *distinct* (post-dedup) packets it contributed to, and
    when it last actually ran.

    `distinct_packet_count`/`distinct_packet_share` is the meaningful
    "score" -- `raw_decode_count` alone conflates a genuinely better decoder
    with one that just double-counts the same packet across more SatNOGS
    ground stations. A decoder that's stopped running (stale `last_run_at`)
    or whose share has collapsed is the sign to go investigate, not the raw
    counts alone.
    """

    decoder: str
    raw_decode_count: int
    distinct_packet_count: int
    distinct_packet_share: float  # distinct_packet_count / total distinct packets, 0..1
    last_run_at: datetime | None
    last_version: str | None
    avg_runtime_ms: float | None


def decoder_tool_stats(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[DecoderToolStats]:
    """One `DecoderToolStats` per decoder tool that's shown up anywhere
    (raw decodes, distinct-packet contributions, or recorded runs), sorted
    by `distinct_packet_count` descending -- the tool contributing the most
    distinct packets first.
    """
    raw_lf = _scan(output_dir / RAW_PACKETS_FILENAME)
    distinct_lf = _scan(output_dir / DISTINCT_PACKETS_FILENAME)
    runs_lf = _scan(output_dir / DECODER_RUNS_FILENAME)

    raw_counts = (
        raw_lf.group_by("decoder").agg(pl.len().alias("raw_decode_count")).collect()
        if raw_lf is not None
        else pl.DataFrame(schema={"decoder": pl.String, "raw_decode_count": pl.UInt32})
    )

    total_distinct = _row_count(distinct_lf)
    if distinct_lf is not None:
        distinct_counts = (
            distinct_lf.select(pl.col("decoders").str.json_decode(pl.List(pl.String)))
            .explode("decoders")
            .rename({"decoders": "decoder"})
            .group_by("decoder")
            .agg(pl.len().alias("distinct_packet_count"))
            .collect()
        )
    else:
        distinct_counts = pl.DataFrame(
            schema={"decoder": pl.String, "distinct_packet_count": pl.UInt32}
        )

    run_stats = (
        runs_lf.group_by("decoder")
        .agg(
            pl.col("run_at").max().alias("last_run_at"),
            pl.col("runtime_ms").mean().alias("avg_runtime_ms"),
            pl.col("version").sort_by(pl.col("run_at")).last().alias("last_version"),
        )
        .collect()
        if runs_lf is not None
        else pl.DataFrame(
            schema={
                "decoder": pl.String,
                "last_run_at": pl.Datetime("us", "UTC"),
                "avg_runtime_ms": pl.Float64,
                "last_version": pl.String,
            }
        )
    )

    decoders = sorted(
        set(raw_counts["decoder"].to_list())
        | set(distinct_counts["decoder"].to_list())
        | set(run_stats["decoder"].to_list())
    )

    raw_map = dict(
        zip(raw_counts["decoder"], raw_counts["raw_decode_count"], strict=True)
    )
    distinct_map = dict(
        zip(
            distinct_counts["decoder"],
            distinct_counts["distinct_packet_count"],
            strict=True,
        )
    )
    run_map = {row["decoder"]: row for row in run_stats.to_dicts()}

    stats: list[DecoderToolStats] = []
    for decoder in decoders:
        distinct_count = distinct_map.get(decoder, 0)
        run_row = run_map.get(decoder)
        stats.append(
            DecoderToolStats(
                decoder=decoder,
                raw_decode_count=raw_map.get(decoder, 0),
                distinct_packet_count=distinct_count,
                distinct_packet_share=(
                    distinct_count / total_distinct if total_distinct else 0.0
                ),
                last_run_at=run_row["last_run_at"] if run_row is not None else None,
                last_version=run_row["last_version"] if run_row is not None else None,
                avg_runtime_ms=run_row["avg_runtime_ms"]
                if run_row is not None
                else None,
            )
        )

    stats.sort(key=lambda s: s.distinct_packet_count, reverse=True)
    return stats


def latest_packets(path: Path, n: int = 25) -> pl.DataFrame:
    """The `n` most recently received packets of any type, newest first --
    unlike `data.latest_beacons`, this isn't restricted to beacon packets,
    since the point here is "is *anything* still coming down."
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    return lf.sort("received_at", descending=True).head(n).collect()


@dataclass(frozen=True, slots=True)
class PacketCompletenessSummary:
    """Whole-history completeness signal, independent of whatever window a
    "most recent packets" table happens to show: the very first packet
    ever received, and the largest gap between any two consecutive
    packets across the *entire* history -- a dropout sitting a thousand
    rows back would never show up if this were computed over just the
    latest handful of rows.
    """

    first_received_at: datetime | None
    largest_gap: timedelta | None
    largest_gap_from: datetime | None  # the packet right before the gap
    largest_gap_to: datetime | None  # the packet right after the gap


def packet_completeness_summary(path: Path) -> PacketCompletenessSummary:
    """Compute `PacketCompletenessSummary` over every row at `path`.

    Stays lazy end-to-end -- sorts the full `received_at` column, pairs
    each row with the one before it, then reduces straight to a
    single-row aggregation, so this never materializes more than one
    row's worth of history into memory regardless of how large the
    pipeline's packet history has grown.
    """
    lf = _scan(path)
    if lf is None:
        return PacketCompletenessSummary(None, None, None, None)

    lf = lf.sort("received_at").select(
        pl.col("received_at"),
        pl.col("received_at").shift(1).alias("_prev_received_at"),
        pl.col("received_at").diff().alias("_gap"),
    )
    lf = lf.select(
        pl.col("received_at").first().alias("first_received_at"),
        pl.col("_gap").max().alias("largest_gap"),
        pl.col("_prev_received_at")
        .sort_by(pl.col("_gap"), descending=True, nulls_last=True)
        .first()
        .alias("largest_gap_from"),
        pl.col("received_at")
        .sort_by(pl.col("_gap"), descending=True, nulls_last=True)
        .first()
        .alias("largest_gap_to"),
    )
    row = lf.collect()

    first_received_at = row.item(0, "first_received_at")
    if first_received_at is None:
        return PacketCompletenessSummary(None, None, None, None)

    largest_gap = row.item(0, "largest_gap")
    has_gap = largest_gap is not None
    return PacketCompletenessSummary(
        first_received_at=first_received_at,
        largest_gap=largest_gap,
        largest_gap_from=row.item(0, "largest_gap_from") if has_gap else None,
        largest_gap_to=row.item(0, "largest_gap_to") if has_gap else None,
    )
