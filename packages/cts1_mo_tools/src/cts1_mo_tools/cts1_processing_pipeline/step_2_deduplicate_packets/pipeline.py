"""Step 2: deduplicate packets across decoders/observations.

`raw_packets` (see step 1) has one row per *decode*: the same physical
transmission commonly shows up several times over -- once per decoder that
managed to decode it (sso_rx_replay / gr_satellites_pdu /
gr_satellites_kiss / satnogs_data_demod), and again per SatNOGS ground
station that happened to record the same overpass. This step collapses
those into one row per *distinct received packet*, each carrying a single
best-guess `received_at` and JSON columns tracing back to every
observation/decoder that contributed to it.

Before any of the trust/merge logic below, every row is filtered to
`rs_correctable` (RS-uncorrectable frames are dropped) and a `data_hex` that
starts with CTS-SAT-1's own CSP header (`CTS1_CSP_HEADER_HEX`), so noise
from other satellites/garbage decodes never enters the clustering step.

Trust model:
  - `askew_demod_from_file` and `sso_rx_replay` (in that trust order -- see
    `BASELINE_DECODERS`) are the most trustworthy decoders in both content
    and timing (each reports a within-file offset resolved against the
    observation's own start time). Their packets form the baseline: two
    baseline-decoder decodes of the same content are merged if received
    within `BASELINE_DEDUPE_TOLERANCE` of each other -- multiple SatNOGS
    ground stations time-synced to within about a minute of real time
    commonly all catch the same overpass. When a merged baseline cluster
    has an `askew_demod_from_file` decode, its timestamp wins as the
    cluster's `received_at`; otherwise the earliest `sso_rx_replay` decode
    is used.
  - Every other decoder is treated as far less trustworthy on timing (some,
    like gr_satellites_pdu, can't localize a frame within its observation at
    all and just report the observation's end time). A same-content packet
    from another decoder is merged into a baseline packet if it falls
    within `OTHER_DECODER_TOLERANCE` of that baseline packet, or if its own
    observation's window overlaps the observation window(s) the baseline
    packet was seen in.
  - Content with no baseline-decoder decode anywhere is still worth keeping
    (a decoder-exclusive packet isn't nothing), so it's clustered against
    itself using the same wide, low-trust tolerance and reported with its
    earliest contributing decode as a best-effort `received_at`.

Unlike step 1 -- which appends to a DuckDB database as its source of truth,
since it's cheap to run incrementally against new SatNOGS observations --
this step and every step after it are parquet-in-parquet-out: it reads
`raw_packets.parquet`/`raw_observations.parquet` (as exported by step 1)
and writes `distinct_packets_over_time.parquet` straight back out, no
database involved. The whole dataset is cheap enough to reprocess from
scratch on every run, so there's no incremental state to reconcile.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_FILENAME",
    "compute_distinct_packets",
    "run",
]

import hashlib
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import CSP_CRC32C_SIZE, crc32c
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DEFAULT_OUTPUT_DIR = step_1_pipeline.DEFAULT_DB_PATH.parent
OUTPUT_FILENAME = "distinct_packets_over_time.parquet"

ASKEW_DECODER = "askew_demod_from_file"
SSO_DECODER = "sso_rx_replay"
DEMOD_DECODER = "satnogs_data_demod"

# Both have trustworthy within-file timing, so both anchor the baseline
# clustering below -- in this trust order (highest first) for picking a
# cluster's authoritative received_at/decoders when both are present.
BASELINE_DECODERS = (ASKEW_DECODER, SSO_DECODER)

# CTS-SAT-1's CSP header, as the leading bytes of every genuine `data_hex`.
CTS1_CSP_HEADER_HEX = "c2a28a00"

# Multiple ground stations recording the same overpass are all assumed
# time-synced to within about a minute of real time.
BASELINE_DEDUPE_TOLERANCE = timedelta(minutes=1)

# Everything else gets a much wider berth, since e.g. gr_satellites_pdu
# reports the observation's end time for every single frame.
OTHER_DECODER_TOLERANCE = timedelta(minutes=15)

_SOURCE_FIELDS: Sequence[str] = (
    "observation_id",
    "decoder",
    "time_in_file_ms",
    "rssi",
    "rs_corrected_error_count",
    "csp_crc_valid",
    "csp_crc_source",
)


def _packet_id(key: str) -> str:
    """Deterministic packet id, stable across Polars versions.

    Polars' own `.hash()` isn't guaranteed stable across releases (it's not
    a documented, versioned hash), so `packet_id` -- which needs to compare
    equal across separate step-2 runs on the same data -- is computed with
    the standard library's `hashlib` instead, via `map_elements`.
    """
    return hashlib.sha256(key.encode()).hexdigest()[:16]  # first 8 bytes


def _append_crc32c(data_hex: str) -> str:
    """Append a freshly computed CRC-32C to `data_hex`, treating it as a bare
    payload with no trailing CRC.
    """
    payload = bytes.fromhex(data_hex)
    computed = crc32c(payload)
    return data_hex + computed.to_bytes(CSP_CRC32C_SIZE, "big").hex()


def _complete_missing_crc(packets: pl.DataFrame) -> pl.DataFrame:
    """satnogs_data_demod sometimes reports a packet with its trailing CSP
    CRC-32C already stripped and sometimes doesn't -- there's no way to tell
    from SatNOGS's API alone which case a given packet is. Any row from that
    decoder whose `data_hex` doesn't already carry a valid CRC (per step 1's
    `csp_crc_valid`) is treated as the "missing" case: a CRC-32C is computed
    over its bytes and appended, completing it into the same full packet
    another decoder that *did* keep the CRC would report for the identical
    transmission. That's what makes the content-based grouping below able to
    match them up at all -- two decoders' bytes-for-bytes-identical payloads
    always produce the same computed CRC.

    Adds `csp_crc_source`: "decoded" for a CRC that was already there and
    verified as-is, "computed" for one synthesized here (i.e. not
    independently confirmed by the satellite -- just internally consistent
    by construction).
    """
    needs_crc = (pl.col("decoder") == DEMOD_DECODER) & ~pl.col(
        "csp_crc_valid"
    ).fill_null(value=False)

    incomplete = packets.filter(needs_crc).with_columns(
        data_hex=pl.col("data_hex").map_elements(_append_crc32c, return_dtype=pl.Utf8),
        data_length_bytes=pl.col("data_length_bytes") + CSP_CRC32C_SIZE,
        csp_crc_valid=pl.lit(value=True),  # true by construction
        csp_crc_source=pl.lit("computed"),
    )
    complete = packets.filter(~needs_crc).with_columns(csp_crc_source=pl.lit("decoded"))
    return pl.concat([complete, incomplete], how="vertical_relaxed")


def _with_source_struct(packets: pl.DataFrame) -> pl.DataFrame:
    """Attach a `_source` struct column: one full per-decode trace record."""
    return packets.with_columns(
        _source=pl.struct(
            *(pl.col(f) for f in _SOURCE_FIELDS),
            received_at=pl.col("received_at").dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ"),
        )
    )


def _cluster_by_content_and_time(
    df: pl.DataFrame, *, tolerance: timedelta, cluster_col: str
) -> pl.DataFrame:
    """Assign `cluster_col`: rows with the same `data_hex` chained together
    by consecutive `received_at` gaps no larger than `tolerance`
    (gap-and-island / "sessionization").

    A long chain of close-together rows can end up spanning more than
    `tolerance` end-to-end even though no single gap in it exceeds
    `tolerance` -- an accepted approximation, standard for this kind of
    event clustering.
    """
    return (
        df.sort(["data_hex", "received_at"])
        .with_columns(
            _gap=pl.col("received_at") - pl.col("received_at").shift(1).over("data_hex")
        )
        .with_columns(
            **{
                cluster_col: (
                    (pl.col("_gap").is_null() | (pl.col("_gap") > tolerance))
                    .cum_sum()
                    .over("data_hex")
                )
            }
        )
        .drop("_gap")
    )


def _cluster_baseline(baseline: pl.DataFrame) -> pl.DataFrame:
    """Cluster askew_demod_from_file/sso_rx_replay rows into distinct baseline
    packet events.

    Both decoders have trustworthy timing, so they're clustered together by
    content+time same as a single decoder would be. Within a cluster, rows
    are reordered so `BASELINE_DECODERS`' highest-trust decoder present
    sorts first -- `.first()` below then picks that decoder's own
    received_at/metadata/decoder name (`received_at_source`) as the
    cluster's authoritative values, falling back to the next-most-trusted
    decoder present.
    """
    clustered = _cluster_by_content_and_time(
        baseline, tolerance=BASELINE_DEDUPE_TOLERANCE, cluster_col="cluster_id"
    )
    trust_rank = pl.col("decoder").replace_strict(
        {decoder: rank for rank, decoder in enumerate(BASELINE_DECODERS)},
        return_dtype=pl.UInt32,
    )
    clustered = clustered.sort(["data_hex", "cluster_id", trust_rank, "received_at"])
    return (
        clustered.group_by(["data_hex", "cluster_id"], maintain_order=True)
        .agg(
            pl.col("received_at").first(),  # highest-trust decoder's own time
            pl.col("decoder").first().alias("received_at_source"),
            pl.col("data_length_bytes").first(),
            pl.col("csp_crc_valid").first(),
            pl.col("csp_crc_source").first(),
            pl.col("rssi").first(),
            pl.col("rs_corrected_error_count").first(),
            pl.col("rs_correctable").first(),
            pl.col("obs_start").min().alias("cluster_obs_start"),
            pl.col("obs_end").max().alias("cluster_obs_end"),
            pl.col("observation_id").unique().alias("baseline_observation_ids"),
            pl.col("decoder").unique().alias("baseline_decoders"),
            pl.col("_source").alias("sources"),
        )
        .drop("cluster_id")
        .with_row_index("baseline_cluster_idx")
    )


def _match_others_to_baseline(
    others: pl.DataFrame, baseline_clusters: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Match each `others` row to its nearest same-content baseline cluster.

    A match requires: the same `data_hex`, and either the row's `received_at`
    falling within `OTHER_DECODER_TOLERANCE` of the cluster's own time span,
    or the row's observation window overlapping an observation window the
    baseline cluster was itself seen in.

    Returns `(matched, unmatched)`: `matched` has one row per `others` row
    that found a baseline match (nearest one, if several clusters of the
    same content exist), `unmatched` is everything left over.
    """
    empty_matched = others.clear().with_columns(
        baseline_cluster_idx=pl.lit(None, dtype=pl.UInt32)
    )
    if others.is_empty() or baseline_clusters.is_empty():
        return empty_matched, others

    pairs = others.join(
        baseline_clusters.select(
            "data_hex",
            "baseline_cluster_idx",
            "cluster_obs_start",
            "cluster_obs_end",
        ),
        on="data_hex",
        how="inner",
    )
    if pairs.is_empty():
        return empty_matched, others

    window_overlap = (pl.col("obs_start") <= pl.col("cluster_obs_end")) & (
        pl.col("obs_end") >= pl.col("cluster_obs_start")
    )
    dist = pl.min_horizontal(
        (pl.col("received_at") - pl.col("cluster_obs_start")).abs(),
        (pl.col("received_at") - pl.col("cluster_obs_end")).abs(),
    )
    pairs = pairs.with_columns(
        _dist=pl.when(window_overlap).then(pl.duration(microseconds=0)).otherwise(dist),
        _match_ok=window_overlap | (dist <= OTHER_DECODER_TOLERANCE),
    )

    matched = (
        pairs.filter(pl.col("_match_ok"))
        .sort("_dist")
        .unique(subset=["_row_id"], keep="first")
    )
    unmatched = others.join(matched.select("_row_id"), on="_row_id", how="anti")
    return matched, unmatched


def _attach_others_to_baseline(
    baseline_clusters: pl.DataFrame, matched: pl.DataFrame
) -> pl.DataFrame:
    """Fold matched low-trust decodes into their baseline cluster's row."""
    if matched.is_empty():
        attached = baseline_clusters.with_columns(
            _other_decoders=pl.lit([], dtype=pl.List(pl.String)),
            _other_observation_ids=pl.lit([], dtype=pl.List(pl.Int64)),
            _other_sources=pl.lit([], dtype=baseline_clusters.schema["sources"]),
        )
    else:
        agg = matched.group_by("baseline_cluster_idx").agg(
            pl.col("decoder").alias("_other_decoders"),
            pl.col("observation_id").alias("_other_observation_ids"),
            pl.col("_source").alias("_other_sources"),
        )
        attached = baseline_clusters.join(agg, on="baseline_cluster_idx", how="left")
        attached = attached.with_columns(
            pl.col("_other_decoders").fill_null([]),
            pl.col("_other_observation_ids").fill_null([]),
            pl.col("_other_sources").fill_null([]),
        )

    return attached.with_columns(
        decoders=pl.col("_other_decoders")
        .list.concat(pl.col("baseline_decoders"))
        .list.unique()
        .list.sort(),
        observation_ids=pl.concat_list(
            [pl.col("baseline_observation_ids"), pl.col("_other_observation_ids")]
        )
        .list.unique()
        .list.sort(),
        sources=pl.concat_list([pl.col("sources"), pl.col("_other_sources")]),
    ).select(
        "data_hex",
        "received_at",
        "received_at_source",
        "data_length_bytes",
        "csp_crc_valid",
        "csp_crc_source",
        "rssi",
        "rs_corrected_error_count",
        "rs_correctable",
        "decoders",
        "observation_ids",
        "sources",
    )


def _cluster_leftover_others(unmatched: pl.DataFrame) -> pl.DataFrame:
    """Distinct-packet rows for content with no baseline-decoder decode at
    all: clustered against themselves with the wide, low-trust tolerance,
    reporting the earliest contributing decode as a best-effort `received_at`.
    """
    clustered = _cluster_by_content_and_time(
        unmatched, tolerance=OTHER_DECODER_TOLERANCE, cluster_col="cluster_id"
    )
    return (
        clustered.group_by(["data_hex", "cluster_id"], maintain_order=True)
        .agg(
            pl.col("received_at").min(),  # earliest available guess
            pl.col("data_length_bytes").first(),
            pl.col("csp_crc_valid").first(),
            pl.col("csp_crc_source").first(),
            pl.lit(None).alias("rssi"),
            pl.lit(None).alias("rs_corrected_error_count"),
            pl.lit(None).alias("rs_correctable"),
            pl.col("decoder").unique().sort().alias("decoders"),
            pl.col("observation_id").unique().sort().alias("observation_ids"),
            pl.col("_source").alias("sources"),
        )
        .drop("cluster_id")
        .with_columns(received_at_source=pl.lit("estimated"))
    )


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Encode the traceability columns to JSON and add id/bookkeeping columns."""
    quoted_decoders = pl.col("decoders").list.eval(pl.format('"{}"', pl.element()))
    encoded_sources = pl.col("sources").list.eval(pl.element().struct.json_encode())
    df = df.with_columns(
        decoders="[" + quoted_decoders.list.join(",") + "]",
        observation_ids=(
            "[" + pl.col("observation_ids").cast(pl.List(pl.Utf8)).list.join(",") + "]"
        ),
        packet_count=pl.col("sources").list.len(),
        sources="[" + encoded_sources.list.join(",") + "]",
    )
    df = df.with_columns(
        packet_id=pl.concat_str(
            ["data_hex", pl.col("received_at").dt.strftime("%Y-%m-%dT%H:%M:%S%.6fZ")]
        ).map_elements(_packet_id, return_dtype=pl.Utf8)
    )
    return df.select(
        "packet_id",
        "data_hex",
        "data_length_bytes",
        "csp_crc_valid",
        "csp_crc_source",
        "received_at",
        "received_at_source",
        "rssi",
        "rs_corrected_error_count",
        "rs_correctable",
        "packet_count",
        "decoders",
        "observation_ids",
        "sources",
    ).sort("received_at")


def compute_distinct_packets(
    packets_df: pl.DataFrame, observations_df: pl.DataFrame
) -> pl.DataFrame:
    """Pure computation: `raw_packets`/`raw_observations` -> distinct packets.

    Args:
        packets: `raw_packets.parquet`, as exported by step 1.
        observations: `raw_observations.parquet`, as exported by step 1.

    Returns:
        One row per distinct received packet -- see module docstring for the
        merge/trust rules, and `_finalize` for the output schema.
    """
    observations_df = observations_df.select(
        observation_id=pl.col("id"),
        obs_start=pl.col("start").dt.replace_time_zone("UTC"),
        obs_end=pl.col("end").dt.replace_time_zone("UTC"),
    )

    packets_df = packets_df.filter(
        pl.col("data_hex").is_not_null()
        & (pl.col("data_hex") != "")
        & pl.col("data_hex").str.starts_with(CTS1_CSP_HEADER_HEX)
        & pl.col("rs_correctable")
    )
    packets_df = _complete_missing_crc(packets_df)
    is_baseline = pl.col("decoder").is_in(BASELINE_DECODERS)

    packets_df = packets_df.join(observations_df, on="observation_id", how="left")
    packets_df = _with_source_struct(packets_df)

    baseline = packets_df.filter(is_baseline).with_row_index("_row_id")
    others = packets_df.filter(~is_baseline).with_row_index("_row_id")

    baseline_clusters = _cluster_baseline(baseline)
    matched, unmatched = _match_others_to_baseline(others, baseline_clusters)

    baseline_final = _attach_others_to_baseline(baseline_clusters, matched)
    leftover_final = _cluster_leftover_others(unmatched)

    result = pl.concat([baseline_final, leftover_final], how="diagonal_relaxed")
    return _finalize(result)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Recompute `distinct_packets_over_time.parquet` from step 1's exports.

    Reads `raw_packets.parquet`/`raw_observations.parquet` from `output_dir`
    (where step 1 exports them) and writes the result back into the same
    directory -- parquet-in-parquet-out, no database involved.
    """
    packets_path = output_dir / "raw_packets.parquet"
    observations_path = output_dir / "raw_observations.parquet"
    for path in (packets_path, observations_path):
        if not path.exists():
            msg = f"{path} not found -- run step_1 first."
            raise FileNotFoundError(msg)

    logger.info(f"Reading {packets_path} and {observations_path}")
    packets_df = pl.read_parquet(packets_path)
    observations_df = pl.read_parquet(observations_path)

    logger.info(
        f"Loaded {len(packets_df):,} raw packets across "
        f"{len(observations_df):,} raw observations."
    )

    result_df = compute_distinct_packets(packets_df, observations_df)

    out_path = output_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)

    logger.info(f"Done. {len(result_df):,} distinct packet(s) written to {out_path}.")

    mean_packet_count = result_df["packet_count"].mean()
    median_packet_count = result_df["packet_count"].median()
    logger.info(
        f"On average, each packet was received+decoded "
        f"{mean_packet_count:.2f} times (mean) "
        f"or {median_packet_count:.1f} times (median)."
    )

    # `packets_df.height` includes rows compute_distinct_packets drops before
    # grouping (empty data_hex, RS-uncorrectable sso frames) -- those were
    # never "received+decoded copies" of anything, so they're excluded here
    # rather than inflating the ratio above.
    contributing_packets = result_df["packet_count"].sum()
    dropped_packets = packets_df.height - contributing_packets
    logger.info(
        f"{packets_df.height:,} raw packets loaded, "
        f"{dropped_packets:,} dropped as noise (e.g., RS errors, empty data_hex), "
        f"{contributing_packets:,} received+decoded into "
        f"{result_df.height:,} distinct packets."
    )
