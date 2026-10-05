"""Step 2: deduplicate packets across decoders/observations.

`raw_packets` (see step 1) has one row per *decode*: the same physical
transmission commonly shows up several times over -- once per decoder that
managed to decode it (sso_rx_replay / gr_satellites_pdu /
gr_satellites_kiss / satnogs_client_live_data), and again per SatNOGS ground
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

__all__ = [
    "DEFAULT_DATA_DIR",
    "OUTPUT_FILENAME",
    "compute_distinct_packets",
    "run",
]

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl
import polars_hash
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import CSP_CRC32C_SIZE
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)

DEFAULT_DATA_DIR = step_1_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "distinct_packets_over_time.parquet"

ASKEW_DECODER = "askew_demod_from_file"
SSO_DECODER = "sso_rx_replay"
DEMOD_DECODER = "satnogs_client_live_data"

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
    "rssi_db",
    "rs_corrected_error_count",
    "csp_crc_valid",
    "csp_crc_source",
)


def _complete_missing_crc(packets: pl.LazyFrame) -> pl.LazyFrame:
    """satnogs_client_live_data sometimes reports a packet with its trailing CSP
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
    needs_crc = (pl.col("decoder") == pl.lit(DEMOD_DECODER)) & pl.col(
        "csp_crc_valid"
    ).fill_null(value=False).not_()

    incomplete = (
        packets.filter(needs_crc)
        .with_columns(
            temp_data_binary=pl.col("data_hex").str.decode("hex"),
        )
        .with_columns(
            data_hex=(
                pl.col("data_hex")
                # Add on the computed CRC32.
                + polars_hash.col("temp_data_binary")
                .nchash.crc32c(return_binary=True, byte_order="big")
                .bin.encode("hex")
            ),
            data_length_bytes=pl.col("data_length_bytes") + CSP_CRC32C_SIZE,
            csp_crc_valid=(
                # True by construction.
                pl.lit(value=True, dtype=pl.Boolean)
            ),
            csp_crc_source=pl.lit("computed"),
        )
        .drop("temp_data_binary")
    )
    complete = packets.filter(~needs_crc).with_columns(csp_crc_source=pl.lit("decoded"))
    return pl.concat([complete, incomplete], how="vertical_relaxed")


def _prepare_rows(packets: pl.LazyFrame, observations: pl.LazyFrame) -> pl.LazyFrame:
    """Filter out noise, complete missing CRCs, and attach everything the
    clustering below keys on.

    Adds, per decode:
      - `_row_id`: a stable row key, so the clustering can work on narrow
        key-only frames and join its verdict back onto the full rows once.
      - `_data`: `data_hex` parsed to raw bytes (half the size, and
        hex-case-insensitive); `data_hex` itself is dropped until
        `_finalize` re-encodes it once per distinct packet. A value that
        isn't valid hex raises: every step 1 decoder hex-encodes bytes, so
        that means step 1 is broken, not radio noise.
      - `_content_id`: a dense integer stand-in for `_data` (an exact rank,
        not a hash, so distinct contents can never collide). Every sort,
        window, join and group-by below keys on content, and packets run to
        hundreds of bytes -- comparing those millions of times over is most
        of the cost if done on the bytes themselves.
      - `_trust_rank`: position in `BASELINE_DECODERS` (null for others).
      - `_source`: one full per-decode trace record, already JSON-encoded.
    """
    observations = observations.select(
        observation_id=pl.col("id"),
        obs_start=pl.col("start").dt.replace_time_zone("UTC"),
        obs_end=pl.col("end").dt.replace_time_zone("UTC"),
    )
    packets = packets.filter(
        pl.col("data_hex").is_not_null()
        & (pl.col("data_hex") != "")
        & pl.col("rs_correctable")
    )
    return (
        _complete_missing_crc(packets)
        .with_columns(_data=pl.col("data_hex").str.decode("hex"))
        .drop("data_hex")
        .filter(pl.col("_data").bin.starts_with(bytes.fromhex(CTS1_CSP_HEADER_HEX)))
        .join(observations, on="observation_id", how="left")
        .with_row_index("_row_id")
        .with_columns(
            _content_id=pl.col("_data").rank("dense").cast(pl.UInt32),
            _trust_rank=pl.col("decoder").replace_strict(
                {decoder: rank for rank, decoder in enumerate(BASELINE_DECODERS)},
                default=None,
                return_dtype=pl.UInt32,
            ),
            _source=pl.struct(
                *(pl.col(f) for f in _SOURCE_FIELDS),
                received_at=pl.col("received_at").dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ"),
            ).struct.json_encode(),
        )
    )


def _cluster_by_content_and_time(
    df: pl.LazyFrame, *, tolerance: timedelta
) -> pl.LazyFrame:
    """Assign `_cluster`: rows with the same `_content_id` chained together
    by consecutive `received_at` gaps no larger than `tolerance`
    (gap-and-island / "sessionization"). Cluster ids are unique across all
    content, not just within one content.

    A long chain of close-together rows can end up spanning more than
    `tolerance` end-to-end even though no single gap in it exceeds
    `tolerance` -- an accepted approximation, standard for this kind of
    event clustering.
    """
    starts_new_cluster = (
        (pl.col("_content_id") != pl.col("_content_id").shift(1))
        | ((pl.col("received_at") - pl.col("received_at").shift(1)) > tolerance)
    ).fill_null(value=True)  # the very first row
    return df.sort(["_content_id", "received_at"]).with_columns(
        _cluster=starts_new_cluster.cum_sum()
    )


def _assign_clusters(rows: pl.LazyFrame) -> pl.LazyFrame:
    """Decide which distinct packet each decode belongs to.

    Returns one `(_row_id, _cluster, _is_leftover)` row per input row:
    `(_cluster, _is_leftover)` is the distinct packet's key.

      - Baseline-decoder rows are clustered by content+time with
        `BASELINE_DEDUPE_TOLERANCE`.
      - Every other row is matched to its nearest same-content baseline
        cluster: either its `received_at` falls within
        `OTHER_DECODER_TOLERANCE` of the cluster's observation-window span,
        or its own observation window overlaps that span.
      - Whatever's left is clustered against itself with the wide
        `OTHER_DECODER_TOLERANCE` (`_is_leftover`).

    Everything here works on narrow key/timestamp-only columns: the
    other-to-baseline join fans out to every same-content baseline cluster
    before the nearest is picked, so for content that repeats a lot (e.g.
    identical idle beacons) it's the one place the row count can blow up.
    """
    keys = ["_row_id", "_content_id", "received_at", "obs_start", "obs_end"]
    is_baseline = pl.col("_trust_rank").is_not_null()

    baseline = _cluster_by_content_and_time(
        rows.filter(is_baseline).select(keys), tolerance=BASELINE_DEDUPE_TOLERANCE
    )
    others = rows.filter(~is_baseline).select(keys)

    cluster_spans = baseline.group_by("_cluster").agg(
        pl.col("_content_id").first(),
        pl.col("obs_start").min().alias("cluster_obs_start"),
        pl.col("obs_end").max().alias("cluster_obs_end"),
    )
    window_overlap = (pl.col("obs_start") <= pl.col("cluster_obs_end")) & (
        pl.col("obs_end") >= pl.col("cluster_obs_start")
    )
    dist = pl.min_horizontal(
        (pl.col("received_at") - pl.col("cluster_obs_start")).abs(),
        (pl.col("received_at") - pl.col("cluster_obs_end")).abs(),
    )
    matched = (
        others.join(cluster_spans, on="_content_id", how="inner")
        .with_columns(
            _dist=pl.when(window_overlap)
            .then(pl.duration(microseconds=0))
            .otherwise(dist),
        )
        .filter(window_overlap | (dist <= OTHER_DECODER_TOLERANCE))
        .group_by("_row_id")
        # Nearest cluster; ties (e.g. a window overlapping several) go to the
        # earliest, so the output is deterministic run to run.
        .agg(pl.col("_cluster").sort_by(["_dist", "_cluster"]).first())
    )
    leftover = _cluster_by_content_and_time(
        others.join(matched, on="_row_id", how="anti"),
        tolerance=OTHER_DECODER_TOLERANCE,
    )

    return pl.concat(
        [
            baseline.select("_row_id", "_cluster", _is_leftover=pl.lit(value=False)),
            matched.select("_row_id", "_cluster", _is_leftover=pl.lit(value=False)),
            leftover.select("_row_id", "_cluster", _is_leftover=pl.lit(value=True)),
        ]
    )


def _aggregate_clusters(rows: pl.LazyFrame) -> pl.LazyFrame:
    """Collapse each cluster's decodes into its one distinct-packet row.

    Rows are ordered within a cluster baseline-first (by `_trust_rank`, then
    `received_at`), so `.first()` picks the highest-trust, earliest baseline
    decode when there is one, and `sources` lists baseline decodes first.

    For a baseline cluster, only rows from `BASELINE_DECODERS`'
    highest-trust decoder *present* (never a mix of both) contribute to the
    cluster's `received_at`: their median timestamp. Several ground stations
    catching the same overpass can be time-synced to within a few seconds of
    each other rather than exactly, and propagation delay between stations
    is only milliseconds -- so which single copy arrived "first" mostly
    reflects which station's clock happens to run behind, not which
    timestamp is more correct. The median cancels that symmetric clock skew
    out instead of picking a side of it; with only one copy (the common
    case) it's just that copy's own time.

    `rssi_db` is picked separately: askew_demod_from_file is preferred over
    any other decoder, and if several askew_demod_from_file decodes landed
    in the same cluster (e.g. multiple ground stations), the strongest
    (max) of their rssi_db values is used. Only when no
    askew_demod_from_file decode is present does it fall back to the
    cluster's first (highest-trust) row's value.

    A leftover cluster (no baseline decode at all) reports its earliest
    contributing decode as a best-effort `received_at`, with no
    `rssi_db`/`rs_*` values.
    """
    is_askew = pl.col("decoder") == ASKEW_DECODER
    # The highest-trust decoder's own rows within the cluster -- i.e. every
    # row tied for the lowest `_trust_rank` present, not just the first one.
    # Null (never true) for non-baseline rows.
    is_best_rank = pl.col("_trust_rank") == pl.col("_trust_rank").min()
    if_baseline = pl.when(~pl.col("_is_leftover"))
    return (
        rows.sort(
            ["_is_leftover", "_cluster", "_trust_rank", "received_at"],
            nulls_last=True,
        )
        .group_by(["_is_leftover", "_cluster"], maintain_order=True)
        .agg(
            pl.col("_content_id").first(),
            pl.col("received_at").filter(is_best_rank).median().alias("_baseline_at"),
            pl.col("received_at").min().alias("_earliest_at"),
            pl.col("decoder").filter(is_best_rank).first().alias("received_at_source"),
            pl.col("data_length_bytes").first(),
            pl.col("csp_crc_valid").first(),
            pl.col("csp_crc_source").first(),
            pl.coalesce(
                pl.col("rssi_db").filter(is_askew).max(),
                pl.col("rssi_db").first(),
            ).alias("rssi_db"),
            pl.col("rs_corrected_error_count").first(),
            pl.col("rs_correctable").first(),
            pl.col("decoder").unique().sort().alias("decoders"),
            pl.col("observation_id").unique().sort().alias("observation_ids"),
            pl.col("_source").alias("sources"),
        )
        # Looked up only now, once per distinct packet, rather than carried
        # through the sort and group-by above.
        .join(
            rows.select("_content_id", "_data").unique("_content_id"),
            on="_content_id",
            how="left",
        )
        .with_columns(
            received_at=if_baseline.then("_baseline_at").otherwise("_earliest_at"),
            received_at_source=if_baseline.then("received_at_source").otherwise(
                pl.lit("estimated")
            ),
            rssi_db=if_baseline.then("rssi_db"),
            rs_corrected_error_count=if_baseline.then("rs_corrected_error_count"),
            rs_correctable=if_baseline.then("rs_correctable"),
        )
    )


def _finalize(df: pl.LazyFrame) -> pl.LazyFrame:
    """Encode the traceability columns to JSON and add id/bookkeeping columns."""
    quoted_decoders = pl.col("decoders").list.eval(pl.format('"{}"', pl.element()))
    df = df.with_columns(
        data_hex=pl.col("_data").bin.encode("hex"),
        decoders="[" + quoted_decoders.list.join(",") + "]",
        observation_ids=(
            "["
            + pl.col("observation_ids").cast(pl.List(pl.String)).list.join(",")
            + "]"
        ),
        packet_count=pl.col("sources").list.len(),
        sources="[" + pl.col("sources").list.join(",") + "]",
    )
    df = df.with_columns(
        packet_id=(
            polars_hash.concat_str(
                [
                    "data_hex",
                    pl.col("received_at").dt.strftime("%Y-%m-%dT%H:%M:%S%.6fZ"),
                ]
            )
            .chash.sha2_256()
            .str.slice(0, 16)  # first 8 bytes
        )
    )
    return df.select(
        "packet_id",
        "data_hex",
        "data_length_bytes",
        "csp_crc_valid",
        "csp_crc_source",
        "received_at",
        "received_at_source",
        "rssi_db",
        "rs_corrected_error_count",
        "rs_correctable",
        "packet_count",
        "decoders",
        "observation_ids",
        "sources",
    ).sort("received_at")


def compute_distinct_packets(
    packets: pl.DataFrame | pl.LazyFrame, observations: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame:
    """Pure computation: `raw_packets`/`raw_observations` -> distinct packets.

    Args:
        packets: `raw_packets.parquet`, as exported by step 1 -- ideally a
            `pl.scan_parquet` LazyFrame, so the noise filters are pushed
            down into the scan.
        observations: `raw_observations.parquet`, as exported by step 1.

    Returns:
        One row per distinct received packet -- see module docstring for the
        merge/trust rules, and `_finalize` for the output schema.
    """
    # Materialized once up front: every branch below (clustering, matching,
    # the final aggregation) reads these rows and joins back on `_row_id`, so
    # they must all see the exact same `_row_id` assignment -- not depend on
    # the optimizer deduplicating the shared subplan.
    rows = _prepare_rows(packets.lazy(), observations.lazy()).collect().lazy()
    clusters = _assign_clusters(rows)
    clustered_rows = rows.join(clusters, on="_row_id", how="inner")
    return _finalize(_aggregate_clusters(clustered_rows)).collect()


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `distinct_packets_over_time.parquet` from step 1's exports.

    Reads `raw_packets.parquet`/`raw_observations.parquet` from `data_dir`
    (where step 1 exports them) and writes the result back into the same
    directory -- parquet-in-parquet-out, no database involved.
    """
    packets_path = data_dir / "raw_packets.parquet"
    observations_path = data_dir / "raw_observations.parquet"
    for path in (packets_path, observations_path):
        if not path.exists():
            msg = f"{path} not found -- run step_1 first."
            raise FileNotFoundError(msg)

    packets = pl.scan_parquet(packets_path)
    observations = pl.scan_parquet(observations_path)
    # Both counts come straight from the parquet footers.
    raw_packet_count = packets.select(pl.len()).collect().item()
    raw_observation_count = observations.select(pl.len()).collect().item()
    logger.info(
        f"Scanning {raw_packet_count:,} raw packets across "
        f"{raw_observation_count:,} raw observations "
        f"from {packets_path} and {observations_path}."
    )

    result_df = compute_distinct_packets(packets, observations)

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)

    stats = result_df.select(
        distinct=pl.len(),
        contributing=pl.col("packet_count").sum(),
        mean=pl.col("packet_count").mean(),
        median=pl.col("packet_count").median(),
    ).row(0, named=True)
    logger.info(
        f"Done. {stats['distinct']:,} distinct packet(s) written to {out_path}."
    )
    logger.info(
        f"On average, each packet was received+decoded "
        f"{stats['mean']:.2f} times (mean) "
        f"or {stats['median']:.1f} times (median)."
    )

    # `raw_packet_count` includes rows compute_distinct_packets drops before
    # grouping (empty data_hex, RS-uncorrectable sso frames) -- those were
    # never "received+decoded copies" of anything, so they're excluded here
    # rather than inflating the ratio above.
    dropped_packets = raw_packet_count - stats["contributing"]
    logger.info(
        f"{raw_packet_count:,} raw packets loaded, "
        f"{dropped_packets:,} dropped as noise (e.g., RS errors, empty data_hex), "
        f"{stats['contributing']:,} received+decoded into "
        f"{stats['distinct']:,} distinct packets."
    )
