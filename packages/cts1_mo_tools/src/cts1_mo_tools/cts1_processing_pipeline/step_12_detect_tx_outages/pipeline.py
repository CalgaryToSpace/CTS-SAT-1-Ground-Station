"""Step 12: estimate when the satellite stopped transmitting (TX outages).

The satellite has a glitched mode where it stops transmitting for hours.
From the ground, that looks like a long gap in packets -- but so does
nobody listening. So this step only calls a gap an outage when stations
that nearly always hear the satellite were listening and heard nothing:

1. Per station (`ground_station`), its hit rate: the share of its decoded
   observations (any `decoder_runs` row without an error) with at least one
   good packet (`rs_correctable` and `csp_crc_valid`), Laplace-smoothed
   (`(hits + 1) / (n + 2)`). A "reliable" station
   has at least `RELIABLE_STATION_MIN_OBSERVATIONS` of them and a hit rate
   of at least `RELIABLE_STATION_MIN_HIT_RATE`.
2. Every gap of at least `MIN_OUTAGE_DURATION` between consecutive good
   packets (from any station) is a candidate, as is the gap since the
   latest good packet (an ongoing one, measured up to the end of the
   latest decoded observation).
3. A candidate is an outage if reliable stations' observations that lie
   wholly inside it all missed, from at least `MIN_MISSING_STATIONS`
   different stations (one station missing a pass is often just that
   station).
4. Outages dilute the hit rates they're found with, so the hit rates are
   estimated again without the observations inside outages, and the
   outages found again with those.

`probability_by_chance` is how likely all those misses would be with the
satellite transmitting normally: the product of each missed observation's
station miss rate (1 - hit rate). That assumes misses are independent,
which they aren't quite (a bad attitude can hurt several passes in a row),
so it's a rough guide, not a calibrated probability.

`last_uncorrectable_packet_at` is the latest packet that failed Reed-Solomon
correction after the last good packet but before the first missed pass: a
sign the satellite was still transmitting (weakly, or garbled) after the
last good packet. (Not just the latest one in the gap: that's nearly always
a garbled first frame of the recovery pass.)

Parquet-in-parquet-out, reading step 1's exports (`raw_observations`,
`raw_packets`, `decoder_runs`), and cheap enough to recompute every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "MIN_MISSING_STATIONS",
    "MIN_OUTAGE_DURATION",
    "OUTPUT_FILENAME",
    "RELIABLE_STATION_MIN_HIT_RATE",
    "RELIABLE_STATION_MIN_OBSERVATIONS",
    "compute_tx_outages",
    "run",
]

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "tx_outages.parquet"

RELIABLE_STATION_MIN_OBSERVATIONS = 50
RELIABLE_STATION_MIN_HIT_RATE = 0.6
MIN_OUTAGE_DURATION = timedelta(hours=3)
MIN_MISSING_STATIONS = 2

_UTC = pl.Datetime("us", "UTC")
_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=UTC)
_OUTPUT_SCHEMA = {
    "last_good_packet_at": _UTC,
    "last_uncorrectable_packet_at": _UTC,
    "first_good_packet_after_at": _UTC,
    "is_ongoing": pl.Boolean,
    "gap_duration_ms": pl.Int64,
    "missed_observation_count": pl.UInt32,
    "missed_station_count": pl.UInt32,
    "missed_stations": pl.List(pl.String),
    "missed_observation_ids": pl.List(pl.Int64),
    "probability_by_chance": pl.Float64,
}


def _find_outages(
    observations: pl.DataFrame, gaps: pl.DataFrame, excluded_ids: pl.Series
) -> pl.DataFrame:
    """Steps 1 and 3 of the module docstring, with hit rates estimated
    without the observations in `excluded_ids`."""
    hit_rates = (
        observations.filter(~pl.col("id").is_in(excluded_ids.implode()))
        .group_by("ground_station")
        # Laplace-smoothed, so a perfect record doesn't make a miss impossible.
        .agg(n=pl.len(), hit_rate=(pl.col("hit").sum() + 1) / (pl.len() + 2))
        .filter(
            (pl.col("n") >= RELIABLE_STATION_MIN_OBSERVATIONS)
            & (pl.col("hit_rate") >= RELIABLE_STATION_MIN_HIT_RATE)
        )
    )
    missed = (
        observations.join(hit_rates, on="ground_station")
        .filter(~pl.col("hit"))
        .join_where(
            gaps,
            pl.col("start") > pl.col("last_good_packet_at"),
            pl.col("end") < pl.col("gap_end"),
        )
    )
    return (
        missed.group_by("last_good_packet_at")
        .agg(
            missed_observation_count=pl.len(),
            missed_station_count=pl.col("ground_station").n_unique(),
            missed_stations=pl.format("{} (#{})", "station_name", "ground_station")
            .unique()
            .sort(),
            missed_observation_ids=pl.col("id").sort(),
            probability_by_chance=(1 - pl.col("hit_rate")).product(),
            first_missed_start=pl.col("start").min(),
        )
        .filter(pl.col("missed_station_count") >= MIN_MISSING_STATIONS)
    )


def compute_tx_outages(
    observations: pl.DataFrame,
    decoder_runs: pl.DataFrame,
    packets: pl.LazyFrame,
) -> pl.DataFrame:
    """Pure computation: step 1's tables -> one row per TX outage, newest
    first. See the module docstring for the model, `_OUTPUT_SCHEMA` for the
    columns.

    Args:
        observations: `raw_observations` (`id`, `ground_station`,
            `station_name`, `start`, `end`).
        decoder_runs: `decoder_runs` (`observation_id`, `error`).
        packets: `raw_packets` (`observation_id`, `received_at`,
            `rs_correctable`, `csp_crc_valid`) -- lazy, as it's big.
    """
    packets = packets.select(
        "observation_id", "received_at", "rs_correctable", "csp_crc_valid"
    )
    is_good = pl.col("rs_correctable") & pl.col("csp_crc_valid")
    good_times = (
        packets.filter(is_good).select(t="received_at").unique().sort("t").collect()
    )
    if observations.is_empty() or good_times.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    hit_ids = packets.filter(is_good).select("observation_id").unique().collect()
    decoded_ids = decoder_runs.filter(pl.col("error").is_null())["observation_id"]
    decoded = observations.filter(
        pl.col("id").is_in(decoded_ids.implode())
        & pl.col("ground_station").is_not_null()
    ).with_columns(hit=pl.col("id").is_in(hit_ids["observation_id"].implode()))

    # The ongoing gap (after the latest good packet) has no end yet: treat it
    # as open-ended, and check its length up to the latest decoded pass (as
    # far as the data shows the satellite silent).
    gaps = good_times.select(
        last_good_packet_at=pl.col("t"),
        first_good_packet_after_at=pl.col("t").shift(-1),
    ).with_columns(gap_end=pl.col("first_good_packet_after_at").fill_null(_FAR_FUTURE))
    gaps = gaps.filter(
        pl.col("first_good_packet_after_at").is_null()
        | (
            pl.col("first_good_packet_after_at") - pl.col("last_good_packet_at")
            >= MIN_OUTAGE_DURATION
        )
    )

    outages = _find_outages(decoded, gaps, pl.Series([], dtype=pl.Int64))
    outages = _find_outages(
        decoded, gaps, outages["missed_observation_ids"].explode(empty_as_null=False)
    )

    uncorrectable = (
        packets.filter(~pl.col("rs_correctable"))
        .select(last_uncorrectable_packet_at="received_at")
        .sort("last_uncorrectable_packet_at")
        .collect()
    )
    result = (
        outages.join(gaps, on="last_good_packet_at")
        .sort("first_missed_start")
        .join_asof(
            uncorrectable,
            left_on="first_missed_start",
            right_on="last_uncorrectable_packet_at",
            strategy="backward",
            allow_exact_matches=False,
        )
        .with_columns(
            last_uncorrectable_packet_at=pl.when(
                pl.col("last_uncorrectable_packet_at") > pl.col("last_good_packet_at")
            ).then("last_uncorrectable_packet_at"),
            is_ongoing=pl.col("first_good_packet_after_at").is_null(),
            gap_duration_ms=(
                pl.col("first_good_packet_after_at") - pl.col("last_good_packet_at")
            ).dt.total_milliseconds(),
        )
        .filter(
            pl.col("first_good_packet_after_at").is_not_null()
            | (
                decoded["end"].max() - pl.col("last_good_packet_at")
                >= MIN_OUTAGE_DURATION
            )
        )
    )
    return result.select([pl.col(c).cast(t) for c, t in _OUTPUT_SCHEMA.items()]).sort(
        "last_good_packet_at", descending=True
    )


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `tx_outages.parquet` from step 1's exports in `data_dir`."""
    paths = {
        name: data_dir / f"{name}.parquet"
        for name in ("raw_observations", "raw_packets", "decoder_runs")
    }
    for path in paths.values():
        if not path.exists():
            msg = f"{path} not found -- run step_1 first."
            raise FileNotFoundError(msg)

    result = compute_tx_outages(
        pl.read_parquet(
            paths["raw_observations"],
            columns=["id", "ground_station", "station_name", "start", "end"],
        ),
        pl.read_parquet(paths["decoder_runs"], columns=["observation_id", "error"]),
        pl.scan_parquet(paths["raw_packets"]),
    )
    out_path = data_dir / OUTPUT_FILENAME
    tmp_path = out_path.with_suffix(".tmp")
    result.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    logger.info(
        f"Done. {result.height:,} TX outage(s) written to {out_path} "
        f"({result['is_ongoing'].sum()} ongoing)."
    )
