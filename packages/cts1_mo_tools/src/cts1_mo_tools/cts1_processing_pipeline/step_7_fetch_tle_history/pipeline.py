"""Step 7: fetch the satellite's TLE history.

Every TLE Space-Track has published for the satellite, from the
`juliensimon/space-track-tle-history` Hugging Face dataset (read straight
from its parquet files with DuckDB's `hf://` support), so that a time in
the past can be propagated from the TLE published right before it rather
than from today's -- see `orbit.py`, which step 8 uses to place each MPI
collection.

The dataset is only updated about once a day, and the query takes tens of
seconds, so this is refetched at most every `REFRESH_INTERVAL`. A failed
fetch (e.g. no network) keeps the last good `tle_history.parquet` and only
logs a warning: nothing downstream should stop over stale TLEs.

Like step 0, this is a sourcing step: it doesn't depend on any other step.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "OUTPUT_FILENAME",
    "REFRESH_INTERVAL",
    "TLE_HISTORY_SCHEMA",
    "fetch_tle_history",
    "run",
]

import time
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.common import connect_duckdb
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "tle_history.parquet"
REFRESH_INTERVAL = timedelta(hours=6)

# https://huggingface.co/datasets/juliensimon/space-track-tle-history
_DATASET_GLOB = "hf://datasets/juliensimon/space-track-tle-history/data/tle_*.parquet"

# One row per published TLE, oldest first. The dataset stores several of
# these as Float32; they're widened here, which doesn't matter for accuracy.
TLE_HISTORY_SCHEMA = {
    "norad_id": pl.Int64,
    "epoch": pl.Datetime("us", "UTC"),
    "inclination": pl.Float64,  # Degrees.
    "raan": pl.Float64,  # Degrees.
    "eccentricity": pl.Float64,
    "arg_perigee": pl.Float64,  # Degrees.
    "mean_anomaly": pl.Float64,  # Degrees.
    "mean_motion": pl.Float64,  # Revolutions per day.
    "mean_motion_dot": pl.Float64,
    "bstar": pl.Float64,
    "intl_designator": pl.String,
    "altitude_km": pl.Float64,
}


def fetch_tle_history(norad_id: str) -> pl.DataFrame:
    """Every TLE in the dataset for `norad_id`, oldest first -- see
    `TLE_HISTORY_SCHEMA`. Blocking, and needs the network.
    """
    with connect_duckdb() as con:
        # Its progress bar would otherwise fill the daemon's log.
        con.execute("SET enable_progress_bar = false")
        df = con.execute(
            f"""
            SELECT {", ".join(TLE_HISTORY_SCHEMA)}
            FROM read_parquet('{_DATASET_GLOB}')
            WHERE norad_id = $norad_id
            ORDER BY epoch
            """,  # noqa: S608 -- only constants are interpolated.
            {"norad_id": int(norad_id)},
        ).pl()
    # DuckDB hands back `epoch` (a TIMESTAMPTZ) in the machine's local time
    # zone -- the same instants, but converted so every reader sees UTC.
    return df.with_columns(pl.col("epoch").dt.convert_time_zone("UTC")).select(
        pl.col(name).cast(dtype) for name, dtype in TLE_HISTORY_SCHEMA.items()
    )


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, norad_id: str = "69015", data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Refresh `tle_history.parquet` in `data_dir`, unless it was already
    refreshed within the last `REFRESH_INTERVAL`.
    """
    out_path = data_dir / OUTPUT_FILENAME
    if out_path.exists():
        age = timedelta(seconds=time.time() - out_path.stat().st_mtime)
        if age < REFRESH_INTERVAL:
            logger.info(
                f"{out_path} is {age.total_seconds() / 3600:.1f} h old (under "
                f"{REFRESH_INTERVAL.total_seconds() / 3600:.0f} h) -- not refetching."
            )
            return

    logger.info(f"Fetching the TLE history of NORAD {norad_id} from {_DATASET_GLOB}")
    try:
        df = fetch_tle_history(norad_id)
    except Exception as e:  # noqa: BLE001 -- any failure: keep the old file.
        logger.warning(
            f"Couldn't fetch the TLE history ({e!r}) -- "
            + ("keeping the existing one." if out_path.exists() else "none yet.")
        )
        return
    if df.is_empty():
        logger.warning(f"No TLEs found for NORAD {norad_id} -- not writing anything.")
        return

    _write_parquet_atomic(df, out_path)
    logger.info(
        f"Done. {df.height:,} TLE(s) written to {out_path}, with epochs from "
        f"{df['epoch'].min()} to {df['epoch'].max()}."
    )
