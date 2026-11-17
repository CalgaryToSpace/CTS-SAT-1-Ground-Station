"""Step 12: fetch the SatNOGS schedule.

Every SatNOGS observation of the satellite scheduled over the next
`HORIZON` (status "future"), so the web UI's Plan page can plan
downlinks for when stations are listening without calling the SatNOGS API
itself.

Refetched at most every `REFRESH_INTERVAL`. A failed fetch keeps the last
good file and only logs a warning. Like steps 0 and 7, a sourcing step: it
doesn't depend on any other step.
"""

from __future__ import annotations

__all__ = [
    "HORIZON",
    "OUTPUT_FILENAME",
    "REFRESH_INTERVAL",
    "SCHEDULE_SCHEMA",
    "fetch_schedule",
    "run",
]

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.step_7_fetch_tle_history import (
    pipeline as step_7_pipeline,
)
from cts1_mo_tools.cts1_satnogs_interval import fetch_all_observations

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_7_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "satnogs_schedule.parquet"
REFRESH_INTERVAL = timedelta(minutes=30)
HORIZON = timedelta(days=3)

# One row per scheduled observation, by start.
SCHEDULE_SCHEMA = {
    "observation_id": pl.Int64,
    "ground_station_id": pl.Int64,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "fetched_at": pl.Datetime("us", "UTC"),
}


def fetch_schedule(norad_id: str, now: datetime) -> pl.DataFrame:
    """Every future observation starting in the next `HORIZON` (or the last
    hour, so one in progress is kept). Blocking, and needs the network.
    """
    observations = fetch_all_observations(
        norad_id, start_gt=now - timedelta(hours=1), start_lt=now + HORIZON
    )
    return (
        pl.DataFrame(
            [
                {
                    "observation_id": o["id"],
                    "ground_station_id": o["ground_station"],
                    "start": datetime.fromisoformat(o["start"]),
                    "end": datetime.fromisoformat(o["end"]),
                    "fetched_at": now,
                }
                for o in observations
            ],
            schema=SCHEDULE_SCHEMA,
        )
        .unique("observation_id")
        .sort("start")
    )


def run(*, norad_id: str = "69015", data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Refresh `satnogs_schedule.parquet` in `data_dir`, unless it was
    already refreshed within the last `REFRESH_INTERVAL`.
    """
    out_path = data_dir / OUTPUT_FILENAME
    if out_path.exists():
        age = timedelta(seconds=time.time() - out_path.stat().st_mtime)
        if age < REFRESH_INTERVAL:
            logger.info(f"{out_path} is {age} old -- not refetching.")
            return

    logger.info(f"Fetching the SatNOGS schedule of NORAD {norad_id}")
    try:
        df = fetch_schedule(norad_id, datetime.now(UTC))
    except Exception as e:  # noqa: BLE001 -- any failure: keep the old file.
        logger.warning(f"Couldn't fetch the SatNOGS schedule ({e!r}).")
        return

    # Written even if empty: "nothing scheduled" is news too.
    tmp_path = out_path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    logger.info(f"Done. {df.height:,} scheduled observation(s) written to {out_path}.")
