"""Daemon: run steps 1-3 continuously.

First does one backfill of `start` (a duration like "24 hours" or an ISO
8601 date/datetime -- same syntax as step 1's own `--start`), running steps
1 through 3 once. Then, every `interval` minutes, requeries step 1 for
observations starting in the trailing `interval + 30` minutes and reruns
steps 2 and 3 again -- the 30-minute overlap is there so a SatNOGS
observation still uploading/being vetted during one poll gets picked up on
the next one, rather than falling into the gap between two non-overlapping
windows.

The overlap doesn't waste decode time: step 1 already skips any
observation/decoder pair recorded in `decoder_runs` (see `force_rerun` in
`step_1_download_and_demodulate.pipeline.run`), so requerying the same
trailing window repeatedly only ever does new work for observations that
weren't fully decoded yet.

Invoked via the top-level CLI's `daemon` subcommand -- see
`cts1_mo_tools.cts1_processing_pipeline.cli`.
"""

from __future__ import annotations

__all__ = ["run"]

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from .step_1_download_and_demodulate import pipeline as step_1_pipeline
from .step_2_deduplicate_packets import pipeline as step_2_pipeline
from .step_3_decode_packets import pipeline as step_3_pipeline

if TYPE_CHECKING:
    from pathlib import Path

# The overlap added to `interval` for every requery after the initial
# backfill -- see the module docstring for why.
REQUERY_OVERLAP = timedelta(minutes=30)


def _run_all_steps(  # noqa: PLR0913
    *,
    norad_id: str,
    db_path: Path,
    start: str | None,
    limit: int | None,
    workers: int,
    temp_dir: Path | None,
    force_rerun: bool,
) -> None:
    step_1_pipeline.run(
        norad_id=norad_id,
        db_path=db_path,
        start=start,
        limit=limit,
        workers=workers,
        temp_dir=temp_dir,
        force_rerun=force_rerun,
    )
    step_2_pipeline.run(output_dir=db_path.parent)
    step_3_pipeline.run(output_dir=db_path.parent)


def run(  # noqa: PLR0913
    *,
    norad_id: str = "69015",
    db_path: Path = step_1_pipeline.DEFAULT_DB_PATH,
    start: str = "24 hours",
    interval: float = 15.0,
    limit: int | None = None,
    workers: int = 4,
    temp_dir: Path | None = None,
    force_rerun: bool = False,
) -> None:
    """Run steps 1-3 forever: one backfill of `start`, then a requery of
    the trailing `interval + 30` minutes every `interval` minutes.

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        db_path: DuckDB database file step 1 appends to; later steps use
            its directory to find/write their parquet files.
        start: How far back the initial backfill reaches: a duration like
            "3 days" (relative to now) or an ISO 8601 date/datetime -- see
            `step_1_download_and_demodulate.pipeline._parse_start_filter`.
        interval: Minutes between requeries.
        limit: Cap the number of observations decoded per step-1 run (for
            testing).
        workers: Concurrency for step 1's decoders.
        temp_dir: Directory for step 1's per-observation temp dirs. None
            uses the platform default.
        force_rerun: Passed through to every step-1 run -- see
            `step_1_download_and_demodulate.pipeline.run`.

    Runs until interrupted (Ctrl+C / SIGINT) -- `KeyboardInterrupt` is left
    to propagate to the caller (the top-level CLI catches it and exits
    cleanly).
    """
    logger.info(f"Daemon: initial backfill, start={start!r}")
    _run_all_steps(
        norad_id=norad_id,
        db_path=db_path,
        start=start,
        limit=limit,
        workers=workers,
        temp_dir=temp_dir,
        force_rerun=force_rerun,
    )

    interval_delta = timedelta(minutes=interval)
    requery_window = interval_delta + REQUERY_OVERLAP

    while True:
        logger.info(f"Daemon: sleeping {interval} minute(s) until next requery")
        time.sleep(interval_delta.total_seconds())

        requery_start = (datetime.now(UTC) - requery_window).isoformat()
        logger.info(f"Daemon: requerying since {requery_start}")
        _run_all_steps(
            norad_id=norad_id,
            db_path=db_path,
            start=requery_start,
            limit=limit,
            workers=workers,
            temp_dir=temp_dir,
            force_rerun=force_rerun,
        )
