"""Daemon: run steps 1-4 continuously.

First does one backfill of `start` (a duration like "24 hours" or an ISO
8601 date/datetime -- same syntax as step 1's own `--start`), running steps
1 through 4 once. Then, every `interval` minutes, requeries step 1 for
observations starting in the trailing `interval + 30` minutes and reruns
steps 2 through 4 again -- the 30-minute overlap is there so a SatNOGS
observation still uploading/being vetted during one poll gets picked up on
the next one, rather than falling into the gap between two non-overlapping
windows.

The overlap doesn't waste decode time: step 1 already skips any
observation/decoder pair recorded in `decoder_runs` (see `force_rerun` in
`step_1_download_and_demodulate.pipeline.run`), so requerying the same
trailing window repeatedly only ever does new work for observations that
weren't fully decoded yet.

While sleeping between requeries, the daemon polls `data_dir` for a pipeline-trigger
request dropped there by the web UI's "Trigger Pipeline" button, and starts its
next run immediately when it finds one. It also publishes what it's
currently doing to a status file in the same directory, which is what backs
the web UI's "daemon is running..." indicator. Both files -- and the reason
they're files rather than anything cleverer -- are described in
`daemon_signals`.

Invoked via the top-level CLI's `daemon` subcommand -- see
`cts1_mo_tools.cts1_processing_pipeline.cli`.
"""

from __future__ import annotations

__all__ = ["STEP_NAMES", "run", "run_all_steps", "sleep_until_next_run"]

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from . import daemon_signals, resource_limits
from .daemon_signals import DaemonState, StatusReporter
from .step_1_download_and_demodulate import pipeline as step_1_pipeline
from .step_2_deduplicate_packets import pipeline as step_2_pipeline
from .step_3_decode_packets import pipeline as step_3_pipeline
from .step_4_detect_satellite_events import pipeline as step_4_pipeline

if TYPE_CHECKING:
    from pathlib import Path

# What each step is announced as, both in the log ("Starting step 2/4:
# deduplicate packets.") and in the web UI's status indicator. One table so
# the two can't drift apart, and so the step count in those messages stays
# right if a step 5 ever shows up.
STEP_NAMES = {
    1: "download and demodulate",
    2: "deduplicate packets",
    3: "decode packets",
    4: "detect satellite events",
}

# The overlap added to `interval` for every requery after the initial
# backfill -- see the module docstring for why.
REQUERY_OVERLAP = timedelta(minutes=30)

# How often the sleep between requeries wakes up to check for a trigger
# request from the web UI. Matched to the status heartbeat so one poll loop
# serves both directions -- see `daemon_signals`.
TRIGGER_POLL_INTERVAL_SEC = daemon_signals.HEARTBEAT_INTERVAL_SEC


def run_all_steps(  # noqa: PLR0913
    *,
    norad_id: str,
    data_dir: Path,
    start: str | None,
    limit: int | None,
    workers: int,
    temp_dir: Path | None,
    force_rerun: bool,
    tools: tuple[str, ...] | None,
    reporter: StatusReporter,
    run_label: str,
) -> None:
    """Run steps 1-4 once, publishing which step is in flight as it goes.

    `run_label` names this run in the status detail (and so in the web UI's
    indicator) -- e.g. the backfill vs. a scheduled requery vs. one a
    person asked for from the web UI.
    """

    def announce(number: int) -> None:
        """Log and publish that step `number` is starting."""
        name = STEP_NAMES[number]
        logger.info(f"Starting step {number}/{len(STEP_NAMES)}: {name} -- {run_label}.")
        reporter.set(
            DaemonState.PROCESSING, detail=f"{run_label}: step {number} ({name})"
        )

    announce(1)
    step_1_pipeline.run(
        norad_id=norad_id,
        data_dir=data_dir,
        start=start,
        limit=limit,
        workers=workers,
        temp_dir=temp_dir,
        force_rerun=force_rerun,
        tools=tools,
    )
    announce(2)
    step_2_pipeline.run(data_dir=data_dir)
    announce(3)
    step_3_pipeline.run(data_dir=data_dir)
    announce(4)
    step_4_pipeline.run(data_dir=data_dir)

    logger.info(f"Finished all {len(STEP_NAMES)} steps -- {run_label}.")


def sleep_until_next_run(
    *, data_dir: Path, interval: float, reporter: StatusReporter
) -> bool:
    """Sleep `interval` minutes, waking early for a web UI trigger request.

    Returns True if a trigger request cut the sleep short, False if the
    full interval elapsed. Consumes (deletes) the request before returning,
    so a request arriving *during* the run that follows is a fresh one that
    gets honoured on the next pass rather than being swallowed here.
    """
    interval_sec = timedelta(minutes=interval).total_seconds()
    deadline = time.monotonic() + interval_sec

    while True:
        request = daemon_signals.read_trigger_request(data_dir)
        if request is not None:
            daemon_signals.clear_trigger_request(data_dir)
            note = f" ({request.note})" if request.note else ""
            logger.info(f"Daemon: pipeline run triggered from the web UI{note}")
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False

        next_run_at = datetime.now(UTC) + timedelta(seconds=remaining)
        reporter.set(
            DaemonState.SLEEPING,
            detail=f"next requery in {remaining / 60:.1f} minute(s)",
            next_run_at=next_run_at,
        )
        time.sleep(min(TRIGGER_POLL_INTERVAL_SEC, remaining))


def run(  # noqa: PLR0913
    *,
    norad_id: str = "69015",
    data_dir: Path = step_1_pipeline.DEFAULT_DATA_DIR,
    start: str = "24 hours",
    interval: float = 15.0,
    limit: int | None = None,
    workers: int = resource_limits.DEFAULT_DECODER_WORKERS,
    temp_dir: Path | None = None,
    force_rerun: bool = False,
    tools: tuple[str, ...] | None = None,
) -> None:
    """Run steps 1-4 forever: one backfill of `start`, then a requery of
    the trailing `interval + 30` minutes every `interval` minutes.

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        data_dir: Directory step 1 reads/writes its DuckDB database in;
            every later step finds/writes its own parquet file(s) in the
            same directory.
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
        tools: Which step-1 decoders to run, from step 1's DECODERS. None
            (the default) runs all of them. Passed through to every step-1
            run.

    Publishes its state to `data_dir` throughout (see `daemon_signals`) and,
    while sleeping, honours a trigger request left there by the web UI.

    Runs until interrupted (Ctrl+C / SIGINT) -- `KeyboardInterrupt` is left
    to propagate to the caller (the top-level CLI catches it and exits
    cleanly), after the status file has been marked stopped.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    requery_window = timedelta(minutes=interval) + REQUERY_OVERLAP

    with StatusReporter(data_dir) as reporter:
        # A request sitting here from before the daemon started is about
        # data this backfill is going to cover anyway -- drop it rather
        # than letting it trigger an immediate redundant requery.
        daemon_signals.clear_trigger_request(data_dir)

        logger.info(f"Daemon: initial backfill, start={start!r}")
        run_all_steps(
            norad_id=norad_id,
            data_dir=data_dir,
            start=start,
            limit=limit,
            workers=workers,
            temp_dir=temp_dir,
            force_rerun=force_rerun,
            tools=tools,
            reporter=reporter,
            run_label="backfill",
        )

        while True:
            logger.info(f"Daemon: sleeping up to {interval} minute(s) until next run")
            was_triggered = sleep_until_next_run(
                data_dir=data_dir, interval=interval, reporter=reporter
            )

            requery_start = (datetime.now(UTC) - requery_window).isoformat()
            logger.info(f"Daemon: requerying since {requery_start}")
            run_all_steps(
                norad_id=norad_id,
                data_dir=data_dir,
                start=requery_start,
                limit=limit,
                workers=workers,
                temp_dir=temp_dir,
                force_rerun=force_rerun,
                tools=tools,
                reporter=reporter,
                run_label="requery (requested)" if was_triggered else "requery",
            )
