"""Daemon: run steps 0-6 continuously.

Runs steps 0 through 6 once as a backfill of `start` (a duration like "24
hours" or an ISO 8601 date/datetime -- same syntax as step 0/1's own
`--start`), then again every `interval` minutes.

`start` is resolved to an absolute time once, when the daemon starts, and
every run reuses it: step 0 only re-lists the tail of the SatNOGS listing
windows since then that aren't settled yet (about the last hour of
observations -- see `step_0_list_observations.pipeline`), and step 1 only decodes the
observations since then not already recorded in `decoder_runs` (see
`force_rerun` in `step_1_download_and_demodulate.pipeline.run`). So each
periodic run only does new work, without the daemon having to pick a
trailing window to requery: an observation that was still uploading
during one run is picked up by the next, because its window is
re-listed until it settles.

While sleeping between requeries, the daemon polls `data_dir` for a pipeline-trigger
request dropped there by the web UI's "Trigger Pipeline" button, and starts its
next run immediately when it finds one. It also publishes what it's
currently doing to a status file in the same directory, which is what backs
the web UI's "daemon is running..." indicator. Both files -- and the reason
they're files rather than anything cleverer -- are described in
`daemon_signals`.

With idle backfill enabled (`idle_backfill`, or the `CTS1_IDLE_BACKFILL`
environment variable), the daemon spends that sleep running steps 0 and 1
over the history before `start` instead, back to the satellite's first
observations, one 12h chunk at a time -- see `idle_backfill`.

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
from .common import parse_start_filter
from .daemon_signals import DaemonState, StatusReporter
from .idle_backfill import IdleBackfill
from .step_0_list_observations import pipeline as step_0_pipeline
from .step_1_download_and_demodulate import pipeline as step_1_pipeline
from .step_2_deduplicate_packets import pipeline as step_2_pipeline
from .step_3_decode_packets import pipeline as step_3_pipeline
from .step_4_detect_satellite_events import pipeline as step_4_pipeline
from .step_5_reassemble_tcmd_responses import pipeline as step_5_pipeline
from .step_6_deduplicate_gnss_samples import pipeline as step_6_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# What each step is announced as, both in the log ("Starting step 2/5:
# deduplicate packets.") and in the web UI's status indicator. One table so
# the two can't drift apart, and so the step count in those messages stays
# right as steps are added.
STEP_NAMES = {
    0: "list observations",
    1: "download and demodulate",
    2: "deduplicate packets",
    3: "decode packets",
    4: "detect satellite events",
    5: "reassemble telecommand responses",
    6: "deduplicate GNSS samples",
}
LAST_STEP = max(STEP_NAMES)

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
    """Run steps 0-6 once, publishing which step is in flight as it goes.

    `start` bounds both step 0's listing and step 1's decoding -- see the
    module docstring.

    `run_label` names this run in the status detail (and so in the web UI's
    indicator) -- e.g. the backfill vs. a scheduled requery vs. one a
    person asked for from the web UI.
    """

    def announce(number: int) -> None:
        """Log and publish that step `number` is starting."""
        name = STEP_NAMES[number]
        logger.info(f"Starting step {number}/{LAST_STEP}: {name} -- {run_label}.")
        reporter.set(
            DaemonState.PROCESSING, detail=f"{run_label}: step {number} ({name})"
        )

    announce(0)
    step_0_pipeline.run(norad_id=norad_id, data_dir=data_dir, start=start)
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
    # Steps 4, 5 and 6 are independent (all only read step 3's output); they
    # just run one after the other here to keep the daemon single-threaded.
    announce(5)
    step_5_pipeline.run(data_dir=data_dir)
    announce(6)
    step_6_pipeline.run(data_dir=data_dir)

    logger.info(f"Finished steps 0-{LAST_STEP} -- {run_label}.")


def sleep_until_next_run(
    *,
    data_dir: Path,
    interval: float,
    reporter: StatusReporter,
    idle_work: Callable[[StatusReporter], bool] | None = None,
) -> bool:
    """Sleep `interval` minutes, waking early for a web UI trigger request.

    Returns True if a trigger request cut the sleep short, False if the
    full interval elapsed. Consumes (deletes) the request before returning,
    so a request arriving *during* the run that follows is a fresh one that
    gets honoured on the next pass rather than being swallowed here.

    `idle_work`, if given, is called repeatedly in place of sleeping, each
    call doing one chunk of work and returning False once there's none
    left (after which this sleeps as usual). The trigger request and the
    deadline are checked between chunks, so a chunk in flight delays them
    -- keep chunks short.
    """
    interval_sec = timedelta(minutes=interval).total_seconds()
    deadline = time.monotonic() + interval_sec
    has_idle_work = idle_work is not None

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

        if has_idle_work:
            assert idle_work is not None
            has_idle_work = idle_work(reporter)
            if has_idle_work:
                continue

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
    idle_backfill: bool = False,
) -> None:
    """Run steps 0-6 forever: one backfill of `start`, then a rerun every
    `interval` minutes covering the same span (plus whatever's new).

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        data_dir: Directory steps 0/1 read/write their DuckDB database in;
            every later step finds/writes its own parquet file(s) in the
            same directory.
        start: How far back the initial backfill reaches: a duration like
            "3 days" (relative to now) or an ISO 8601 date/datetime -- see
            `common.parse_start_filter`. Resolved once, at startup, and
            reused by every run after.
        interval: Minutes between runs.
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
        idle_backfill: While waiting between runs, backfill steps 0 and 1
            for the history before `start` -- see `idle_backfill`.

    Publishes its state to `data_dir` throughout (see `daemon_signals`) and,
    while sleeping, honours a trigger request left there by the web UI.

    Runs until interrupted (Ctrl+C / SIGINT) -- `KeyboardInterrupt` is left
    to propagate to the caller (the top-level CLI catches it and exits
    cleanly), after the status file has been marked stopped.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    # Pinned down now, so "24 hours" means the 24 hours before startup on
    # every run, rather than a window that slides forward and stops
    # covering observations listed/decoded on earlier runs.
    start_at = parse_start_filter(start).isoformat()

    def run_once(run_label: str) -> None:
        run_all_steps(
            norad_id=norad_id,
            data_dir=data_dir,
            start=start_at,
            limit=limit,
            workers=workers,
            temp_dir=temp_dir,
            force_rerun=force_rerun,
            tools=tools,
            reporter=reporter,
            run_label=run_label,
        )

    backfiller = (
        IdleBackfill(
            norad_id=norad_id,
            data_dir=data_dir,
            until=datetime.fromisoformat(start_at),
            workers=workers,
            temp_dir=temp_dir,
            tools=tools,
        )
        if idle_backfill
        else None
    )

    with StatusReporter(data_dir) as reporter:
        # A request sitting here from before the daemon started is about
        # data this backfill is going to cover anyway -- drop it rather
        # than letting it trigger an immediate redundant requery.
        daemon_signals.clear_trigger_request(data_dir)

        logger.info(f"Daemon: initial backfill, start={start!r} ({start_at})")
        run_once("backfill")

        while True:
            logger.info(f"Daemon: sleeping up to {interval} minute(s) until next run")
            was_triggered = sleep_until_next_run(
                data_dir=data_dir,
                interval=interval,
                reporter=reporter,
                idle_work=backfiller.run_next_chunk if backfiller else None,
            )

            logger.info(f"Daemon: requerying since {start_at}")
            run_once("requery (requested)" if was_triggered else "requery")
