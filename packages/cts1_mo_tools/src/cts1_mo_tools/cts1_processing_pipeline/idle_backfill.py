"""Idle backfill: fill in history older than the daemon's `start` while idle.

The daemon only ever covers `start` onwards (24 hours by default). With
idle backfill enabled, it also spends the time it would otherwise sleep
between requeries running steps 0 and 1 over the history before that, back
to `step_0_list_observations.pipeline.DEFAULT_HISTORY_START` (just before
the satellite's first SatNOGS observation) -- one step-0 listing window
(12h) at a time, newest first, so it can stop between chunks to honour a
web UI trigger request or the next scheduled requery.

A window is a chunk worth backfilling when either:

  - step 0 still needs to list it (see `windows_needing_listing`), or
  - step 1 still has an observation starting in it to decode (see
    `step_1_download_and_demodulate.pipeline.pending_observation_starts`).

Both checks read the same DuckDB tables the steps themselves use, so a
window backfilled by an earlier daemon process (or by a one-off backfill
run) is never redone. Steps 2-5 aren't run per chunk: they reprocess
everything step 1 has landed on every requery, so backfilled packets show
up after the next one.

Each window is attempted at most once per daemon process, whatever the
outcome -- a window that keeps failing to list, or an observation whose
audio keeps failing to download, would otherwise be retried back to back
for as long as the daemon is idle. Restarting the daemon retries them.

SatNOGS rate-limits observation listing (240 requests/hour with an API
key), and the regular requeries need some of that, so a chunk only starts
while fewer than `LISTING_REQUESTS_PER_HOUR_LIMIT` listing requests went
out in the past hour -- see `satnogs_data.listing_requests_in_last_hour`.

Enabled with the daemon's `--idle-backfill`, or `CTS1_IDLE_BACKFILL=1` --
see `ENV_VAR`.
"""

from __future__ import annotations

__all__ = [
    "ENV_VAR",
    "LISTING_REQUESTS_PER_HOUR_LIMIT",
    "IdleBackfill",
    "enabled_by_env",
]

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

from loguru import logger

from cts1_mo_tools import satnogs_data

from . import landing_db
from .daemon_signals import DaemonState
from .step_0_list_observations import db as step_0_db
from .step_0_list_observations import pipeline as step_0_pipeline
from .step_1_download_and_demodulate import pipeline as step_1_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from .daemon_signals import StatusReporter
    from .step_0_list_observations.pipeline import ListingWindow

ENV_VAR = "CTS1_IDLE_BACKFILL"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# SatNOGS's 240 listing requests/hour, less 50 kept for the regular requeries.
LISTING_REQUESTS_PER_HOUR_LIMIT: Final = 240 - 50


def enabled_by_env() -> bool:
    """Whether `ENV_VAR` asks for idle backfill (1/true/yes/on)."""
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


@dataclass(slots=True)
class IdleBackfill:
    """Backfills steps 0 and 1 one listing window at a time -- see the
    module docstring.

    `until` is the daemon's resolved `start`: the regular runs cover
    everything from there on, so backfill covers the windows before it
    (the window it falls in included, since step 1's regular runs only
    decode that window from `until` onwards).
    """

    norad_id: str
    data_dir: Path
    until: datetime
    workers: int
    temp_dir: Path | None
    tools: tuple[str, ...] | None
    since: datetime = step_0_pipeline.DEFAULT_HISTORY_START
    _attempted: set[datetime] = field(default_factory=set[datetime])
    _announced_caught_up: bool = False

    def pending_windows(self) -> list[ListingWindow]:
        """Windows still needing step 0 and/or step 1, newest first,
        excluding any already attempted by this process.
        """
        windows = step_0_pipeline.listing_windows(self.since, self.until)
        if not windows:
            return []
        with landing_db.connect(self.data_dir / landing_db.DB_FILENAME) as con:
            states = step_0_db.window_states(con)
            undecoded_starts = step_1_pipeline.pending_observation_starts(
                con,
                norad_id=self.norad_id,
                tools=self.tools,
                start_lt=windows[0].end,
            )
        pending = {
            w.start for w in step_0_pipeline.windows_needing_listing(windows, states)
        }
        pending.update(
            step_0_pipeline.window_containing(at).start for at in undecoded_starts
        )
        return [
            w for w in windows if w.start in pending and w.start not in self._attempted
        ]

    def run_next_chunk(self, reporter: StatusReporter) -> bool:
        """Backfill the newest pending window. Returns False, doing nothing,
        if there's none left.

        An exception from either step is logged rather than raised: idle
        backfill is opportunistic, and shouldn't take down the daemon's
        regular runs with it.

        Also returns False, doing nothing, while the past hour's SatNOGS
        listing requests are at `LISTING_REQUESTS_PER_HOUR_LIMIT`.
        """
        used = satnogs_data.count_listing_requests_in_last_hour()
        if used >= LISTING_REQUESTS_PER_HOUR_LIMIT:
            logger.info(
                f"Daemon: idle backfill paused -- {used} SatNOGS listing "
                f"request(s) in the past hour (limit "
                f"{LISTING_REQUESTS_PER_HOUR_LIMIT})"
            )
            return False
        try:
            pending = self.pending_windows()
        except Exception:  # noqa: BLE001
            logger.exception("Idle backfill: failed finding windows to backfill")
            return False
        if not pending:
            if not self._announced_caught_up:
                logger.info(
                    f"Daemon: idle backfill caught up -- nothing left to do "
                    f"between {self.since.isoformat()} and {self.until.isoformat()}"
                    + (
                        f" ({len(self._attempted)} window(s) attempted this run)"
                        if self._attempted
                        else ""
                    )
                )
                self._announced_caught_up = True
            return False

        window = pending[0]
        self._attempted.add(window.start)
        label = (
            f"idle backfill of {window.start.isoformat()} "
            f"({len(pending)} window(s) left)"
        )
        logger.info(f"Daemon: {label}")
        try:
            reporter.set(DaemonState.PROCESSING, detail=f"{label}: step 0")
            step_0_pipeline.run(
                norad_id=self.norad_id,
                data_dir=self.data_dir,
                start=window.start.isoformat(),
                end=window.end.isoformat(),
            )
            reporter.set(DaemonState.PROCESSING, detail=f"{label}: step 1")
            step_1_pipeline.run(
                norad_id=self.norad_id,
                data_dir=self.data_dir,
                start=window.start.isoformat(),
                end=window.end.isoformat(),
                workers=self.workers,
                temp_dir=self.temp_dir,
                tools=self.tools,
            )
        except Exception:  # noqa: BLE001
            logger.exception(f"Daemon: {label} failed; moving on")
        return True
