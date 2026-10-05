"""Step 0: list observations.

Enumerate every SatNOGS observation of a satellite into DuckDB's
`raw_observations`, which step 1 then reads its decode candidates from --
this step never downloads or decodes anything itself.

The SatNOGS API is queried in fixed listing windows rather than one big
open-ended query: `LISTING_WINDOW`-long (12h) windows aligned to 00:00/12:00
UTC, each queried for observations starting within it plus
`LISTING_WINDOW_OVERLAP` (25 minutes) either side, so an observation
starting right on a boundary is caught by both neighbours rather than
falling between them (`raw_observations` is upserted by id, so the overlap
costs nothing but a few repeated rows from the API).

Every attempt at listing a window is logged to
`observation_listing_history`, and every window's latest successful listing
to `observation_listing_windows` (see `db`), which is what lets a run skip
windows already listed for good. A window needs (re-)listing when:

  - it has never been listed successfully, or
  - its latest listing started before the window had settled -- i.e.
    before `REFETCH_SETTLE_PERIOD` after the end of its query range, by
    which point every observation in it is expected to have finished and
    uploaded its audio/demodulated data. Until then, an observation can
    still be in progress, or finished but not uploaded yet, and a listing
    taken then may be missing data a later listing would see; or
  - someone set its `needs_refetch` by hand, or passed `refetch_all`.

Re-listing a window that's still settling is incremental: rather than the
whole window again, it only queries observations starting from
`RELIST_LOOKBACK` (45 minutes) before the window's previous listing, which
is what can have changed since -- new observations, plus ones that were
still in progress or uploading back then. Once the window has settled, it
gets one last full listing, which is the one trusted as final.

So a run that's repeated every few minutes (the daemon) only queries about
the last hour of observations -- the current window's tail, plus the
previous window's for its first couple of hours -- with a full 12h listing
of a window once, after it settles, and of any window an earlier run
failed to list or never got to (e.g. after an interruption).

Invoked via the top-level CLI's `step_0` subcommand -- see
`cts1_mo_tools.cts1_processing_pipeline.cli`.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_HISTORY_START",
    "LISTING_WINDOW",
    "LISTING_WINDOW_OVERLAP",
    "REFETCH_SETTLE_PERIOD",
    "RELIST_LOOKBACK",
    "ListingWindow",
    "incremental_since",
    "list_window",
    "listing_windows",
    "run",
    "window_containing",
    "windows_needing_listing",
]

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline import landing_db
from cts1_mo_tools.cts1_processing_pipeline.common import parse_start_filter
from cts1_mo_tools.satnogs_data import fetch_all_observations

from . import db

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import duckdb

DEFAULT_DATA_DIR = landing_db.DEFAULT_DATA_DIR

LISTING_WINDOW = timedelta(hours=12)
LISTING_WINDOW_OVERLAP = timedelta(minutes=25)
# How long after the end of a window's query range a listing of it is
# trusted to be final -- see the module docstring. Generous on purpose:
# re-listing a settling window is a handful of API pages, while missing an
# observation that uploaded late means never decoding it.
REFETCH_SETTLE_PERIOD = timedelta(hours=2)
# How far before a still-settling window's previous listing an incremental
# re-listing of it reaches back -- see the module docstring. Covers an
# observation that was in progress, or finished but not yet uploaded, when
# the previous listing ran (the old trailing-requery daemon's margin).
RELIST_LOOKBACK = timedelta(minutes=45)
# Where listing starts when no `start` is given. Comfortably before
# CTS-SAT-1's first SatNOGS observation (2026-08-08); pass an explicit
# `start` to reach further back for another satellite.
DEFAULT_HISTORY_START = datetime(2026, 8, 1, tzinfo=UTC)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ListingWindow:
    """One `LISTING_WINDOW`-long window, [start, end), aligned to the epoch."""

    start: datetime

    @property
    def end(self) -> datetime:
        return self.start + LISTING_WINDOW

    @property
    def settled_at(self) -> datetime:
        """When a listing of this window can first be trusted to be final."""
        return self.end + LISTING_WINDOW_OVERLAP + REFETCH_SETTLE_PERIOD

    def query_bounds(
        self, now: datetime, *, since: datetime | None = None
    ) -> tuple[datetime, datetime]:
        """The (start_gt, start_lt) bounds to query the API with as of `now`.

        The overlap either side, but never past `now`: SatNOGS also lists
        scheduled future observations, which have nothing to decode yet.
        `since` narrows the lower bound for an incremental re-listing (see
        `incremental_since`); None queries the whole window.
        """
        start_gt = self.start - LISTING_WINDOW_OVERLAP
        if since is not None:
            start_gt = max(start_gt, since)
        return start_gt, min(self.end + LISTING_WINDOW_OVERLAP, now)


def window_containing(at: datetime) -> ListingWindow:
    """The window `at` falls in."""
    return ListingWindow(_EPOCH + ((at - _EPOCH) // LISTING_WINDOW) * LISTING_WINDOW)


def listing_windows(start: datetime, now: datetime) -> list[ListingWindow]:
    """Every window from the one containing `start` to the one containing
    `now`, newest first -- so a long backfill lands fresh data first.
    """
    windows: list[ListingWindow] = []
    window_start = window_containing(start).start
    while window_start < now:
        windows.append(ListingWindow(window_start))
        window_start += LISTING_WINDOW
    windows.reverse()
    return windows


def windows_needing_listing(
    windows: Sequence[ListingWindow],
    states: Mapping[datetime, db.WindowState],
    *,
    refetch_all: bool = False,
) -> list[ListingWindow]:
    """The subset of `windows` to list this run, in the same order.

    `states` is `db.window_states`: a window absent from it has never been
    listed successfully.
    """
    if refetch_all:
        return list(windows)
    return [
        w for w in windows if w.start not in states or states[w.start].needs_refetch
    ]


def incremental_since(
    window: ListingWindow, state: db.WindowState | None, now: datetime
) -> datetime | None:
    """The lower bound to re-list `window` from incrementally as of `now`,
    or None if it needs a full listing.

    Incremental only while the window is still settling and its previous
    listing was a settling-period one too. A full listing is needed for a
    window never listed before; for its final listing once it has settled;
    and for a settled window someone flagged `needs_refetch` by hand.
    """
    if state is None or not state.needs_refetch:
        return None
    if now >= window.settled_at or state.last_listed_at >= window.settled_at:
        return None
    return state.last_listed_at - RELIST_LOOKBACK


def _observations_frame(page: list[dict[str, Any]]) -> pl.DataFrame:
    """One API page of observations, typed for `raw_observations`."""
    observations_df = pl.DataFrame(page, infer_schema_length=None)
    return observations_df.with_columns(
        pl.col("start").str.to_datetime(time_unit="us", time_zone="UTC"),
        pl.col("end").str.to_datetime(time_unit="us", time_zone="UTC"),
        pl.col("demoddata").struct.json_encode(),
    )


def list_window(
    con: duckdb.DuckDBPyConnection,
    *,
    norad_id: str,
    window: ListingWindow,
    now: datetime,
    since: datetime | None = None,
) -> db.ListingRecord:
    """List one window into raw_observations, returning how it went.

    `since` makes it an incremental re-listing -- see `incremental_since`.

    A failure partway through (the API erroring past its retries) is
    caught and returned as a failed record, so one bad window doesn't stop
    the rest of the run; whatever pages landed before it stay landed.
    """
    query_start_gt, query_start_lt = window.query_bounds(now, since=since)
    started_at = datetime.now(UTC)
    observation_count = 0
    error: str | None = None

    try:
        for page in fetch_all_observations(
            norad_id,
            start_gt_filter=query_start_gt,
            start_lt_filter=query_start_lt,
            statuses=None,
        ):
            db.upsert_observations(con, _observations_frame(page))
            observation_count += len(page)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed listing window starting {window.start.isoformat()}")
        error = f"{type(exc).__name__}: {exc}"

    # A query cut short by `now` hasn't covered the whole window yet, however
    # long the run took to get to it.
    is_truncated = query_start_lt < window.end + LISTING_WINDOW_OVERLAP
    return db.ListingRecord(
        window_start=window.start,
        window_end=window.end,
        query_start_gt=query_start_gt,
        query_start_lt=query_start_lt,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        observation_count=observation_count,
        needs_refetch=(
            error is not None or is_truncated or started_at < window.settled_at
        ),
        error=error,
    )


def run(
    *,
    norad_id: str = "69015",
    data_dir: Path = DEFAULT_DATA_DIR,
    start: str | None = None,
    end: str | None = None,
    refetch_all: bool = False,
) -> None:
    """List every window from `start` to `end` (default: now) that needs
    (re-)listing.

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        data_dir: Directory to write/find the DuckDB database
            (`landing_db.DB_FILENAME`) in -- raw_observations and the
            listing tables land there, and their parquet exports land
            alongside it.
        start: How far back to list: a duration like "3 days" (relative to
            now) or an ISO 8601 date/datetime. Rounded down to the start of
            the listing window it falls in. None means
            `DEFAULT_HISTORY_START`.
        end: Where listing stops, in the same syntax as `start`: only
            windows starting before it are listed (the window it falls in
            included). None means now. Never past now either way.
        refetch_all: Re-list every window from `start` to now, ignoring
            the record of which ones are already listed for good.
    """
    db_path = data_dir / landing_db.DB_FILENAME
    now = datetime.now(UTC)
    start_at = parse_start_filter(start) if start is not None else DEFAULT_HISTORY_START
    end_at = min(parse_start_filter(end), now) if end is not None else now
    windows = listing_windows(start_at, end_at)

    with landing_db.connect(db_path) as con:
        states = db.window_states(con)
        to_list = windows_needing_listing(windows, states, refetch_all=refetch_all)
        logger.info(
            f"Listing observations for NORAD {norad_id} since "
            f"{start_at.isoformat()}"
            + (f" until {end_at.isoformat()}" if end is not None else "")
            + f": {len(to_list)} of {len(windows)} "
            f"{LISTING_WINDOW}-long window(s) need listing"
            + (" (--refetch-all)" if refetch_all else "")
        )

        total_observations = 0
        failed = 0
        run_started_at = time.monotonic()
        for number, window in enumerate(to_list, start=1):
            since = (
                None
                if refetch_all
                else incremental_since(window, states.get(window.start), now)
            )
            record = list_window(
                con, norad_id=norad_id, window=window, now=now, since=since
            )
            db.record_listing(con, record)
            total_observations += record.observation_count
            failed += not record.succeeded
            scope = "full" if since is None else f"since {since.isoformat()}"
            logger.info(
                f"[{number}/{len(to_list)}] window {window.start.isoformat()} "
                f"({scope}): {record.observation_count} observation(s)"
                + (", still settling" if record.needs_refetch else "")
                + ("" if record.succeeded else " -- FAILED, will retry next run")
            )

        con.execute("CHECKPOINT")
        db.export_parquets(con, db_path)

    if failed:
        logger.error(
            f"{failed} of {len(to_list)} window(s) failed to list; they'll be "
            f"retried on the next run."
        )
    logger.info(
        f"Done. {len(to_list)} window(s) listed ({total_observations} "
        f"observation(s), counting overlaps twice) in "
        f"{time.monotonic() - run_started_at:.1f}s."
    )
