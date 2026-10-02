"""Step 0's tables in the shared landing-zone DuckDB database.

  - raw_observations: one row per SatNOGS observation, upserted by `id`.
    Tolerant of new columns showing up in later runs (the SatNOGS API is
    allowed to grow fields over time). Step 1 reads its decode candidates
    from here.
  - observation_listing_history: one row per attempt at listing one
    listing window from the SatNOGS API, append-only -- including failed
    attempts, with the error that ended them.
  - observation_listing_windows: one row per listing window that has ever
    been listed successfully, upserted by `window_start`, with when it was
    last listed and whether it needs listing again (`needs_refetch`).
    Setting `needs_refetch` to true by hand is how to ask the next run to
    re-list a window.

See `step_0_list_observations.pipeline` for what a listing window is.
"""

from __future__ import annotations

__all__ = [
    "LISTING_HISTORY_TABLE",
    "LISTING_WINDOWS_TABLE",
    "RAW_OBSERVATIONS_TABLE",
    "ListingRecord",
    "export_parquets",
    "record_listing",
    "upsert_observations",
    "window_refetch_flags",
]

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.landing_db import (
    add_missing_columns,
    export_table_to_parquet,
    insert_with_type_repair,
    quote_ident,
    table_exists,
)

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb
    import polars as pl

RAW_OBSERVATIONS_TABLE = "raw_observations"
LISTING_HISTORY_TABLE = "observation_listing_history"
LISTING_WINDOWS_TABLE = "observation_listing_windows"


@dataclass(frozen=True, slots=True)
class ListingRecord:
    """One attempt at listing one listing window from the SatNOGS API."""

    window_start: datetime
    window_end: datetime
    query_start_gt: datetime
    """Lower bound actually sent to the API (window start minus overlap)."""
    query_start_lt: datetime
    """Upper bound actually sent to the API (window end plus overlap, but
    never past the time of the listing)."""
    started_at: datetime
    finished_at: datetime
    observation_count: int
    """Observations the API returned (so far, if the attempt failed)."""
    needs_refetch: bool
    """Whether the window still needs listing again after this attempt."""
    error: str | None = None
    """Why the attempt failed, or None if it succeeded."""

    @property
    def succeeded(self) -> bool:
        return self.error is None


def upsert_observations(
    con: duckdb.DuckDBPyConnection, df: pl.DataFrame, *, key_col: str = "id"
) -> None:
    """Insert/replace rows in raw_observations, keyed by `id`."""
    if df.is_empty():
        return

    con.register("_incoming_observations", df)
    try:
        if not table_exists(con, RAW_OBSERVATIONS_TABLE):
            con.execute(
                f"CREATE TABLE {quote_ident(RAW_OBSERVATIONS_TABLE)} AS "  # noqa: S608
                f"SELECT * FROM _incoming_observations"
            )
        else:
            add_missing_columns(con, RAW_OBSERVATIONS_TABLE, "_incoming_observations")
            con.execute(
                f"DELETE FROM {quote_ident(RAW_OBSERVATIONS_TABLE)} "  # noqa: S608
                f"WHERE {quote_ident(key_col)} IN "
                f"(SELECT {quote_ident(key_col)} FROM _incoming_observations)"
            )
            insert_with_type_repair(
                con, RAW_OBSERVATIONS_TABLE, "_incoming_observations"
            )
    finally:
        con.unregister("_incoming_observations")

    logger.debug(f"{RAW_OBSERVATIONS_TABLE}: upserted {len(df)} row(s)")


def _create_listing_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_ident(LISTING_HISTORY_TABLE)} ("
        f"window_start TIMESTAMPTZ NOT NULL, "
        f"window_end TIMESTAMPTZ NOT NULL, "
        f"query_start_gt TIMESTAMPTZ NOT NULL, "
        f"query_start_lt TIMESTAMPTZ NOT NULL, "
        f"started_at TIMESTAMPTZ NOT NULL, "
        f"finished_at TIMESTAMPTZ NOT NULL, "
        f"succeeded BOOLEAN NOT NULL, "
        f"observation_count INTEGER NOT NULL, "
        f"needs_refetch BOOLEAN NOT NULL, "
        f"error VARCHAR)"
    )
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_ident(LISTING_WINDOWS_TABLE)} ("
        f"window_start TIMESTAMPTZ PRIMARY KEY, "
        f"window_end TIMESTAMPTZ NOT NULL, "
        f"first_listed_at TIMESTAMPTZ NOT NULL, "
        f"last_listed_at TIMESTAMPTZ NOT NULL, "
        f"last_query_start_gt TIMESTAMPTZ NOT NULL, "
        f"last_query_start_lt TIMESTAMPTZ NOT NULL, "
        f"last_observation_count INTEGER NOT NULL, "
        f"listing_count INTEGER NOT NULL, "
        f"needs_refetch BOOLEAN NOT NULL)"
    )


def record_listing(con: duckdb.DuckDBPyConnection, record: ListingRecord) -> None:
    """Log `record` to the listing history, and -- if it succeeded -- make it
    the window's latest listing in the windows table.

    A failed attempt leaves the windows table alone: a window never listed
    successfully stays absent from it (so still gets listed next run), and
    one listed successfully before keeps that earlier listing's state.
    """
    _create_listing_tables(con)
    con.execute(
        f"INSERT INTO {quote_ident(LISTING_HISTORY_TABLE)} "  # noqa: S608
        f"(window_start, window_end, query_start_gt, query_start_lt, started_at, "
        f"finished_at, succeeded, observation_count, needs_refetch, error) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            record.window_start,
            record.window_end,
            record.query_start_gt,
            record.query_start_lt,
            record.started_at,
            record.finished_at,
            record.succeeded,
            record.observation_count,
            record.needs_refetch,
            record.error,
        ],
    )
    if not record.succeeded:
        return

    # `last_listed_at` is when the listing *started*, not finished: an
    # observation uploaded mid-listing may or may not have made it in, so
    # the start is the latest time the listing is known to be complete as of.
    con.execute(
        f"INSERT INTO {quote_ident(LISTING_WINDOWS_TABLE)} "  # noqa: S608
        f"(window_start, window_end, first_listed_at, last_listed_at, "
        f"last_query_start_gt, last_query_start_lt, last_observation_count, "
        f"listing_count, needs_refetch) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) "
        f"ON CONFLICT (window_start) DO UPDATE SET "
        f"window_end = excluded.window_end, "
        f"last_listed_at = excluded.last_listed_at, "
        f"last_query_start_gt = excluded.last_query_start_gt, "
        f"last_query_start_lt = excluded.last_query_start_lt, "
        f"last_observation_count = excluded.last_observation_count, "
        f"listing_count = listing_count + 1, "
        f"needs_refetch = excluded.needs_refetch",
        [
            record.window_start,
            record.window_end,
            record.started_at,
            record.started_at,
            record.query_start_gt,
            record.query_start_lt,
            record.observation_count,
            record.needs_refetch,
        ],
    )


def window_refetch_flags(con: duckdb.DuckDBPyConnection) -> dict[datetime, bool]:
    """Map each successfully-listed window's start (UTC) to its `needs_refetch`.

    A window missing from the result has never been listed successfully.
    """
    if not table_exists(con, LISTING_WINDOWS_TABLE):
        return {}
    # Via polars rather than `fetchall()`: DuckDB needs pytz to hand back
    # TIMESTAMPTZ values as Python objects, while polars uses zoneinfo.
    rows = con.execute(
        f"SELECT window_start, needs_refetch "  # noqa: S608
        f"FROM {quote_ident(LISTING_WINDOWS_TABLE)}"
    ).pl()
    return {
        window_start.astimezone(UTC): needs for window_start, needs in rows.iter_rows()
    }


def export_parquets(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Copy step 0's tables out to Parquet files next to db_path."""
    out_dir = db_path.parent
    order_by = {
        RAW_OBSERVATIONS_TABLE: ["id"],
        LISTING_HISTORY_TABLE: ["started_at", "window_start"],
        LISTING_WINDOWS_TABLE: ["window_start"],
    }
    for table, order_by_columns in order_by.items():
        if table_exists(con, table):
            export_table_to_parquet(
                con,
                table=table,
                out_path=out_dir / f"{table}.parquet",
                order_by_columns=order_by_columns,
            )
