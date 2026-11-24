"""Shared helpers across the processing pipeline's steps and web UI."""

from __future__ import annotations

__all__ = [
    "DEFAULT_DUCKDB_MEMORY_LIMIT",
    "DEFAULT_DUCKDB_THREADS",
    "connect_duckdb",
    "dedupe_by_beacon_time",
    "default_duckdb_memory_limit",
    "drop_timezones_for_excel",
    "parse_start_filter",
]

import os
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import duckdb
import polars as pl

from cts1_mo_tools.cts1_processing_pipeline import resource_limits

if TYPE_CHECKING:
    from pathlib import Path

# Limit DuckDB to 1/8th of the host's memory, clamped within 500MB to 4GB.
_DUCKDB_MEMORY_SHARE = 8
_DUCKDB_MEMORY_FLOOR_MB = 500
_DUCKDB_MEMORY_CEILING_MB = 4000


def default_duckdb_memory_limit() -> str:
    """DuckDB's working-memory cap for this box -- see the comment above."""
    usable = resource_limits.usable_memory_bytes()
    if usable is None:
        return f"{_DUCKDB_MEMORY_FLOOR_MB}MB"
    megabytes = usable // _DUCKDB_MEMORY_SHARE // 1_000_000
    clamped = min(max(megabytes, _DUCKDB_MEMORY_FLOOR_MB), _DUCKDB_MEMORY_CEILING_MB)
    return f"{clamped}MB"


DEFAULT_DUCKDB_MEMORY_LIMIT = (
    os.environ.get("CTS1_DUCKDB_MEMORY_LIMIT") or default_duckdb_memory_limit()
)

DEFAULT_DUCKDB_THREADS = resource_limits.env_int(
    "CTS1_DUCKDB_THREADS", resource_limits.half_the_cores()
)


def connect_duckdb(
    database: str | Path = ":memory:",
    *,
    memory_limit: str = DEFAULT_DUCKDB_MEMORY_LIMIT,
    threads: int = DEFAULT_DUCKDB_THREADS,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with `memory_limit`/`threads` capped up front.

    The session time zone is pinned to UTC (rather than the host's), so
    TIMESTAMPTZ values exported to polars (`.pl()`) come back in UTC. With a
    local zone (e.g. America/Edmonton), exported values were seen off by an
    hour after a DST change, since DuckDB and polars each resolve the zone's
    offsets themselves.
    """
    con = duckdb.connect(database)
    con.execute("SET memory_limit = ?", [memory_limit])
    con.execute("SET threads = ?", [threads])
    con.execute("SET TimeZone = 'UTC'")
    return con


def drop_timezones_for_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Excel has no timezone-aware datetime type -- normalize every
    tz-aware Datetime column to naive UTC before handing `df` to
    `write_excel`, or xlsxwriter raises trying to format the cell.
    """
    exprs = [
        pl.col(name).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
    ]
    return df.with_columns(exprs) if exprs else df


_DURATION_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>second|minute|hour|day|week)s?\s*$",
    re.IGNORECASE,
)
_DURATION_UNIT_TO_TIMEDELTA_KWARG = {
    "second": "seconds",
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
}


def parse_start_filter(value: str, *, now: datetime | None = None) -> datetime:
    """Parse --start as either a relative duration or an absolute date/datetime.

    A duration like "3 days" or "6 hours" is measured back from `now`
    (default: the current time, UTC). Anything else is parsed as ISO 8601
    ("2026-08-01" or "2026-08-01T00:00:00Z"); a value with no timezone is
    treated as UTC.

    Raises:
        ValueError: If `value` matches neither form.
    """
    m = _DURATION_RE.match(value)
    if m:
        amount = float(m.group("value"))
        kwarg = _DURATION_UNIT_TO_TIMEDELTA_KWARG[m.group("unit").lower()]
        return (now or datetime.now(UTC)) - timedelta(**{kwarg: amount})

    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        msg = (
            f"Could not parse --start={value!r}; expected a duration like "
            f"'3 days' or an ISO 8601 date/datetime."
        )
        raise ValueError(msg) from exc

    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def dedupe_by_beacon_time(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """One row per distinct onboard beacon time `column`, sorted by it.

    The same beacon can be received more than once: live (" X5") and again
    later replayed from storage (" S5"), or by several ground stations. Of
    the copies, keep a CRC-valid one, then a live one, then the earliest
    received -- so each beacon counts once.
    """
    schema = lf.collect_schema()
    preference = [
        (expr, descending)
        for name, expr, descending in (
            # CRC-valid first.
            ("csp_crc_valid", pl.col("csp_crc_valid").fill_null(value=False), True),
            # Live first (null = not a v5+ beacon, so not stored either).
            (
                "is_stored_beacon",
                pl.col("is_stored_beacon").fill_null(value=False),
                False,
            ),
            # Earliest received first.
            ("received_at", pl.col("received_at"), False),
        )
        if name in schema
    ]
    if preference:
        lf = lf.sort(
            [expr for expr, _ in preference],
            descending=[descending for _, descending in preference],
            nulls_last=True,
        )
    return lf.unique(column, keep="first", maintain_order=True).sort(column)
