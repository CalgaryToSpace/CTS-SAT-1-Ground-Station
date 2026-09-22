"""Shared helpers across the processing pipeline's steps and web UI."""

from __future__ import annotations

__all__ = [
    "DEFAULT_DUCKDB_MEMORY_LIMIT",
    "DEFAULT_DUCKDB_THREADS",
    "connect_duckdb",
    "drop_timezones_for_excel",
]

import os
from typing import TYPE_CHECKING

import duckdb
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

# DuckDB otherwise defaults to ~80% of *host* RAM -- far more than this
# pipeline's single-satellite data volume needs, and enough to crowd out
# everything else on a small deployment box (see
# `cts1_mo_tools/docs/resource-tuning.md`).
#
# Unlike a container memory cap, this one can't get anything OOM-killed:
# DuckDB spills to its temp directory when it hits this, and only raises
# `duckdb.OutOfMemoryException` if it can't. So the failure mode to look
# for if this is too low is an error in the log, not a dead daemon.
# Overridable (e.g. `CTS1_DUCKDB_MEMORY_LIMIT=1GB`) for a big backfill.
DEFAULT_DUCKDB_MEMORY_LIMIT = os.environ.get("CTS1_DUCKDB_MEMORY_LIMIT", "500MB")

# ...and, likewise, defaults to one thread per *host* core, on top of the
# thread pools polars and step 1's decoder pool have already sized
# themselves from the same core count (see `resource_limits`). One thread
# is plenty for the queries here, which are appends and full-table exports
# over a single satellite's packets rather than anything that parallelizes
# interestingly.
DEFAULT_DUCKDB_THREADS = 1


def connect_duckdb(
    database: str | Path = ":memory:",
    *,
    memory_limit: str = DEFAULT_DUCKDB_MEMORY_LIMIT,
    threads: int = DEFAULT_DUCKDB_THREADS,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with `memory_limit`/`threads` capped up front."""
    con = duckdb.connect(database)
    con.execute("SET memory_limit = ?", [memory_limit])
    con.execute("SET threads = ?", [threads])
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
