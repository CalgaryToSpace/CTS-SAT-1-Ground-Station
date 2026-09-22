"""Shared helpers across the processing pipeline's steps and web UI."""

from __future__ import annotations

__all__ = [
    "DEFAULT_DUCKDB_MEMORY_LIMIT",
    "DEFAULT_DUCKDB_THREADS",
    "connect_duckdb",
    "default_duckdb_memory_limit",
    "drop_timezones_for_excel",
]

import os
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
