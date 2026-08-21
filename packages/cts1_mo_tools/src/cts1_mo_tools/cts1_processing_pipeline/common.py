"""Shared helpers across the processing pipeline's steps and web UI."""

from __future__ import annotations

__all__ = [
    "DEFAULT_DUCKDB_MEMORY_LIMIT",
    "connect_duckdb",
    "drop_timezones_for_excel",
]

from typing import TYPE_CHECKING

import duckdb
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

# DuckDB otherwise defaults to ~80% of *host* RAM -- far more than this
# pipeline's single-satellite data volume needs, and enough to crowd out
# everything else on a small deployment box (see ../DEPLOY.md).
DEFAULT_DUCKDB_MEMORY_LIMIT = "500MB"


def connect_duckdb(
    database: str | Path = ":memory:",
    *,
    memory_limit: str = DEFAULT_DUCKDB_MEMORY_LIMIT,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with `memory_limit` capped up front."""
    con = duckdb.connect(database)
    con.execute("SET memory_limit = ?", [memory_limit])
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
