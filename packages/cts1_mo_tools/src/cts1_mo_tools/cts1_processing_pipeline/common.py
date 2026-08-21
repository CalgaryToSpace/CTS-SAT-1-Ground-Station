"""Shared helpers across the processing pipeline's steps and web UI.

Currently just the one thing every DuckDB connection opener in this package
needs in common: a conservative default `memory_limit`, since DuckDB
otherwise defaults to ~80% of *host* RAM -- far more than this pipeline's
single-satellite data volume needs, and enough to crowd out everything else
on a small deployment box (see ../DEPLOY.md).
"""

from __future__ import annotations

__all__ = ["DEFAULT_DUCKDB_MEMORY_LIMIT", "connect_duckdb"]

from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from pathlib import Path

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
