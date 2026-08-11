"""DuckDB landing-zone tables for the processing pipeline.

Two tables, both append/upsert-friendly and tolerant of new columns showing
up in later runs (the SatNOGS API and our decoder wrappers are both allowed
to grow fields over time):

  - raw_observations: one row per SatNOGS observation, upserted by `id`.
  - raw_packets: one row per decoded frame/PDU (from either decoder),
    append-only.
"""

from __future__ import annotations

__all__ = [
    "RAW_OBSERVATIONS_TABLE",
    "RAW_PACKETS_TABLE",
    "already_decoded_pairs",
    "append_packets",
    "connect",
    "upsert_observations",
]

from typing import TYPE_CHECKING

import duckdb
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

RAW_OBSERVATIONS_TABLE = "raw_observations"
RAW_PACKETS_TABLE = "raw_packets"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    assert row is not None
    return bool(row[0] > 0)


def _add_missing_columns(
    con: duckdb.DuckDBPyConnection, table: str, incoming_view: str
) -> None:
    """Widen `table` with any columns present in incoming_view but not in it."""
    existing_cols = {
        row[1]
        for row in con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    }
    for col_name, col_type, *_ in con.execute(
        f"DESCRIBE {incoming_view}"
    ).fetchall():
        if col_name not in existing_cols:
            logger.info(f"{table}: adding new column {col_name!r} ({col_type})")
            con.execute(  # noqa: S608
                f"ALTER TABLE {_quote_ident(table)} "
                f"ADD COLUMN {_quote_ident(col_name)} {col_type}"
            )


def upsert_observations(
    con: duckdb.DuckDBPyConnection, df: pl.DataFrame, *, key_col: str = "id"
) -> None:
    """Insert/replace rows in raw_observations, keyed by `id`."""
    if df.is_empty():
        return

    con.register("_incoming_observations", df)
    try:
        if not _table_exists(con, RAW_OBSERVATIONS_TABLE):
            con.execute(  # noqa: S608
                f"CREATE TABLE {_quote_ident(RAW_OBSERVATIONS_TABLE)} AS "
                f"SELECT * FROM _incoming_observations"
            )
        else:
            _add_missing_columns(con, RAW_OBSERVATIONS_TABLE, "_incoming_observations")
            con.execute(  # noqa: S608
                f"DELETE FROM {_quote_ident(RAW_OBSERVATIONS_TABLE)} "
                f"WHERE {_quote_ident(key_col)} IN "
                f"(SELECT {_quote_ident(key_col)} FROM _incoming_observations)"
            )
            con.execute(  # noqa: S608
                f"INSERT INTO {_quote_ident(RAW_OBSERVATIONS_TABLE)} BY NAME "
                f"SELECT * FROM _incoming_observations"
            )
    finally:
        con.unregister("_incoming_observations")

    logger.info(f"{RAW_OBSERVATIONS_TABLE}: upserted {len(df)} row(s)")


def append_packets(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    """Append decoded-packet rows to raw_packets."""
    if df.is_empty():
        return

    con.register("_incoming_packets", df)
    try:
        if not _table_exists(con, RAW_PACKETS_TABLE):
            con.execute(  # noqa: S608
                f"CREATE TABLE {_quote_ident(RAW_PACKETS_TABLE)} AS "
                f"SELECT * FROM _incoming_packets"
            )
        else:
            _add_missing_columns(con, RAW_PACKETS_TABLE, "_incoming_packets")
            con.execute(  # noqa: S608
                f"INSERT INTO {_quote_ident(RAW_PACKETS_TABLE)} BY NAME "
                f"SELECT * FROM _incoming_packets"
            )
    finally:
        con.unregister("_incoming_packets")

    logger.info(f"{RAW_PACKETS_TABLE}: appended {len(df)} row(s)")


def already_decoded_pairs(con: duckdb.DuckDBPyConnection) -> set[tuple[int, str]]:
    """Return {(observation_id, decoder)} already present in raw_packets."""
    if not _table_exists(con, RAW_PACKETS_TABLE):
        return set()
    rows = con.execute(  # noqa: S608
        f"SELECT DISTINCT observation_id, decoder "
        f"FROM {_quote_ident(RAW_PACKETS_TABLE)}"
    ).fetchall()
    return {(obs_id, decoder) for obs_id, decoder in rows}
