"""The DuckDB "landing zone" database shared by steps 0 and 1.

Step 0 (list observations) and step 1 (download and demodulate) both write
into one DuckDB file, `DB_FILENAME`, inside `data_dir`: step 0 lands the
SatNOGS observation listing there, and step 1 reads that listing back out to
decide what to decode. Each step owns its own tables (see each step's
`db` module) and exports them to parquet files alongside the database, which
is all that later steps and the web UI ever read.

This module holds the table-agnostic plumbing both steps' `db` modules
share: opening the database, and the schema-tolerant insert/export helpers
-- the SatNOGS API and our decoder wrappers are both allowed to grow fields
over time, so every landing table widens itself to fit whatever shows up.
"""

from __future__ import annotations

__all__ = [
    "DB_FILENAME",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DB_PATH",
    "add_missing_columns",
    "connect",
    "export_table_to_parquet",
    "insert_with_type_repair",
    "quote_ident",
    "quote_literal",
    "table_exists",
]

import os
from pathlib import Path

import duckdb
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.common import (
    DEFAULT_DUCKDB_MEMORY_LIMIT,
    connect_duckdb,
)

# Every step (and the web UI) takes a single `data_dir` argument and finds
# its own file(s) inside it by a fixed filename -- see each step's
# `DEFAULT_DATA_DIR`/`OUTPUT_FILENAME` -- so pointing every process (the
# daemon container and the web UI container alike) at the same directory is
# one env var, `CTS1_DATA_DIR`, rather than a `--db-path`/`--parquet-path`
# kept in sync by hand across both.
DEFAULT_DATA_DIR = Path(os.environ.get("CTS1_DATA_DIR", "output"))
DB_FILENAME = "cts1_processing_pipeline.duckdb"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / DB_FILENAME


def connect(
    db_path: Path, *, memory_limit: str = DEFAULT_DUCKDB_MEMORY_LIMIT
) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return connect_duckdb(db_path, memory_limit=memory_limit)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    assert row is not None
    return bool(row[0] > 0)


def add_missing_columns(
    con: duckdb.DuckDBPyConnection, table: str, incoming_view: str
) -> None:
    """Widen `table` with any columns present in incoming_view but not in it."""
    existing_cols = {
        row[1]
        for row in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    }
    for col_name, col_type, *_ in con.execute(f"DESCRIBE {incoming_view}").fetchall():
        if col_name not in existing_cols:
            logger.info(f"{table}: adding new column {col_name!r} ({col_type})")
            con.execute(
                f"ALTER TABLE {quote_ident(table)} "
                f"ADD COLUMN {quote_ident(col_name)} {col_type}"
            )


def _alter_column_type(
    con: duckdb.DuckDBPyConnection, table: str, col_name: str, new_type: str
) -> bool:
    """Try ALTER COLUMN ... TYPE; return whether it succeeded."""
    try:
        con.execute(
            f"ALTER TABLE {quote_ident(table)} "
            f"ALTER COLUMN {quote_ident(col_name)} TYPE {new_type}"
        )
    except duckdb.ConversionException:
        return False
    return True


def _widen_mismatched_columns(
    con: duckdb.DuckDBPyConnection, table: str, incoming_view: str
) -> None:
    """Widen any existing column that can't hold the incoming batch's type.

    Small per-page/per-observation batches routinely have columns that are
    entirely NULL (e.g. `payload` is null for most observations); DuckDB
    infers those as a narrow type (often INTEGER) at CREATE TABLE time, which
    then fails to hold a later batch's real data of a different type.

    DuckDB's own ALTER COLUMN ... TYPE already knows how to promote sensibly
    (INTEGER -> BIGINT -> DOUBLE, DATE -> TIMESTAMP, etc.) and simply raises
    if the existing column's data can't be represented in the new type, so
    the incoming batch's type is tried first and VARCHAR is only the
    fallback when that specific promotion isn't representable.
    """
    existing_types = {
        row[1]: row[2]
        for row in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    }
    for col_name, col_type, *_ in con.execute(f"DESCRIBE {incoming_view}").fetchall():
        existing_type = existing_types.get(col_name)
        if existing_type is None or existing_type in (col_type, "VARCHAR"):
            continue

        if _alter_column_type(con, table, col_name, col_type):
            target_type = col_type
        else:
            _alter_column_type(con, table, col_name, "VARCHAR")
            target_type = "VARCHAR"
        logger.warning(
            f"{table}: widened column {col_name!r} from {existing_type} to "
            f"{target_type} (incoming batch has type {col_type})"
        )


def insert_with_type_repair(
    con: duckdb.DuckDBPyConnection, table: str, incoming_view: str
) -> None:
    """INSERT INTO ... BY NAME, self-healing on a column-type conflict.

    Retries exactly once after widening the offending column(s) to VARCHAR.
    """
    insert_sql = (
        f"INSERT INTO {quote_ident(table)} BY NAME SELECT * FROM {incoming_view}"  # noqa: S608
    )
    try:
        con.execute(insert_sql)
    except duckdb.ConversionException:
        logger.warning(
            f"{table}: column type conflict inserting incoming batch; "
            f"widening and retrying"
        )
        _widen_mismatched_columns(con, table, incoming_view)
        con.execute(insert_sql)


def _copy_to_parquet(
    con: duckdb.DuckDBPyConnection, select_sql: str, out_path: Path
) -> None:
    con.execute(
        f"COPY ({select_sql}) TO {quote_literal(str(out_path))} (FORMAT PARQUET)"
    )


def _copy_to_parquet_in_sorted_chunks(
    con: duckdb.DuckDBPyConnection,
    table: str,
    out_path: Path,
    *,
    order_by_str: str,
    chunk_rows: int,
) -> None:
    """The chunked half of `export_table_to_parquet` -- see its docstring.

    Each row's position in the sorted order is computed once, from just the
    sort keys and `rowid`, into a temp table; each chunk then joins its
    slice of positions back to the full rows and sorts only those. The
    chunks are UNION ALL'd into one COPY, which (with DuckDB's default
    `preserve_insertion_order`) writes them out in order.
    """
    keys_table = quote_ident(f"_sort_keys_{table}")
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {keys_table} AS "  # noqa: S608
        f"SELECT rowid AS _rid, row_number() OVER (ORDER BY {order_by_str}) - 1 "
        f"AS _pos FROM {quote_ident(table)}"
    )
    try:
        row = con.execute(f"SELECT count(*) FROM {keys_table}").fetchone()  # noqa: S608
        assert row is not None
        total_rows = int(row[0])
        chunks = [
            f"(SELECT t.* FROM {quote_ident(table)} AS t "  # noqa: S608
            f"JOIN {keys_table} AS k ON t.rowid = k._rid "
            f"WHERE k._pos >= {lo} AND k._pos < {lo + chunk_rows} "
            f"ORDER BY k._pos)"
            # At least one chunk, even if empty, so the COPY still has a
            # query (and so writes the table's schema).
            for lo in range(0, max(total_rows, 1), chunk_rows)
        ]
        _copy_to_parquet(con, " UNION ALL ".join(chunks), out_path)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {keys_table}")


def export_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    table: str,
    out_path: Path,
    *,
    order_by_columns: list[str],
    sort_chunk_rows: int | None = None,
) -> None:
    """Write `table` to `out_path`, sorted, via a temp file + rename.

    Sorting keeps the parquet files stable to diff and cheap to range-scan
    downstream; the rename means a reader (the web UI, a later step) never
    catches a half-written file.

    `sort_chunk_rows` is for tables whose rows are too wide for DuckDB to
    sort whole within its memory limit (`raw_observations`, whose
    `demoddata` runs to hundreds of KB per row): a plain `ORDER BY` there
    runs out of memory rather than spilling. With it set, only the narrow
    sort keys are sorted up front, and the full rows are then written in
    sorted chunks of that many rows, so peak memory scales with the chunk
    rather than the table. Each chunk rescans the table, so leave it None
    for narrow tables, which sort fine in one go.
    """
    order_by_str = ", ".join([f"{quote_ident(col)} ASC" for col in order_by_columns])
    select_sql = f"SELECT * FROM {quote_ident(table)} ORDER BY {order_by_str}"  # noqa: S608

    write_out_path = out_path.with_suffix(".tmp")
    if sort_chunk_rows is None:
        _copy_to_parquet(con, select_sql, write_out_path)
    else:
        _copy_to_parquet_in_sorted_chunks(
            con,
            table,
            write_out_path,
            order_by_str=order_by_str,
            chunk_rows=sort_chunk_rows,
        )

    write_out_path.replace(out_path)

    logger.info(f"{table}: exported to {out_path}")
