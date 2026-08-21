"""DuckDB landing-zone tables for the processing pipeline.

Three tables, all append/upsert-friendly and tolerant of new columns showing
up in later runs (the SatNOGS API and our decoder wrappers are both allowed
to grow fields over time):

  - raw_observations: one row per SatNOGS observation, upserted by `id`.
  - raw_packets: one row per decoded frame/PDU (from either decoder),
    append-only.
  - decoder_runs: one row per (observation_id, decoder) that has been run,
    upserted by that pair (primary key), along with the decoder tool's
    version and how long the decode took (`runtime_ms`) at the time.
    Recorded unconditionally -- even when a decoder finds no packets -- so
    an observation/decoder pair with no output isn't retried on every
    subsequent run.
"""

from __future__ import annotations

__all__ = [
    "DECODER_RUNS_TABLE",
    "RAW_OBSERVATIONS_TABLE",
    "RAW_PACKETS_TABLE",
    "already_decoded_pairs",
    "append_packets",
    "connect",
    "export_parquets",
    "record_decoder_runs",
    "upsert_observations",
]

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RAW_OBSERVATIONS_TABLE = "raw_observations"
RAW_PACKETS_TABLE = "raw_packets"
DECODER_RUNS_TABLE = "decoder_runs"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
    for col_name, col_type, *_ in con.execute(f"DESCRIBE {incoming_view}").fetchall():
        if col_name not in existing_cols:
            logger.info(f"{table}: adding new column {col_name!r} ({col_type})")
            con.execute(
                f"ALTER TABLE {_quote_ident(table)} "
                f"ADD COLUMN {_quote_ident(col_name)} {col_type}"
            )


def _alter_column_type(
    con: duckdb.DuckDBPyConnection, table: str, col_name: str, new_type: str
) -> bool:
    """Try ALTER COLUMN ... TYPE; return whether it succeeded."""
    try:
        con.execute(
            f"ALTER TABLE {_quote_ident(table)} "
            f"ALTER COLUMN {_quote_ident(col_name)} TYPE {new_type}"
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
        for row in con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
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


def _insert_with_type_repair(
    con: duckdb.DuckDBPyConnection, table: str, incoming_view: str
) -> None:
    """INSERT INTO ... BY NAME, self-healing on a column-type conflict.

    Retries exactly once after widening the offending column(s) to VARCHAR.
    """
    insert_sql = (
        f"INSERT INTO {_quote_ident(table)} BY NAME SELECT * FROM {incoming_view}"  # noqa: S608
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


def upsert_observations(
    con: duckdb.DuckDBPyConnection, df: pl.DataFrame, *, key_col: str = "id"
) -> None:
    """Insert/replace rows in raw_observations, keyed by `id`."""
    if df.is_empty():
        return

    con.register("_incoming_observations", df)
    try:
        if not _table_exists(con, RAW_OBSERVATIONS_TABLE):
            con.execute(
                f"CREATE TABLE {_quote_ident(RAW_OBSERVATIONS_TABLE)} AS "  # noqa: S608
                f"SELECT * FROM _incoming_observations"
            )
        else:
            _add_missing_columns(con, RAW_OBSERVATIONS_TABLE, "_incoming_observations")
            con.execute(
                f"DELETE FROM {_quote_ident(RAW_OBSERVATIONS_TABLE)} "  # noqa: S608
                f"WHERE {_quote_ident(key_col)} IN "
                f"(SELECT {_quote_ident(key_col)} FROM _incoming_observations)"
            )
            _insert_with_type_repair(
                con, RAW_OBSERVATIONS_TABLE, "_incoming_observations"
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
            con.execute(
                f"CREATE TABLE {_quote_ident(RAW_PACKETS_TABLE)} AS "  # noqa: S608
                f"SELECT * FROM _incoming_packets"
            )
        else:
            _add_missing_columns(con, RAW_PACKETS_TABLE, "_incoming_packets")
            _insert_with_type_repair(con, RAW_PACKETS_TABLE, "_incoming_packets")
    finally:
        con.unregister("_incoming_packets")

    if "decoder" in df.columns:
        by_decoder = ", ".join(
            f"{decoder}={count}"
            for decoder, count in df["decoder"].value_counts().sort("decoder").rows()
        )
        logger.info(f"{RAW_PACKETS_TABLE}: appended {len(df)} row(s) ({by_decoder})")
    else:
        logger.info(f"{RAW_PACKETS_TABLE}: appended {len(df)} row(s)")


def record_decoder_runs(
    con: duckdb.DuckDBPyConnection,
    observation_id: int,
    decoder_versions: Mapping[str, str | None],
    *,
    runtime_ms: int | None = None,
) -> None:
    """Record that each decoder in `decoder_versions` has been run.

    `decoder_versions` maps decoder name to that decoder's tool version
    (e.g. the first line of `<tool> --version`), or None when the decoder
    has no versioned external tool.

    `runtime_ms` is how long (in milliseconds) the whole decode of this
    observation took -- all decoders in `decoder_versions` are dispatched
    together as a single unit of work, so the same value is stamped onto
    each of their rows.

    Upserted by (observation_id, decoder): rerunning a pair (e.g. via
    --force-rerun-decoders) just bumps `run_at`/`version`/`runtime_ms`
    rather than adding a duplicate row.
    """
    run_at = datetime.now(UTC)
    rows = [
        {
            "observation_id": observation_id,
            "decoder": decoder,
            "run_at": run_at,
            "version": version,
            "runtime_ms": runtime_ms,
        }
        for decoder, version in decoder_versions.items()
    ]
    if not rows:
        return

    df = pl.DataFrame(rows)
    con.register("_incoming_decoder_runs", df)
    try:
        if not _table_exists(con, DECODER_RUNS_TABLE):
            con.execute(
                f"CREATE TABLE {_quote_ident(DECODER_RUNS_TABLE)} ("
                f"observation_id BIGINT NOT NULL, "
                f"decoder VARCHAR NOT NULL, "
                f"run_at TIMESTAMPTZ NOT NULL, "
                f"version VARCHAR NOT NULL, "
                f"runtime_ms INTEGER NOT NULL, "
                f"PRIMARY KEY (observation_id, decoder))"
            )
        else:
            _add_missing_columns(con, DECODER_RUNS_TABLE, "_incoming_decoder_runs")
        con.execute(
            f"INSERT INTO {_quote_ident(DECODER_RUNS_TABLE)} BY NAME "  # noqa: S608
            f"SELECT * FROM _incoming_decoder_runs "
            f"ON CONFLICT (observation_id, decoder) "
            f"DO UPDATE SET run_at = excluded.run_at, version = excluded.version, "
            f"runtime_ms = excluded.runtime_ms"
        )
    finally:
        con.unregister("_incoming_decoder_runs")


def export_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    table: str,
    out_path: Path,
    *,
    order_by_columns: list[str],
) -> None:
    order_by_str = ", ".join([f"{_quote_ident(col)} ASC" for col in order_by_columns])

    write_out_path = out_path.with_suffix(".tmp")
    con.execute(
        f"""
            COPY (
                SELECT * FROM {_quote_ident(table)}
                ORDER BY {order_by_str}
            )
            TO {_quote_literal(str(write_out_path))} (FORMAT PARQUET)
        """  # noqa: S608
    )

    write_out_path.replace(out_path)

    logger.info(f"{table}: exported to {out_path}")


def export_parquets(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Copy raw_observations/raw_packets out to Parquet files next to db_path.

    Each table is sorted before writing (raw_observations by `id`,
    raw_packets by `observation_id`/`received_at`) so the Parquet files are
    stable to diff and cheap to range-scan downstream.
    """
    out_dir = db_path.parent

    if _table_exists(con, RAW_OBSERVATIONS_TABLE):
        out_path = out_dir / f"{RAW_OBSERVATIONS_TABLE}.parquet"
        export_table_to_parquet(
            con,
            table=RAW_OBSERVATIONS_TABLE,
            out_path=out_path,
            order_by_columns=["id"],
        )
        logger.info(f"{RAW_OBSERVATIONS_TABLE}: exported to {out_path}")

    if _table_exists(con, RAW_PACKETS_TABLE):
        out_path = out_dir / f"{RAW_PACKETS_TABLE}.parquet"
        export_table_to_parquet(
            con,
            table=RAW_PACKETS_TABLE,
            out_path=out_path,
            order_by_columns=["received_at", "observation_id"],
        )
        logger.info(f"{RAW_PACKETS_TABLE}: exported to {out_path}")

    if _table_exists(con, DECODER_RUNS_TABLE):
        out_path = out_dir / f"{DECODER_RUNS_TABLE}.parquet"
        export_table_to_parquet(
            con,
            table=DECODER_RUNS_TABLE,
            out_path=out_path,
            order_by_columns=["observation_id", "decoder"],
        )
        logger.info(f"{DECODER_RUNS_TABLE}: exported to {out_path}")


def already_decoded_pairs(con: duckdb.DuckDBPyConnection) -> set[tuple[int, str]]:
    """Return {(observation_id, decoder)} already recorded in decoder_runs."""
    if not _table_exists(con, DECODER_RUNS_TABLE):
        return set()
    rows = con.execute(
        f"SELECT observation_id, decoder "  # noqa: S608
        f"FROM {_quote_ident(DECODER_RUNS_TABLE)}"
    ).fetchall()
    return {(obs_id, decoder) for obs_id, decoder in rows}
