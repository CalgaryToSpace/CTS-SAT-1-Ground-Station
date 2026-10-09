"""Step 1's tables in the shared landing-zone DuckDB database.

Two tables, both append/upsert-friendly and tolerant of new columns showing
up in later runs (our decoder wrappers are allowed to grow fields over
time):

  - raw_packets: one row per decoded frame/PDU (from any decoder),
    append-only.
  - decoder_runs: one row per (observation_id, decoder) that has been run,
    upserted by that pair (primary key), along with the decoder tool's
    version and how long the decode took (`runtime_ms`) at the time.
    Recorded unconditionally -- even when a decoder finds no packets -- so
    an observation/decoder pair with no output isn't retried on every
    subsequent run. `error` is a short (`ERROR_MAX_LENGTH`) summary of what
    went wrong (e.g. a 503 downloading the audio), or NULL if nothing did.

Step 1's input, `raw_observations`, belongs to step 0 (see
`step_0_list_observations.db`); `load_observations` reads it back.
"""

from __future__ import annotations

__all__ = [
    "DECODER_RUNS_TABLE",
    "ERROR_MAX_LENGTH",
    "QUALITY_TIER_ORDER",
    "RAW_PACKETS_TABLE",
    "already_decoded_pairs",
    "append_packets",
    "export_parquets",
    "format_counts",
    "load_observations",
    "record_decoder_runs",
]

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.landing_db import (
    add_missing_columns,
    export_table_to_parquet,
    insert_with_type_repair,
    quote_ident,
    table_exists,
)
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations.db import (
    RAW_OBSERVATIONS_TABLE,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import duckdb

RAW_PACKETS_TABLE = "raw_packets"
DECODER_RUNS_TABLE = "decoder_runs"
# Longest `decoder_runs.error` kept; anything longer is cut short with "...".
ERROR_MAX_LENGTH = 100

# Quality tiers, best first; unknown tiers sort after these, alphabetically.
# "crc_absent_assumed_good" is satnogs_client_live_data-only -- FEC already ran, but
# its packets arrive with the CSP CRC-32C trailer sometimes stripped, and an
# absent trailer can't be told from a wrong one.
QUALITY_TIER_ORDER = (
    "verified",
    "crc_absent_assumed_good",
    "rs_correctable_crc_fail",
    "believable",
    "candidate",
)


def format_counts(
    df: pl.DataFrame, col: str, *, order: Sequence[str] = ()
) -> str | None:
    """Render "a=2, b=1" counts of `col`'s values, or None if `col` is absent.

    Values listed in `order` come first, in that order; anything else
    (including nulls) follows alphabetically.
    """
    if col not in df.columns:
        return None

    ranks = {value: rank for rank, value in enumerate(order)}
    counts = sorted(
        (
            ("<none>" if value is None else str(value), count)
            for value, count in df[col].value_counts().rows()
        ),
        key=lambda pair: (ranks.get(pair[0], len(ranks)), pair[0]),
    )
    return ", ".join(f"{value}={count}" for value, count in counts)


def append_packets(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    """Append decoded-packet rows to raw_packets."""
    if df.is_empty():
        return

    con.register("_incoming_packets", df)
    try:
        if not table_exists(con, RAW_PACKETS_TABLE):
            con.execute(
                f"CREATE TABLE {quote_ident(RAW_PACKETS_TABLE)} AS "  # noqa: S608
                f"SELECT * FROM _incoming_packets"
            )
        else:
            add_missing_columns(con, RAW_PACKETS_TABLE, "_incoming_packets")
            insert_with_type_repair(con, RAW_PACKETS_TABLE, "_incoming_packets")
    finally:
        con.unregister("_incoming_packets")

    breakdowns = [
        format_counts(df, "decoder"),
        format_counts(df, "quality_tier", order=QUALITY_TIER_ORDER),
    ]
    suffix = "".join(f" ({part})" for part in breakdowns if part)
    logger.info(f"{RAW_PACKETS_TABLE}: appended {len(df)} row(s){suffix}")


def _shorten_error(error: str | None) -> str | None:
    """`error` on one line, cut to at most `ERROR_MAX_LENGTH` characters."""
    if error is None:
        return None
    error = " ".join(error.split())
    if len(error) <= ERROR_MAX_LENGTH:
        return error
    return error[: ERROR_MAX_LENGTH - 3] + "..."


def record_decoder_runs(
    con: duckdb.DuckDBPyConnection,
    observation_id: int,
    decoder_versions: Mapping[str, str | None],
    *,
    runtime_ms: int | None = None,
    errors: Mapping[str, str] | None = None,
) -> None:
    """Record that each decoder in `decoder_versions` has been run.

    `decoder_versions` maps decoder name to that decoder's tool version
    (e.g. the first line of `<tool> --version`), or None when the decoder
    has no versioned external tool.

    `runtime_ms` is how long (in milliseconds) the whole decode of this
    observation took -- all decoders in `decoder_versions` are dispatched
    together as a single unit of work, so the same value is stamped onto
    each of their rows.

    `errors` maps a decoder name to what went wrong running it, stored
    shortened to `ERROR_MAX_LENGTH` characters; a decoder absent from it
    gets a NULL `error`.

    Upserted by (observation_id, decoder): rerunning a pair (e.g. via
    --force-rerun-decoders) just bumps `run_at`/`version`/`runtime_ms`/
    `error` rather than adding a duplicate row.
    """
    run_at = datetime.now(UTC)
    errors = errors or {}
    rows = [
        {
            "observation_id": observation_id,
            "decoder": decoder,
            "run_at": run_at,
            "version": version,
            "runtime_ms": runtime_ms,
            "error": _shorten_error(errors.get(decoder)),
        }
        for decoder, version in decoder_versions.items()
    ]
    if not rows:
        return

    # Typed explicitly: a batch where nothing failed is all-NULL, which
    # polars would otherwise infer as its Null type.
    df = pl.DataFrame(rows, schema_overrides={"error": pl.String})
    con.register("_incoming_decoder_runs", df)
    try:
        if not table_exists(con, DECODER_RUNS_TABLE):
            con.execute(
                f"CREATE TABLE {quote_ident(DECODER_RUNS_TABLE)} ("
                f"observation_id BIGINT NOT NULL, "
                f"decoder VARCHAR NOT NULL, "
                f"run_at TIMESTAMPTZ NOT NULL, "
                f"version VARCHAR NOT NULL, "
                f"runtime_ms INTEGER NOT NULL, "
                f"error VARCHAR, "
                f"PRIMARY KEY (observation_id, decoder))"
            )
        else:
            add_missing_columns(con, DECODER_RUNS_TABLE, "_incoming_decoder_runs")
        con.execute(
            f"INSERT INTO {quote_ident(DECODER_RUNS_TABLE)} BY NAME "  # noqa: S608
            f"SELECT * FROM _incoming_decoder_runs "
            f"ON CONFLICT (observation_id, decoder) "
            f"DO UPDATE SET run_at = excluded.run_at, version = excluded.version, "
            f"runtime_ms = excluded.runtime_ms, error = excluded.error"
        )
    finally:
        con.unregister("_incoming_decoder_runs")


def export_parquets(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Copy raw_packets/decoder_runs out to Parquet files next to db_path."""
    out_dir = db_path.parent
    order_by = {
        RAW_PACKETS_TABLE: ["received_at", "observation_id"],
        DECODER_RUNS_TABLE: ["observation_id", "decoder"],
    }
    for table, order_by_columns in order_by.items():
        if table_exists(con, table):
            export_table_to_parquet(
                con,
                table=table,
                out_path=out_dir / f"{table}.parquet",
                order_by_columns=order_by_columns,
            )


def load_observations(
    con: duckdb.DuckDBPyConnection,
    *,
    norad_id: str,
    start_gte: datetime | None = None,
    start_lt: datetime | None = None,
) -> list[dict[str, Any]]:
    """Read step 0's raw_observations back as observation dicts, newest first.

    `start_gte`/`start_lt` bound which observations are read by their start
    time; None leaves that side open.

    Only the columns step 1 uses are read. `start`/`end` come back as
    tz-aware UTC datetimes, and `demoddata` as the list of dicts the
    SatNOGS API returned (step 0 stores it JSON-encoded). Empty if step 0
    hasn't landed anything yet.
    """
    if not table_exists(con, RAW_OBSERVATIONS_TABLE):
        logger.warning(f"{RAW_OBSERVATIONS_TABLE} not found -- run step_0 first.")
        return []

    sql = (
        f'SELECT id, start, "end", payload, demoddata '  # noqa: S608
        f"FROM {quote_ident(RAW_OBSERVATIONS_TABLE)} WHERE norad_cat_id = ?"
    )
    params: list[Any] = [int(norad_id)]
    if start_gte is not None:
        sql += " AND start >= ?"
        params.append(start_gte)
    if start_lt is not None:
        sql += " AND start < ?"
        params.append(start_lt)
    sql += " ORDER BY start DESC, id DESC"

    # Via polars rather than `fetchall()`: DuckDB needs pytz to hand back
    # TIMESTAMPTZ values as Python objects, while polars uses zoneinfo.
    observations_df = con.execute(sql, params).pl()
    return [
        {
            **obs,
            "start": obs["start"].astimezone(UTC),
            "end": obs["end"].astimezone(UTC),
            "demoddata": json.loads(obs["demoddata"]) if obs["demoddata"] else [],
        }
        for obs in observations_df.iter_rows(named=True)
    ]


def already_decoded_pairs(con: duckdb.DuckDBPyConnection) -> set[tuple[int, str]]:
    """Return {(observation_id, decoder)} already recorded in decoder_runs."""
    if not table_exists(con, DECODER_RUNS_TABLE):
        return set()
    rows = con.execute(
        f"SELECT observation_id, decoder "  # noqa: S608
        f"FROM {quote_ident(DECODER_RUNS_TABLE)}"
    ).fetchall()
    return {(obs_id, decoder) for obs_id, decoder in rows}
