"""Loading, filtering, and exporting the pipeline's tables for the web UI's
Export Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars (plus
duckdb for the two file-based container formats -- sqlite, duckdb -- that
polars doesn't natively write to).

`TABLE_SPECS` enumerates every parquet file the pipeline produces, in the
order the pipeline produces them -- see `pipeline_status.py` for where each
filename comes from, and the individual step modules' docstrings for what
each table actually contains.
"""

from __future__ import annotations

__all__ = [
    "EXPORT_FORMATS",
    "MAX_OUTPUT_BYTES",
    "MAX_ROWS_EXCEL",
    "MAX_ROWS_OTHER",
    "PACKET_TYPES",
    "TABLE_SPECS",
    "ExportFormat",
    "ExportResult",
    "TableSpec",
    "export_selected_tables",
    "load_filtered_table",
]

import io
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never

import duckdb
import polars as pl
import xlsxwriter

from cts1_mo_tools.cts1_decode_satnogs_packets import PACKET_TYPE_MAP
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

from . import pipeline_status

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

# "UNKNOWN" is the decoder's own fallback name for a packet type byte it
# doesn't recognize -- see `cts1_decode_satnogs_packets.decode_packet_safe`.
PACKET_TYPES: tuple[str, ...] = (*sorted(PACKET_TYPE_MAP.values()), "UNKNOWN")


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One exportable table: where its parquet file lives, and how the
    Export Data page's filters apply to it.

    `time_column` is None for a table with no meaningful per-row timestamp
    (none currently, but a future addition might not have one) -- the time
    filter is silently skipped for it rather than erroring.
    """

    key: str
    step: str
    description: str
    filename: str
    time_column: str | None
    has_packet_type: bool


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        key="raw_observations",
        step="Step 1 -- Download & Demodulate",
        description=(
            "Every SatNOGS observation window fetched for the satellite "
            "(metadata only -- no packet content)."
        ),
        filename=pipeline_status.RAW_OBSERVATIONS_FILENAME,
        time_column="start",
        has_packet_type=False,
    ),
    TableSpec(
        key="raw_packets",
        step="Step 1 -- Download & Demodulate",
        description=(
            "One row per (decoder, packet) decode attempt straight off the "
            "downlinked audio -- the same physical transmission commonly "
            "appears many times over, once per decoder/ground station that "
            "caught it."
        ),
        filename=pipeline_status.RAW_PACKETS_FILENAME,
        time_column="received_at",
        has_packet_type=False,
    ),
    TableSpec(
        key="decoder_runs",
        step="Step 1 -- Download & Demodulate",
        description="One row per decoder tool invocation per observation -- run time, version, and duration.",
        filename=pipeline_status.DECODER_RUNS_FILENAME,
        time_column="run_at",
        has_packet_type=False,
    ),
    TableSpec(
        key="distinct_packets_over_time",
        step="Step 2 -- Deduplicate Packets",
        description=(
            "raw_packets collapsed to one row per distinct received "
            "packet, each with a single best-guess received_at and "
            "traceability back to every observation/decoder that "
            "contributed to it."
        ),
        filename=pipeline_status.DISTINCT_PACKETS_FILENAME,
        time_column="received_at",
        has_packet_type=False,
    ),
    TableSpec(
        key="everything_decoded",
        step="Step 3 -- Decode Packets",
        description=(
            "distinct_packets_over_time plus every field the CTS-SAT-1 "
            "packet decoder could pull out of each packet's payload."
        ),
        filename=step_3_pipeline.OUTPUT_FILENAME,
        time_column="received_at",
        has_packet_type=True,
    ),
)

ExportFormat = Literal["csv", "excel", "parquet", "sqlite", "duckdb"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("csv", "excel", "parquet", "sqlite", "duckdb")

_MEDIA_TYPES: dict[ExportFormat, str] = {
    "csv": "text/csv",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/octet-stream",
    "sqlite": "application/vnd.sqlite3",
    "duckdb": "application/octet-stream",
}

# Excel chokes/crawls well before its actual 1,048,576-row-per-sheet limit;
# the other formats are cheap to write at any size we're likely to see, but
# still get a cap so a "select everything, no filter" click can't spend
# minutes materializing hundreds of millions of raw_packets rows in the
# server process. `build_export` checks this against the *filtered* row
# count, so narrowing the time range/packet-type filter is the way past it
# -- the raw, unfiltered download (see `export_raw.py`) is the other way,
# since it never loads rows into polars at all.
MAX_ROWS_EXCEL = 50_000
MAX_ROWS_OTHER = 1_000_000

_ROW_LIMITS: dict[ExportFormat, int] = {
    "csv": MAX_ROWS_OTHER,
    "parquet": MAX_ROWS_OTHER,
    "excel": MAX_ROWS_EXCEL,
    "sqlite": MAX_ROWS_OTHER,
    "duckdb": MAX_ROWS_OTHER,
}

# Applies to the final bytes actually being sent back (post-zip, if
# zipping) -- a belt-and-suspenders check alongside the row limits above,
# since a handful of very wide/text-heavy rows can still add up to a large
# payload well under the row cap. The raw, unfiltered download (see
# `export_raw.py`) has no such limit, since it streams a file straight off
# disk rather than materializing one in the server process.
MAX_OUTPUT_BYTES = 100 * 1024 * 1024  # 100 MiB


def _drop_all_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop every column that's entirely null -- a no-op on an empty
    (0-row) `df`, since "all null" is vacuously true for every column
    there and dropping them would just leave an empty table with no
    columns at all, which isn't useful to anyone.
    """
    if df.is_empty():
        return df
    null_counts = df.null_count().row(0, named=True)
    keep_columns = [name for name, count in null_counts.items() if count < df.height]
    return df.select(keep_columns)


def load_filtered_table(
    spec: TableSpec,
    output_dir: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    packet_types: Sequence[str] | None,
    drop_all_null_columns: bool,
) -> pl.DataFrame:
    """Load `spec`'s parquet file (if the pipeline has produced it yet),
    with the time range and packet-type filters pushed down into the scan.

    `packet_types` is ignored for a table where `spec.has_packet_type` is
    False -- there's nothing to filter on. Returns an empty DataFrame (no
    rows, no columns) if the file doesn't exist yet.
    """
    path = output_dir / spec.filename
    if not path.exists():
        return pl.DataFrame()

    lf = pl.scan_parquet(path)
    if spec.time_column is not None:
        if start is not None:
            lf = lf.filter(pl.col(spec.time_column) >= start)
        if end is not None:
            lf = lf.filter(pl.col(spec.time_column) <= end)
    if spec.has_packet_type and packet_types:
        lf = lf.filter(pl.col("packet_type").is_in(list(packet_types)))

    df = lf.collect()
    if drop_all_null_columns:
        df = _drop_all_null_columns(df)
    return df


@dataclass(frozen=True, slots=True)
class ExportResult:
    filename: str
    data: bytes
    media_type: str


_EXCEL_SHEET_NAME_INVALID_CHARS = re.compile(r"[\[\]:*?/\\]")
_EXCEL_MAX_SHEET_NAME_LEN = 31


def _excel_sheet_name(key: str) -> str:
    return _EXCEL_SHEET_NAME_INVALID_CHARS.sub("_", key)[:_EXCEL_MAX_SHEET_NAME_LEN]


def _write_csv_files(tables: Mapping[TableSpec, pl.DataFrame]) -> dict[str, bytes]:
    return {f"{spec.key}.csv": df.write_csv().encode("utf-8") for spec, df in tables.items()}


def _write_parquet_files(tables: Mapping[TableSpec, pl.DataFrame]) -> dict[str, bytes]:
    files = {}
    for spec, df in tables.items():
        buf = io.BytesIO()
        df.write_parquet(buf)
        files[f"{spec.key}.parquet"] = buf.getvalue()
    return files


def _drop_timezones(df: pl.DataFrame) -> pl.DataFrame:
    """Excel has no timezone-aware datetime type -- normalize every
    tz-aware Datetime column to naive UTC before handing it to
    `write_excel`, or xlsxwriter raises trying to format the cell.
    """
    exprs = [
        pl.col(name).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
    ]
    return df.with_columns(exprs) if exprs else df


def _write_excel_file(tables: Mapping[TableSpec, pl.DataFrame]) -> dict[str, bytes]:
    buf = io.BytesIO()
    workbook = xlsxwriter.Workbook(buf, {"in_memory": True})
    for spec, df in tables.items():
        df = _drop_timezones(df)
        df.write_excel(workbook=workbook, worksheet=_excel_sheet_name(spec.key))
    workbook.close()
    return {"export.xlsx": buf.getvalue()}


def _write_sqlite_file(tables: Mapping[TableSpec, pl.DataFrame]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "export.sqlite"
        con = duckdb.connect(":memory:")
        try:
            con.execute("INSTALL sqlite")
            con.execute("LOAD sqlite")
            con.execute(f"ATTACH '{db_path}' AS sq (TYPE SQLITE)")
            for spec, df in tables.items():
                view_name = f"_export_{spec.key}"
                con.register(view_name, df)
                con.execute(f'CREATE TABLE sq."{spec.key}" AS SELECT * FROM {view_name}')
            con.execute("DETACH sq")
        finally:
            con.close()
        return {"export.sqlite": db_path.read_bytes()}


def _write_duckdb_file(tables: Mapping[TableSpec, pl.DataFrame]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "export.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            for spec, df in tables.items():
                view_name = f"_export_{spec.key}"
                con.register(view_name, df)
                con.execute(f'CREATE TABLE "{spec.key}" AS SELECT * FROM {view_name}')
        finally:
            con.close()
        return {"export.duckdb": db_path.read_bytes()}


def _zip_files(files: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in files.items():
            zf.writestr(filename, data)
    return buf.getvalue()


def build_export(
    tables: Mapping[TableSpec, pl.DataFrame],
    *,
    export_format: ExportFormat,
    zip_output: bool,
) -> ExportResult:
    """Write `tables` out as `export_format`, and package the result.

    csv/parquet produce one file per table; excel/sqlite/duckdb are
    multi-table containers, so every table lands in a single file
    regardless. `zip_output` zips whatever comes out of that into one
    `export.zip`.

    Raises:
        ValueError: if `tables` is empty, if the combined row count exceeds
            `MAX_ROWS_EXCEL`/`MAX_ROWS_OTHER` for `export_format`, if the
            resulting file(s) exceed `MAX_OUTPUT_BYTES`, or if csv/parquet
            produced more than one file and `zip_output` is False -- there's
            no way to hand back more than one file from a single download
            otherwise.
    """
    if not tables:
        msg = "No tables selected."
        raise ValueError(msg)

    total_rows = sum(df.height for df in tables.values())
    row_limit = _ROW_LIMITS[export_format]
    if total_rows > row_limit:
        msg = (
            f"{total_rows:,} selected rows exceeds the {row_limit:,}-row "
            f"limit for {export_format} exports -- narrow the time range "
            "or packet-type filter, or use the raw unfiltered download "
            "instead (see export_raw.py), which never loads rows into "
            "polars at all."
        )
        raise ValueError(msg)

    if export_format == "csv":
        files = _write_csv_files(tables)
    elif export_format == "parquet":
        files = _write_parquet_files(tables)
    elif export_format == "excel":
        files = _write_excel_file(tables)
    elif export_format == "sqlite":
        files = _write_sqlite_file(tables)
    elif export_format == "duckdb":
        files = _write_duckdb_file(tables)
    else:
        assert_never(export_format)

    if len(files) > 1 and not zip_output:
        msg = (
            f"Exporting {len(files)} tables as {export_format} produces one "
            'file per table -- check "Zip output" to download them together.'
        )
        raise ValueError(msg)

    if zip_output:
        data = _zip_files(files)
        if len(data) > MAX_OUTPUT_BYTES:
            msg = (
                f"Even zipped, the export is {len(data):,} bytes, over the "
                f"{MAX_OUTPUT_BYTES:,}-byte limit -- narrow the time range "
                "or packet-type filter."
            )
            raise ValueError(msg)
        return ExportResult(filename="export.zip", data=data, media_type="application/zip")

    ((filename, data),) = files.items()
    if len(data) > MAX_OUTPUT_BYTES:
        msg = (
            f"The export is {len(data):,} bytes, over the "
            f'{MAX_OUTPUT_BYTES:,}-byte limit -- check "Zip output" to '
            "compress it, or narrow the time range/packet-type filter."
        )
        raise ValueError(msg)
    return ExportResult(filename=filename, data=data, media_type=_MEDIA_TYPES[export_format])


def export_selected_tables(
    specs: Sequence[TableSpec],
    output_dir: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    packet_types: Sequence[str] | None,
    drop_all_null_columns: bool,
    export_format: ExportFormat,
    zip_output: bool,
) -> ExportResult:
    """Load + filter every table in `specs`, then package them per
    `export_format` -- see `load_filtered_table` and `build_export`.
    """
    tables = {
        spec: load_filtered_table(
            spec,
            output_dir,
            start=start,
            end=end,
            packet_types=packet_types,
            drop_all_null_columns=drop_all_null_columns,
        )
        for spec in specs
    }
    return build_export(tables, export_format=export_format, zip_output=zip_output)
