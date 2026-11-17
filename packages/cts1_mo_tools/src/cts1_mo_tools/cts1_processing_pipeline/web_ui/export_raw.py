"""Raw, unfiltered parquet downloads for the Export Data page's per-table
"direct parquet download" links.

A plain FastAPI route registered directly on NiceGUI's underlying app (not
a `ui.page`), since the whole point is to hand the ASGI server's own
file-streaming straight to the client -- `FileResponse` streams a table's
parquet file from disk exactly as it sits in `output/`, with no polars
load/filter/re-encode step in between and no zipping. That makes it the
cheapest possible way to get a whole table out, at the cost of not
supporting any of the filtering `export_tables.export_selected_tables` does
(or downloading more than one table per request). A month-partitioned table
(see `partitions`) is the one exception: its months are first streamed into
a single temp file, which is served and then deleted -- so DuckDB, which
reads remote parquet with many small range requests, is pointed at each
month's file (`{RAW_EXPORT_PATH}/{table_key}/{month}.parquet`, served
straight off disk) instead; see `raw_export_urls`.
"""

__all__ = [
    "RAW_EXPORT_PATH",
    "raw_export_url",
    "raw_export_urls",
    "register_raw_export_route",
]

import tempfile
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from cts1_mo_tools.cts1_processing_pipeline import partitions

from . import export_tables

RAW_EXPORT_PATH = "/raw-export"

_SPECS_BY_KEY = {spec.key: spec for spec in export_tables.TABLE_SPECS}


def raw_export_url(table_key: str) -> str:
    return f"{RAW_EXPORT_PATH}/{table_key}.parquet"


def raw_export_urls(table_key: str, data_dir: Path) -> list[str]:
    """The URL of each of a table's month files if it's month-partitioned,
    else just its single-file URL."""
    path = data_dir / _SPECS_BY_KEY[table_key].filename
    return [
        f"{RAW_EXPORT_PATH}/{table_key}/{month}.parquet"
        for month in partitions.month_files(path)
    ] or [raw_export_url(table_key)]


def register_raw_export_route(app: FastAPI, data_dir: Path) -> None:
    """Register `GET`/`HEAD {RAW_EXPORT_PATH}/{table_key}.parquet` on `app`, streaming
    that table's parquet file straight off disk via `FileResponse`.
    """

    def raw_export(table_key: str) -> FileResponse:
        spec = _SPECS_BY_KEY.get(table_key)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown table: {table_key!r}")

        path = data_dir / spec.filename
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"{table_key} hasn't been produced by the pipeline yet.",
            )

        if not path.is_dir():
            return FileResponse(
                path, filename=spec.filename, media_type="application/octet-stream"
            )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            merged = Path(tmp.name)
        pl.scan_parquet(path).sink_parquet(merged)
        return FileResponse(
            merged,
            filename=spec.filename,
            media_type="application/octet-stream",
            background=BackgroundTask(merged.unlink, missing_ok=True),
        )

    def raw_export_month(table_key: str, month: str) -> FileResponse:
        spec = _SPECS_BY_KEY.get(table_key)
        # Looked up among the files actually there, so `month` can't escape the dir.
        file = spec and partitions.month_files(data_dir / spec.filename).get(month)
        if not file:
            raise HTTPException(
                status_code=404, detail=f"No {month!r} partition of {table_key!r}"
            )
        return FileResponse(
            file,
            filename=f"{table_key}_{month}.parquet",
            media_type="application/octet-stream",
        )

    # HEAD so DuckDB's httpfs can size the file up front instead of logging a 405.
    # Registered via `add_api_route` rather than the `@app.get(...)` decorator
    # form so `raw_export` is passed as an argument (visibly "used") instead
    # of a decorated-but-never-called nested function, which pyright's
    # `reportUnusedFunction` otherwise flags.
    app.add_api_route(
        RAW_EXPORT_PATH + "/{table_key}.parquet", raw_export, methods=["GET", "HEAD"]
    )
    app.add_api_route(
        RAW_EXPORT_PATH + "/{table_key}/{month}.parquet",
        raw_export_month,
        methods=["GET", "HEAD"],
    )
