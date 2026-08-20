"""Raw, unfiltered parquet downloads for the Export Data page's per-table
"direct parquet download" links.

A plain FastAPI route registered directly on NiceGUI's underlying app (not
a `ui.page`), since the whole point is to hand the ASGI server's own
file-streaming straight to the client -- `FileResponse` streams a table's
parquet file from disk exactly as it sits in `output/`, with no polars
load/filter/re-encode step in between and no zipping. That makes it the
cheapest possible way to get a whole table out, at the cost of not
supporting any of the filtering `export_tables.export_selected_tables` does
(or downloading more than one table per request).
"""

from __future__ import annotations

__all__ = ["RAW_EXPORT_PATH", "raw_export_url", "register_raw_export_route"]

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import FileResponse

from . import export_tables

if TYPE_CHECKING:
    from fastapi import FastAPI

RAW_EXPORT_PATH = "/raw-export"

_SPECS_BY_KEY = {spec.key: spec for spec in export_tables.TABLE_SPECS}


def raw_export_url(table_key: str) -> str:
    return f"{RAW_EXPORT_PATH}/{table_key}"


def register_raw_export_route(app: FastAPI, output_dir: Path) -> None:
    """Register `GET {RAW_EXPORT_PATH}/{table_key}` on `app`, streaming
    that table's parquet file straight off disk via `FileResponse`.
    """

    @app.get(RAW_EXPORT_PATH + "/{table_key}")
    def raw_export(table_key: str) -> FileResponse:
        spec = _SPECS_BY_KEY.get(table_key)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown table: {table_key!r}")

        path = output_dir / spec.filename
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"{table_key} hasn't been produced by the pipeline yet.",
            )

        return FileResponse(
            path, filename=spec.filename, media_type="application/octet-stream"
        )
