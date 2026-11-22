"""Loading step 11's `downlinked_files` -- every file on the satellite, and
what's been reassembled of it from bulk downlinks -- and step 10's mentions,
for the web UI's Satellite Files and Image Gallery pages; and serving a
reassembled file's bytes and preview image over HTTP.

The bytes are served by plain FastAPI routes (`register_downlinked_file_routes`)
reading straight from `downlinked_files.parquet`, rather than pushed over the
page's websocket: a browser then lazy-loads the gallery's images by URL,
downloads a file under its own name, and nothing has to be cached in memory
in between (compare `preview_image_route`, whose images exist only in
memory).

Data layer only: no NiceGUI here -- see `downlinked_files_page`.
"""

from __future__ import annotations

__all__ = [
    "FILE_DOWNLOAD_PATH",
    "FILE_PREVIEW_PATH",
    "GALLERY_SORTS",
    "DownlinkedFile",
    "download_filename",
    "file_download_url",
    "file_preview_url",
    "load_downlinked_file",
    "load_downlinked_files",
    "load_file_downlink_time_periods",
    "load_file_mentions",
    "load_gallery_images",
    "register_downlinked_file_routes",
    "sort_gallery_images",
    "status_text",
]

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

import polars as pl
from fastapi import FastAPI, HTTPException, Response

from cts1_mo_tools.cts1_processing_pipeline.file_reassembly import (
    DEFAULT_CONFLICT_POLICY,
    ReassemblyResult,
)
from cts1_mo_tools.cts1_processing_pipeline.step_10_list_satellite_files import (
    pipeline as step_10_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_11_reassemble_downlinked_files import (
    pipeline as step_11_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

FILE_DOWNLOAD_PATH = "/satellite-files/download"
FILE_PREVIEW_PATH = "/satellite-files/preview"

# The (big) columns the tables never need.
_BULKY_COLUMNS = ("data", "preview_image", "segments")

PARTIAL_FILENAME_SUFFIX = "_partial"

GALLERY_SORTS = {
    "Last downlinked": ("last_downlinked_at", "file_path"),
    "File timestamp": ("file_timestamp", "_name_key", "file_path"),
    "File name": ("_name_key", "file_path"),
    "File path": ("_path_key", "file_path"),
}


def _files_path(data_dir: Path) -> Path:
    return data_dir / step_11_pipeline.DOWNLINKED_FILES_OUTPUT_FILENAME


def load_downlinked_files(data_dir: Path) -> pl.DataFrame:
    """Every file -- reassembled, or known of but never downlinked --
    newest downlink first, without its bytes; empty if step 11 hasn't run
    yet.
    """
    path = _files_path(data_dir)
    if not path.exists():
        return pl.DataFrame()
    files = pl.scan_parquet(path).drop(_BULKY_COLUMNS).collect()
    if "file_timestamp" not in files.columns:  # until step 11 next runs after a deploy
        files = files.with_columns(
            file_timestamp=pl.lit(None, pl.Datetime("us", "UTC"))
        )
    return files


def load_gallery_images(data_dir: Path) -> pl.DataFrame:
    """`load_downlinked_files`' rows that have a preview image."""
    files = load_downlinked_files(data_dir)
    if files.is_empty():
        return files
    return files.filter(pl.col("preview_media_type").is_not_null())


def sort_gallery_images(
    images: pl.DataFrame, order: str, *, descending: bool
) -> pl.DataFrame:
    """`load_gallery_images`' rows in one of the `GALLERY_SORTS`. Undated
    images go last in either direction.
    """
    path_key = pl.col("file_path").str.to_lowercase()
    return (
        images.with_columns(
            _path_key=path_key, _name_key=path_key.str.split("/").list.last()
        )
        .sort(GALLERY_SORTS[order], descending=descending, nulls_last=True)
        .drop("_path_key", "_name_key")
    )


def load_file_mentions(data_dir: Path, file_path: str) -> pl.DataFrame:
    """Every response that named `file_path`, newest first."""
    path = data_dir / step_10_pipeline.MENTIONS_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter(pl.col("file_path") == file_path)
        .sort("seen_at", descending=True)
        .collect()
    )


def load_file_downlink_time_periods(data_dir: Path, file_key: str) -> pl.DataFrame:
    """Every downlink of the file `file_key`, newest first."""
    path = data_dir / step_11_pipeline.DOWNLINK_TIME_PERIODS_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter(pl.col("file_key") == file_key)
        .sort("started_at", descending=True)
        .collect()
    )


@dataclass(frozen=True, slots=True)
class DownlinkedFile:
    """One reassembled file, bytes and all."""

    row: dict[str, Any]  # every column but `data`/`preview_image`
    data: bytes
    result: ReassemblyResult  # for `file_reassembler_page.coverage_map`


def load_downlinked_file(data_dir: Path, file_key: str) -> DownlinkedFile | None:
    """The reassembled file `file_key`, or None if there's no such file (or
    it was never downlinked)."""
    path = _files_path(data_dir)
    if not path.exists():
        return None
    rows = (
        pl.scan_parquet(path)
        .filter((pl.col("file_key") == file_key) & pl.col("data").is_not_null())
        .drop("preview_image")
        .collect()
    )
    if rows.is_empty():
        return None
    row = rows.row(0, named=True)
    data: bytes = row.pop("data")
    return DownlinkedFile(
        row=row,
        data=data,
        result=ReassemblyResult(
            total_chunks=row["chunk_count"],
            unique_offsets=0,
            offsets=(),
            segments=step_11_pipeline.decode_segments(row["segments"]),
            data=data,
            span_bytes=len(data),
            policy=DEFAULT_CONFLICT_POLICY,
        ),
    )


def _is_partial(row: dict[str, Any]) -> bool:
    return not (row.get("is_complete") and row.get("sha256_verified") is not False)


def download_filename(row: dict[str, Any], suffix: str | None = None) -> str:
    """The name a reassembled file downloads as: its own name -- with
    `_partial` on the stem unless it's complete (and
    not contradicted by the satellite's SHA-256), as on the File
    Reassembler page. `suffix` replaces the extension (a converted image).
    """
    path = PurePosixPath(PurePosixPath(row["file_path"]).name)
    if suffix is not None:
        path = path.with_suffix(suffix)
    stem = path.stem + (PARTIAL_FILENAME_SUFFIX if _is_partial(row) else "")
    return f"{stem}{path.suffix}"


def status_text(row: dict[str, Any]) -> str:
    """A file's downlink state in a word or two."""
    if not row.get("downlink_count"):
        return "Not downlinked"
    if row.get("sha256_verified") is True:
        return "Verified"
    if row.get("sha256_verified") is False:
        return "SHA-256 mismatch"
    if row.get("is_complete"):
        return "Complete"
    if row.get("conflict_bytes"):
        return "Conflicting"
    return "Partial"


def _query(file_key: str) -> str:
    return urlencode({"key": file_key})


def file_download_url(file_key: str) -> str:
    return f"{FILE_DOWNLOAD_PATH}?{_query(file_key)}"


def file_preview_url(file_key: str) -> str:
    return f"{FILE_PREVIEW_PATH}?{_query(file_key)}"


def _content_disposition(kind: str, filename: str) -> str:
    # RFC 5987's `filename*` keeps a non-ASCII name from breaking the header.
    return f"{kind}; filename*=UTF-8''{quote(filename, safe='')}"


def register_downlinked_file_routes(app: FastAPI, data_dir: Path) -> None:
    """Register `GET {FILE_DOWNLOAD_PATH}?key=...` (a reassembled file's
    bytes, as an attachment) and `GET {FILE_PREVIEW_PATH}?key=...` (its
    preview image, inline) on `app`.
    """

    def _load(key: str, columns: list[str]) -> dict[str, Any]:
        path = _files_path(data_dir)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Run step_11 first.")
        rows = (
            pl.scan_parquet(path)
            .filter((pl.col("file_key") == key) & pl.col("data").is_not_null())
            .select("file_key", "file_path", "is_complete", "sha256_verified", *columns)
            .collect()
        )
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=f"No downlinked file {key!r}.")
        return rows.row(0, named=True)

    def download(key: str) -> Response:
        row = _load(key, ["data"])
        return Response(
            row["data"],
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": _content_disposition(
                    "attachment", download_filename(row)
                )
            },
        )

    def preview(key: str) -> Response:
        row = _load(key, ["data", "preview_image", "preview_media_type"])
        media_type = row["preview_media_type"]
        if media_type is None:
            raise HTTPException(status_code=404, detail="Not an image.")
        converted = row["preview_image"] is not None
        suffix = {"image/jpeg": ".jpg", "image/bmp": ".bmp"}.get(media_type)
        return Response(
            row["preview_image"] if converted else row["data"],
            media_type=media_type,
            headers={
                "Content-Disposition": _content_disposition(
                    "inline",
                    download_filename(row, suffix if converted else None),
                ),
                # Regenerated (if at all) only by the next pipeline run.
                "Cache-Control": "private, max-age=300",
            },
        )

    # `add_api_route` rather than `@app.get(...)` -- see
    # `export_raw.register_raw_export_route`.
    app.add_api_route(FILE_DOWNLOAD_PATH, download, methods=["GET"])
    app.add_api_route(FILE_PREVIEW_PATH, preview, methods=["GET"])
