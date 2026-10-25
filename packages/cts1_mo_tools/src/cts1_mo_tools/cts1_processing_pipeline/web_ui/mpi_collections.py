"""Loading `mpi_collections.parquet` (step 6's per-collection summary) for
the web UI's MPI Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars -- see
`satellite_events.py` for the analogous loader over step 4's output.
"""

from __future__ import annotations

__all__ = [
    "analyses_of",
    "collections_csv_bytes",
    "collections_xlsx_bytes",
    "load_analyses",
    "load_collections",
    "response_properties",
    "strong_signal_counts_text",
]

import io
import json
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import xlsxwriter  # pyright: ignore[reportMissingTypeStubs]

from cts1_mo_tools.cts1_processing_pipeline.common import drop_timezones_for_excel
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_collections(data_dir: Path) -> pl.DataFrame:
    """Every MPI collection, newest first -- empty if step 6 hasn't run yet."""
    path = data_dir / step_6_pipeline.COLLECTIONS_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path).sort("collection_started_at", descending=True)


def load_analyses(data_dir: Path) -> pl.DataFrame:
    """Every distinct MPI analysis (step 6's `mpi_analyses.parquet`) --
    empty if step 6 hasn't run (since analyses were added) yet.
    """
    path = data_dir / step_6_pipeline.ANALYSES_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def analyses_of(
    analyses: pl.DataFrame, collection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every distinct analysis in `analyses` of `collection`'s file (one row
    of `mpi_collections.parquet`), the one the collection uses first -- see
    `step_6_pipeline.sort_mpi_analyses_by_preference`.
    """
    if analyses.is_empty() or collection.get("analysis_action") is None:
        return []
    file = collection.get("analysis_file")
    matching = analyses.filter(
        pl.col("analysis_file") == file
        if file is not None
        # A nameless analysis is its own file.
        else pl.col("analysis_file").is_null()
        & (pl.col("analysis_received_at") == collection["analysis_received_at"])
    )
    return step_6_pipeline.sort_mpi_analyses_by_preference(matching).to_dicts()


def strong_signal_counts_text(counts_json: str | None) -> str:
    """`analysis_strong_signal_counts` (a JSON object of threshold -> count)
    for display, e.g. "≥100 DN: 1 · ≥200 DN: 0".
    """
    if not counts_json:
        return ""
    counts = cast("dict[str, int]", json.loads(counts_json))
    return " · ".join(f"≥{dn} DN: {count}" for dn, count in counts.items())


def response_properties(response_text: str) -> dict[str, Any]:
    """Every property of an analysis's raw `analysis_response_text` (which
    step 6 already checked is a JSON object), as sent.
    """
    return cast("dict[str, Any]", json.loads(response_text))


def collections_csv_bytes(collections: pl.DataFrame) -> bytes:
    """`collections` (every column, not just the ones the page shows) as CSV."""
    return collections.write_csv().encode("utf-8")


def collections_xlsx_bytes(collections: pl.DataFrame) -> bytes:
    """`collections` (every column, not just the ones the page shows) as an
    Excel (.xlsx) workbook.
    """
    buf = io.BytesIO()
    workbook = xlsxwriter.Workbook(buf, {"in_memory": True})
    drop_timezones_for_excel(collections).write_excel(
        workbook=workbook, worksheet="mpi_collections"
    )
    workbook.close()
    return buf.getvalue()
