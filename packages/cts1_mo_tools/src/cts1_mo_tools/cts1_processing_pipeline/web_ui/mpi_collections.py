"""Loading `mpi_collections.parquet` (step 8's per-collection summary) for
the web UI's MPI Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars -- see
`satellite_events.py` for the analogous loader over step 4's output.
"""

from __future__ import annotations

__all__ = [
    "analyses_of",
    "collections_csv_bytes",
    "collections_xlsx_bytes",
    "downlinked_percent_text",
    "integration_periods_text",
    "load_analyses",
    "load_collections",
    "position_text",
    "response_properties",
    "strong_signal_counts_text",
    "strong_signals_downlinked_text",
    "time_sync_sources_text",
]

import io
from typing import TYPE_CHECKING, Any, cast

import orjson
import polars as pl
import xlsxwriter  # pyright: ignore[reportMissingTypeStubs]

from cts1_mo_tools.cts1_processing_pipeline.common import drop_timezones_for_excel
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_collections(data_dir: Path) -> pl.DataFrame:
    """Every MPI collection, newest first -- empty if step 8 hasn't run yet."""
    path = data_dir / step_8_pipeline.COLLECTIONS_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path).sort("collection_started_at", descending=True)


def load_analyses(data_dir: Path) -> pl.DataFrame:
    """Every distinct MPI analysis (step 8's `mpi_analyses.parquet`) --
    empty if step 8 hasn't run (since analyses were added) yet.
    """
    path = data_dir / step_8_pipeline.ANALYSES_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def analyses_of(
    analyses: pl.DataFrame, collection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every distinct analysis in `analyses` of `collection`'s file (one row
    of `mpi_collections.parquet`), the one the collection uses first -- see
    `step_8_pipeline.sort_mpi_analyses_by_preference`.
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
    return step_8_pipeline.sort_mpi_analyses_by_preference(matching).to_dicts()


def strong_signal_counts_text(counts_json: str | None) -> str:
    """`analysis_strong_signal_counts` (a JSON object of threshold -> count)
    for display, e.g. "≥100 DN: 1 · ≥200 DN: 0".
    """
    if not counts_json:
        return ""
    counts = cast("dict[str, int]", orjson.loads(counts_json))
    return " · ".join(f"≥{dn} DN: {count}" for dn, count in counts.items())


def _signal_frame_count(signal: object) -> int | None:
    """How many frames a strong signal spans, from its `frames`
    ([first, last] frame index), if it has one.
    """
    if not isinstance(signal, dict):
        return None
    frames = cast("dict[str, Any]", signal).get("frames")
    if not (isinstance(frames, list) and len(cast("list[Any]", frames)) == 2):  # noqa: PLR2004
        return None
    first, last = cast("list[Any]", frames)
    if not (isinstance(first, int) and isinstance(last, int)) or last < first:
        return None
    return last - first + 1


def strong_signals_downlinked_text(
    signals: list[Any], downlinked_json: str | None
) -> list[str]:
    """Whether each of an analysis's strong `signals` was downlinked, for
    display -- "yes", "partly", or "no", with how many of its frames were,
    e.g. "partly (1/3 frames)". `downlinked_json` is the analysis's
    `analysis_strong_signal_frames_downlinked`; "unknown" for every signal
    without it (an `mpi_analyses.parquet` from before it was added).
    """
    counts = cast("list[int | None]", orjson.loads(downlinked_json or "[]"))
    texts: list[str] = []
    for i, signal in enumerate(signals):
        count = counts[i] if i < len(counts) else None
        if count is None:
            texts.append("unknown")
            continue
        expected = _signal_frame_count(signal)
        if expected is None:
            texts.append(f"yes ({count} frames)" if count else "no")
            continue
        status = "no" if count == 0 else "yes" if count >= expected else "partly"
        texts.append(f"{status} ({count}/{expected} frames)")
    return texts


def downlinked_percent_text(collection: dict[str, Any]) -> str:
    """How much of a collection (one row of `mpi_collections.parquet`) has
    been downlinked, for display, e.g. "37.2%": its frames present, out of
    the valid frames the satellite's analysis counted in the file -- or, with
    no analysis, out of those expected from the frame counters.
    """
    present = collection.get("frames_present")
    total = collection.get("analysis_valid_frame_count") or collection.get(
        "frames_expected"
    )
    if present is None or not total:
        return ""
    return f"{present / total:.1%}"


def position_text(latitude_deg: float | None, longitude_deg: float | None) -> str:
    """A sub-satellite point for display, e.g. "64.98°N 48.35°E"."""
    if latitude_deg is None or longitude_deg is None:
        return ""
    return (
        f"{abs(latitude_deg):.2f}°{'N' if latitude_deg >= 0 else 'S'} "
        f"{abs(longitude_deg):.2f}°{'E' if longitude_deg >= 0 else 'W'}"
    )


def integration_periods_text(periods_json: str | None) -> str:
    """`integration_periods` (a JSON array) for display, e.g. "200, 260"."""
    if not periods_json:
        return ""
    return ", ".join(str(p) for p in cast("list[int]", orjson.loads(periods_json)))


# The firmware's letter for each time sync source (see
# `TIME_sync_source_enum_to_letter_char` in the OBC firmware's
# timekeeping.c), by its `TIME_SYNC_SOURCE_*` enum name there.
_TIME_SYNC_SOURCE_NAMES = {
    "G": "GNSS_UART",
    "P": "GNSS_PPS",
    "T": "TELECOMMAND_ABSOLUTE",
    "C": "TELECOMMAND_CORRECTION",
    "E": "EPS_RTC",
    "N": "NONE",
}


def time_sync_sources_text(sources_json: str | None) -> str:
    """`time_sync_sources` (a JSON array of the firmware's letters) for
    display, e.g. "EPS_RTC (E), GNSS_UART (G)".
    """
    if not sources_json:
        return ""
    return ", ".join(
        f"{_TIME_SYNC_SOURCE_NAMES.get(letter, 'UNKNOWN')} ({letter})"
        for letter in cast("list[str]", orjson.loads(sources_json))
    )


def response_properties(response_text: str) -> dict[str, Any]:
    """Every property of an analysis's raw `analysis_response_text` (which
    step 8 already checked is a JSON object), as sent.
    """
    return cast("dict[str, Any]", orjson.loads(response_text))


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
