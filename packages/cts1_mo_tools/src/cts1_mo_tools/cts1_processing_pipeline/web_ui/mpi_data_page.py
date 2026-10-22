"""The "MPI Data" page: when the satellite collected each MPI frame,
alongside its beacons and GNSS samples, by its own clock, and a summary of
each MPI collection -- see `collection_timeline` and `mpi_collections` for
the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_mpi_data_page"]

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING, Any

from nicegui import run, ui

from . import collection_timeline, mpi_collections
from .charts import collection_timeline_option
from .layout import page_shell

if TYPE_CHECKING:
    import polars as pl


def _collection_timeline_card(timeline: pl.DataFrame | None) -> None:
    """`timeline` is None while it's still loading."""
    with ui.card().classes("w-full"):
        ui.label("Collection Timeline").classes("text-lg font-bold")
        ui.label(
            "When the satellite collected each packet/frame, by its own clock "
            "-- not when it was downlinked. Beacons: the OBC's timestamp when "
            "it built the beacon (ones stamped before launch, from before its "
            "clock was first set, are left out). MPI frames: interpolated (or "
            "extrapolated) from the timestamps in the MPI data file. GNSS "
            "samples: the receiver's solution time. Scroll or drag the slider "
            "to zoom."
        ).classes("text-caption text-grey")

        if timeline is None:
            with ui.row().classes("items-center gap-2 p-2"):
                ui.spinner(size="md")
                ui.label("Loading...").classes("text-grey")
            return
        option = collection_timeline_option(
            timeline,
            sources=collection_timeline.TIMELINE_SOURCES,
            title="",
        )
        if option is None:
            ui.label(
                "Nothing collected yet -- run the pipeline (step_1..step_3, and "
                "step_6 for MPI frames) first."
            ).classes("text-caption text-grey mt-2")
            return
        ui.echart(option).classes("w-full h-80")


# (field, label, numeric) for each column of the collections table.
_COLLECTION_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("collection_started_at", "Started (UTC)", False),
    ("collection_ended_at", "Ended (UTC)", False),
    ("frames_present", "Frames Present", True),
    ("frames_expected", "Frames Expected", True),
    ("frames_missing", "Frames Missing", True),
    ("first_frame_counter", "First Counter", True),
    ("last_frame_counter", "Last Counter", True),
    ("largest_gap_frames", "Largest Gap (Frames)", True),
    ("largest_gap_started_at", "Largest Gap Start (UTC)", False),
    ("largest_gap_duration_ms", "Largest Gap Duration", True),
)


def _cell(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"{value:%Y-%m-%d %H:%M:%S.%f}"[:-3]
    if field == "largest_gap_duration_ms":
        return f"{value / 1000:,.2f} s"
    return value


def _collections_card(collections: pl.DataFrame | None) -> None:
    """`collections` is None while it's still loading."""
    with ui.card().classes("w-full"):
        ui.label("MPI Collections").classes("text-lg font-bold")
        ui.label(
            "Runs of MPI frames from one recording, newest first: a new one "
            "starts 20 minutes after the last one's first frame, or when the "
            "frame counter drops (a new recording, or the counter wrapping "
            "around). Frames expected/missing are from the first to the last "
            "frame counter present, so frames lost before the first or after "
            "the last aren't counted."
        ).classes("text-caption text-grey")

        if collections is None:
            with ui.row().classes("items-center gap-2 p-2"):
                ui.spinner(size="md")
                ui.label("Loading...").classes("text-grey")
            return
        if collections.is_empty():
            ui.label(
                "No MPI collections yet -- run the pipeline (step_1..step_3, "
                "then step_6) first."
            ).classes("text-caption text-grey mt-2")
            return

        columns = [
            {
                "name": field,
                "label": label,
                "field": field,
                "align": "right" if numeric else "left",
            }
            for field, label, numeric in _COLLECTION_COLUMNS
        ]
        rows = [
            {field: _cell(field, row[field]) for field, _, _ in _COLLECTION_COLUMNS}
            for row in collections.iter_rows(named=True)
        ]
        ui.table(columns=columns, rows=rows, row_key="collection_started_at").classes(
            "w-full"
        ).props("dense flat")
        ui.label(f"{collections.height:,} collection(s).").classes(
            "text-caption text-grey mt-2"
        )


@dataclass(frozen=True, slots=True)
class _PageData:
    timeline: pl.DataFrame
    collections: pl.DataFrame


def _load_page_data(data_dir: Path) -> _PageData:
    """Blocking (it may fetch the launch date) -- run it via `run.io_bound`."""
    return _PageData(
        timeline=collection_timeline.load_collection_timeline(data_dir),
        collections=mpi_collections.load_collections(data_dir),
    )


def build_mpi_data_page(data_dir: Path) -> None:
    @ui.refreshable
    def content(page_data: _PageData | None) -> None:
        _collection_timeline_card(page_data and page_data.timeline)
        _collections_card(page_data and page_data.collections)

    async def _reload() -> None:
        # Load in a thread, then rebuild synchronously -- see
        # `pipeline_status_page.build_pipeline_status_page._reload`. Loading
        # can also wait on SatNOGS DB, for the launch date.
        page_data = await run.io_bound(_load_page_data, data_dir)
        if page_data is None:  # Only when the server's shutting down.
            return
        content.refresh(page_data)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Data").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=_reload)
        content(None)

    ui.timer(0, _reload, once=True)
