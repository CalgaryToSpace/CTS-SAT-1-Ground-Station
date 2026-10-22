"""The "MPI Data" page: when the satellite collected each MPI frame,
alongside its beacons and GNSS samples, by its own clock -- see
`collection_timeline` for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_mpi_data_page"]

from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING

from nicegui import run, ui

from . import collection_timeline
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


def build_mpi_data_page(data_dir: Path) -> None:
    @ui.refreshable
    def content(timeline: pl.DataFrame | None) -> None:
        _collection_timeline_card(timeline)

    async def _reload() -> None:
        # Load in a thread, then rebuild synchronously -- see
        # `pipeline_status_page.build_pipeline_status_page._reload`. Loading
        # can also wait on SatNOGS DB, for the launch date.
        timeline = await run.io_bound(
            collection_timeline.load_collection_timeline, data_dir
        )
        if timeline is None:  # Only when the server's shutting down.
            return
        content.refresh(timeline)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Data").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=_reload)
        content(None)

    ui.timer(0, _reload, once=True)
