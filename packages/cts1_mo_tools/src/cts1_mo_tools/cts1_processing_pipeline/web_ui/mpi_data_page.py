"""The "MPI Data" page: when the satellite collected each MPI frame,
alongside its beacons and GNSS samples, by its own clock -- see
`collection_timeline` for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_mpi_data_page"]

from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere

from nicegui import ui

from . import collection_timeline
from .charts import collection_timeline_option
from .layout import page_shell


def _collection_timeline_card(data_dir: Path) -> None:
    with ui.card().classes("w-full"):
        ui.label("Collection Timeline").classes("text-lg font-bold")
        ui.label(
            "When the satellite collected each packet/frame, by its own clock "
            "-- not when it was downlinked. Beacons: the OBC's timestamp when "
            "it built the beacon. MPI frames: interpolated (or extrapolated) "
            "from the timestamps in the MPI data file. GNSS samples: the "
            "receiver's solution time. Scroll or drag the slider to zoom."
        ).classes("text-caption text-grey")

        timeline = collection_timeline.load_collection_timeline(data_dir)
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
    def content() -> None:
        _collection_timeline_card(data_dir)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Data").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=content.refresh)
        content()
