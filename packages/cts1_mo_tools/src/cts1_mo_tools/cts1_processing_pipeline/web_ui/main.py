"""CTS-SAT-1 Processing Pipeline Web UI.

A NiceGUI dashboard over `everything_decoded.parquet` (step 3's output):
recent beacon(s) at a glance, plus line/scatter charts for every field in
the basic and extended beacon packets, grouped by subsystem -- see
`beacon_stats_page`, `file_reassembler_page`, `pipeline_status_page`, and
`export_page` for the individual pages.

Usage (uv):
    uv run cts1_data_web_ui
    uv run cts1_data_web_ui --parquet-path output/everything_decoded.parquet
    uv run cts1_data_web_ui --hours 6
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

# tyro resolves dataclass field annotations at runtime (via get_type_hints),
# so imports used only in annotations below still need to be real imports.

__all__ = ["main"]

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime, see below

import tyro
from nicegui import app, ui

from . import data as beacon_data
from .beacon_stats_page import build_beacon_stats_page
from .export_page import build_export_page
from .export_raw import register_raw_export_route
from .file_reassembler_page import build_file_reassembler_page
from .pipeline_status_page import build_pipeline_status_page


@dataclass(frozen=True, slots=True)
class Args:
    """CTS-SAT-1 processing pipeline web UI."""

    parquet_path: Path = beacon_data.DEFAULT_PARQUET_PATH
    """Path to `everything_decoded.parquet` (step 3's output)."""

    hours: float = 24.0
    """Default chart time window, in hours. Adjustable in the UI afterward."""

    port: int = 8089
    """Port to serve the dashboard on."""


def _build_pages(args: Args) -> None:
    def _drawer() -> None:
        with ui.left_drawer(value=True):
            ui.link("Beacon Stats", "/").classes("text-lg p-2")
            ui.link("File Reassembler", "/file-reassembler").classes("text-lg p-2")
            ui.link("Pipeline Status", "/pipeline-status").classes("text-lg p-2")
            ui.link("Export Data", "/export").classes("text-lg p-2")

    @ui.page("/")
    def beacon_stats_page() -> None:
        _drawer()
        build_beacon_stats_page(args.parquet_path, args.hours)

    @ui.page("/file-reassembler")
    def file_reassembler_page() -> None:
        _drawer()
        build_file_reassembler_page(args.parquet_path)

    @ui.page("/pipeline-status")
    def pipeline_status_page() -> None:
        _drawer()
        build_pipeline_status_page(args.parquet_path)

    @ui.page("/export")
    def export_page() -> None:
        _drawer()
        build_export_page(args.parquet_path)

    register_raw_export_route(app, args.parquet_path.parent)


def main() -> None:
    """Entry point: parse CLI args, build pages, and start the NiceGUI server."""
    args = tyro.cli(Args)
    _build_pages(args)
    ui.run(title="CTS-SAT-1 Ground Station", dark=True, port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
