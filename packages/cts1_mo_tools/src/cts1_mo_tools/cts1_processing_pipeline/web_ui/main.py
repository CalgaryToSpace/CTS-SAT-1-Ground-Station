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

__all__ = ["main"]

from dataclasses import dataclass
from pathlib import Path

import tyro
from nicegui import app, ui

from . import data as beacon_data
from .beacon_stats_page import build_beacon_stats_page
from .export_page import build_export_page
from .export_raw import register_raw_export_route
from .file_reassembler_page import build_file_reassembler_page
from .pipeline_status_page import build_pipeline_status_page

NAV_LINKS = (
    ("Beacon Stats", "/"),
    ("File Reassembler", "/file-reassembler"),
    ("Pipeline Status", "/pipeline-status"),
    ("Export Data", "/export"),
)


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
    def _nav() -> None:
        with ui.header().classes("items-center gap-4"):
            ui.label("FrontierSat Data").classes("text-lg font-bold")
            with ui.row().classes("items-center gap-4 ml-4"):
                for label, path in NAV_LINKS:
                    ui.link(label, path).classes("text-white text-body1 no-underline")

    @ui.page("/")
    def beacon_stats_page() -> None:
        _nav()
        build_beacon_stats_page(args.parquet_path, args.hours)

    @ui.page("/file-reassembler")
    def file_reassembler_page() -> None:
        _nav()
        build_file_reassembler_page(args.parquet_path)

    @ui.page("/pipeline-status")
    def pipeline_status_page() -> None:
        _nav()
        build_pipeline_status_page(args.parquet_path)

    @ui.page("/export")
    def export_page() -> None:
        _nav()
        build_export_page(args.parquet_path)

    register_raw_export_route(app, args.parquet_path.parent)


def main() -> None:
    """Entry point: parse CLI args, build pages, and start the NiceGUI server."""
    args = tyro.cli(Args)
    _build_pages(args)
    ui.run(title="FrontierSat Data", dark=True, port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
