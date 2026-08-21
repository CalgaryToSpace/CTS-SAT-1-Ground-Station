"""CTS-SAT-1 Processing Pipeline Web UI.

A NiceGUI dashboard over `everything_decoded.parquet` (step 3's output):
recent beacon(s) at a glance, plus line/scatter charts for every field in
the basic and extended beacon packets, grouped by subsystem -- see
`beacon_stats_page`, `packet_browser_page`, `satellite_events_page`,
`file_reassembler_page`, `pipeline_status_page`, and `export_page` for the
individual pages.

Usage (uv):
    uv run cts1_data_web_ui
    uv run cts1_data_web_ui --data-dir output
    uv run cts1_data_web_ui --hours 6
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

__all__ = ["main"]

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import tyro
import uvicorn.config
from nicegui import app, ui

from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)

from .beacon_stats_page import build_beacon_stats_page
from .export_page import build_export_page
from .export_raw import register_raw_export_route
from .file_reassembler_page import build_file_reassembler_page
from .packet_browser_page import build_packet_browser_page
from .pipeline_status_page import build_pipeline_status_page
from .satellite_events_page import build_satellite_events_page

# Signs the cookie `app.storage.user` uses to remember each browser's dark
# mode preference across visits -- not protecting anything sensitive (this
# dashboard has no login), so a fixed fallback is fine; override via env if
# you'd rather every deployment use its own.
STORAGE_SECRET = os.environ.get(
    "CTS1_WEB_STORAGE_SECRET", "cts1-processing-pipeline-web-ui"
)

# NiceGUI's own internal traffic -- Socket.IO polling/handshakes and static
# asset fetches -- lives under these prefixes; at uvicorn's "info" level it
# floods the access log with lines that aren't real page hits.
_NOISY_ACCESS_LOG_PREFIXES = ("/_nicegui_ws/", "/_nicegui/")


class _AccessLogNoiseFilter(logging.Filter):
    """Drops access-log records for `_NOISY_ACCESS_LOG_PREFIXES`."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.logging.AccessFormatter's own record shape:
        # args = (client_addr, method, full_path, http_version, status_code).
        try:
            path = record.args[2]  # type: ignore[index]
        except (TypeError, IndexError):
            return True
        return not str(path).startswith(_NOISY_ACCESS_LOG_PREFIXES)


def _build_uvicorn_log_config() -> dict:
    """uvicorn's own default logging config, with `uvicorn.access` at
    "info" (so real requests still get logged) while everything else stays
    at "warning", and NiceGUI's own noisy internal traffic filtered out of
    the access log -- see `_AccessLogNoiseFilter`.

    Passed as `log_config=` alongside `uvicorn_logging_level=None`: uvicorn
    applies `log_level` *after* `log_config` and would otherwise clobber
    the access logger's level right back down to whatever `log_level` is.
    """
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["loggers"]["uvicorn"]["level"] = "WARNING"
    config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    config["loggers"]["uvicorn.access"]["level"] = "INFO"
    config["filters"] = {"access_log_noise": {"()": _AccessLogNoiseFilter}}
    config["handlers"]["access"]["filters"] = ["access_log_noise"]
    return config


NAV_LINKS = (
    ("Beacon Stats", "/"),
    ("Browse Packets", "/browse-packets"),
    ("Satellite Events", "/satellite-events"),
    ("File Reassembler", "/file-reassembler"),
    ("Pipeline Status", "/pipeline-status"),
    ("Export Data", "/export"),
)


@dataclass(frozen=True, slots=True)
class Args:
    """CTS-SAT-1 processing pipeline web UI."""

    data_dir: Path = step_1_pipeline.DEFAULT_DATA_DIR
    """Directory every pipeline step reads/writes its file(s) in -- each
    page finds the file(s) it needs by a fixed filename inside it."""

    hours: float = 24.0
    """Default chart time window, in hours. Adjustable in the UI afterward."""

    port: int = 8089
    """Port to serve the dashboard on."""


def _build_pages(args: Args) -> None:
    def _nav() -> None:
        # Remembered per-browser (see STORAGE_SECRET) so a toggle sticks
        # across page navigations and later visits; defaults to dark for a
        # browser that's never set a preference.
        dark = ui.dark_mode(app.storage.user.get("dark_mode", True))

        def _toggle_dark() -> None:
            dark.toggle()
            app.storage.user["dark_mode"] = dark.value

        with ui.header().classes("items-center gap-4"):
            ui.label("FrontierSat Data").classes("text-lg font-bold")
            with ui.row().classes("items-center gap-0 ml-4 divide-x divide-white/25"):
                for label, path in NAV_LINKS:
                    ui.link(label, path).classes(
                        "text-white text-body1 no-underline px-3"
                    )
            ui.space()
            ui.button(icon="dark_mode", on_click=_toggle_dark).props(
                "flat round color=white"
            ).bind_icon_from(
                dark,
                "value",
                backward=lambda is_dark: "light_mode" if is_dark else "dark_mode",
            ).tooltip("Toggle dark mode")

    @ui.page("/")
    def beacon_stats_page() -> None:
        _nav()
        build_beacon_stats_page(args.data_dir, args.hours)

    @ui.page("/browse-packets")
    def packet_browser_page() -> None:
        _nav()
        build_packet_browser_page(args.data_dir)

    @ui.page("/satellite-events")
    def satellite_events_page() -> None:
        _nav()
        build_satellite_events_page(args.data_dir)

    @ui.page("/file-reassembler")
    def file_reassembler_page() -> None:
        _nav()
        build_file_reassembler_page(args.data_dir)

    @ui.page("/pipeline-status")
    def pipeline_status_page() -> None:
        _nav()
        build_pipeline_status_page(args.data_dir)

    @ui.page("/export")
    def export_page() -> None:
        _nav()
        build_export_page(args.data_dir)

    register_raw_export_route(app, args.data_dir)


def main() -> None:
    """Entry point: parse CLI args, build pages, and start the NiceGUI server."""
    args = tyro.cli(Args)
    _build_pages(args)
    # Real requests get logged (`docker compose logs web` -- or the
    # daemon's own log driver, see DEPLOY.md -- already captures this like
    # every other log line in this stack) without NiceGUI's own
    # Socket.IO/static-asset traffic flooding it -- see
    # `_build_uvicorn_log_config`. `uvicorn_logging_level=None` is required
    # for that log_config's per-logger levels to actually stick.
    ui.run(
        title="FrontierSat Data",
        port=args.port,
        reload=False,
        uvicorn_logging_level=None,  # pyright: ignore[reportArgumentType]
        log_config=_build_uvicorn_log_config(),
        storage_secret=STORAGE_SECRET,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
