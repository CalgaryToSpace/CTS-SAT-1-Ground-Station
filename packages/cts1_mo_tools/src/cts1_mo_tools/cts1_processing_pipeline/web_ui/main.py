"""CTS-SAT-1 Processing Pipeline Web UI.

A NiceGUI dashboard over `everything_decoded.parquet` (step 3's output):
recent beacon(s) at a glance, plus line/scatter charts for every field in
the basic and extended beacon packets, grouped by subsystem.

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
# ruff: noqa: TC003

__all__ = ["main"]

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import tyro
from nicegui import ui

from . import data as beacon_data
from .charts import BEACON_CHART_GROUPS, OTHER_CHART_GROUPS, chart_option

if TYPE_CHECKING:
    import polars as pl
    from nicegui import events

    from .charts import ChartSpec

REFRESH_INTERVAL_SEC = 30.0

# label -> hours; "All time" (None) skips the `received_at` filter entirely.
WINDOW_CHOICES: dict[str, float | None] = {
    "Last 1h": 1.0,
    "Last 6h": 6.0,
    "Last 24h": 24.0,
    "Last 7d": 24.0 * 7,
    "All time": None,
}


def _age_str(received_at: datetime) -> str:
    delta = datetime.now(UTC) - received_at.replace(tzinfo=UTC)
    total_sec = int(delta.total_seconds())
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s ago"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m ago"
    return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m ago"


def _latest_beacon_card(path: Path) -> None:
    latest = beacon_data.latest_beacons(path, n=1)
    with ui.card().classes("w-full"):
        if latest.is_empty():
            ui.label("No beacon packets decoded yet.").classes("text-lg")
            return

        row = latest.to_dicts()[0]
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(f"Latest beacon -- {row['packet_type']}").classes(
                    "text-xl font-bold"
                )
                received_at = row["received_at"]
                ui.label(
                    f"{received_at:%Y-%m-%d %H:%M:%S} UTC ({_age_str(received_at)})"
                ).classes("text-sm text-gray-400")
            ui.label(row.get("friendly_message") or "").classes("text-sm italic")

        stats = [
            ("Satellite", row.get("satellite_name")),
            ("Uptime", f"{row.get('uptime_sec', '?')} s"),
            ("Battery", f"{row.get('eps_battery_percent', '?')}%"),
            ("Battery Voltage", f"{row.get('eps_battery_voltage_V', '?')} V"),
            ("OBC Temp", f"{row.get('obc_temperature_C', '?')} °C"),
            ("EPS Mode", row.get("eps_mode")),
            ("CTS1 State", row.get("cts1_operation_state")),
            ("RSSI", f"{row.get('rssi_db', '?')} dB"),
        ]
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            for label, value in stats:
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-xs text-gray-400 uppercase")
                    ui.label(str(value)).classes("text-base font-medium")


def _recent_beacons_table(path: Path) -> None:
    recent = beacon_data.latest_beacons(path, n=10)
    with ui.card().classes("w-full"):
        ui.label("Recent Beacons").classes("text-lg font-bold")
        if recent.is_empty():
            ui.label("Nothing to show yet.")
            return

        columns = [
            {"name": "received_at", "label": "Received", "field": "received_at"},
            {"name": "packet_type", "label": "Type", "field": "packet_type"},
            {"name": "uptime_sec", "label": "Uptime (s)", "field": "uptime_sec"},
            {
                "name": "eps_battery_percent",
                "label": "Battery %",
                "field": "eps_battery_percent",
            },
            {
                "name": "obc_temperature_C",
                "label": "OBC Temp (°C)",
                "field": "obc_temperature_C",
            },
            {"name": "rssi_db", "label": "RSSI (dB)", "field": "rssi_db"},
        ]
        rows = [
            {
                "received_at": f"{r['received_at']:%Y-%m-%d %H:%M:%S}",
                "packet_type": r["packet_type"],
                "uptime_sec": r.get("uptime_sec"),
                "eps_battery_percent": r.get("eps_battery_percent"),
                "obc_temperature_C": r.get("obc_temperature_C"),
                "rssi_db": r.get("rssi_db"),
            }
            for r in recent.to_dicts()
        ]
        ui.table(columns=columns, rows=rows, row_key="received_at").classes("w-full")


def _render_chart_group(title: str, specs: list[ChartSpec], df: pl.DataFrame) -> None:
    """One collapsible chart group, with charts mounted lazily on first expand.

    An `ui.echart` is a real ECharts instance in the browser (its own canvas,
    its own render loop) -- mounting all ~47 of them up front, even behind a
    collapsed (but still-present) panel, is what makes the page laggy. Only
    building a group's charts the first time its panel is actually opened
    keeps the initial page (and every later re-render) down to whatever the
    user currently has expanded.
    """
    options = [
        option for spec in specs if (option := chart_option(spec, df)) is not None
    ]
    if not options:
        return

    built = False

    def _on_toggle(e: events.ValueChangeEventArguments) -> None:
        nonlocal built
        if not e.value or built:
            return
        built = True
        with (
            container,
            ui.grid(columns="repeat(auto-fit, minmax(420px, 1fr))").classes(
                "w-full gap-4 p-2"
            ),
        ):
            for option in options:
                ui.echart(option).classes("h-72")

    with ui.expansion(title, value=False, on_value_change=_on_toggle).classes(
        "w-full border rounded"
    ):
        container = ui.column().classes("w-full")


def _chart_groups(path: Path, *, since: datetime | None) -> None:
    beacons = beacon_data.load_beacon_window(path, since=since)
    all_packets = beacon_data.load_packet_window(path, since=since)

    for title, specs in BEACON_CHART_GROUPS:
        _render_chart_group(title, specs, beacons)

    for title, specs in OTHER_CHART_GROUPS:
        _render_chart_group(title, specs, all_packets)


@dataclass(frozen=True, slots=True)
class Args:
    """CTS-SAT-1 processing pipeline web UI."""

    parquet_path: Path = beacon_data.DEFAULT_PARQUET_PATH
    """Path to `everything_decoded.parquet` (step 3's output)."""

    hours: float = 24.0
    """Default chart time window, in hours. Adjustable in the UI afterward."""

    port: int = 8089
    """Port to serve the dashboard on."""


def _window_label_for_hours(hours: float | None) -> str:
    for label, value in WINDOW_CHOICES.items():
        if value == hours:
            return label
    return "Last 24h"


def _build_home_page(args: Args) -> None:
    state: dict[str, float | None] = {"hours": args.hours}

    # Split in two so the 30s auto-refresh only ever touches the cheap,
    # DOM-only status widgets -- the chart section (the expensive part,
    # once panels are open) is left alone unless the user explicitly asks
    # for it via the window selector or the "Refresh charts" button. Tearing
    # down and remounting ~47 ECharts instances every 30s regardless of
    # what's open was the main source of lag.
    @ui.refreshable
    def live_status() -> None:
        latest = beacon_data.latest_beacons(args.parquet_path, n=1)
        if latest.is_empty():
            ui.label(
                f"No beacon packets found at {args.parquet_path}. Run the "
                "pipeline (step_1, step_2, step_3) first."
            ).classes("text-lg text-warning")
            return
        _latest_beacon_card(args.parquet_path)
        _recent_beacons_table(args.parquet_path)

    @ui.refreshable
    def chart_section() -> None:
        since = (
            datetime.now(UTC) - timedelta(hours=state["hours"])
            if state["hours"] is not None
            else None
        )
        _chart_groups(args.parquet_path, since=since)

    def _on_window_change(label: str) -> None:
        state["hours"] = WINDOW_CHOICES[label]
        chart_section.refresh()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("CTS-SAT-1 Ground Station").classes("text-2xl font-bold")
            with ui.row().classes("items-center gap-4"):
                ui.select(
                    list(WINDOW_CHOICES),
                    value=_window_label_for_hours(state["hours"]),
                    label="Chart window",
                    on_change=lambda e: _on_window_change(e.value),
                ).classes("w-40")
                ui.button(
                    "Refresh charts", icon="refresh", on_click=chart_section.refresh
                )
        live_status()
        chart_section()

    ui.timer(REFRESH_INTERVAL_SEC, live_status.refresh)


def _build_file_reassembler_page() -> None:
    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("File Reassembler").classes("text-2xl font-bold")
        with ui.card().classes("w-full items-center p-12"):
            ui.icon("construction", size="xl").classes("text-gray-400")
            ui.label("Coming soon").classes("text-xl text-gray-400")
            ui.label(
                "This tool will reassemble downlinked bulk files from "
                "BULK_FILE_DOWNLINK packet chunks."
            ).classes("text-sm text-gray-400")


def _build_pages(args: Args) -> None:
    def _drawer() -> None:
        with ui.left_drawer(value=True).classes("bg-gray-900") as drawer:
            ui.link("Home", "/").classes("text-lg p-2")
            ui.link("File Reassembler", "/file-reassembler").classes("text-lg p-2")
        return drawer

    @ui.page("/")
    def home_page() -> None:
        _drawer()
        _build_home_page(args)

    @ui.page("/file-reassembler")
    def file_reassembler_page() -> None:
        _drawer()
        _build_file_reassembler_page()


def main() -> None:
    """Entry point: parse CLI args, build pages, and start the NiceGUI server."""
    args = tyro.cli(Args)
    _build_pages(args)
    ui.run(title="CTS-SAT-1 Ground Station", dark=True, port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
