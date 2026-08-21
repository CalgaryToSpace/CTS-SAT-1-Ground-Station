"""The "Beacon Stats" page: recent beacon(s) at a glance, plus line/scatter
charts for every field in the basic and extended beacon packets, grouped by
subsystem.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_beacon_stats_page"]

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING

from nicegui import ui

from . import data as beacon_data
from .charts import BEACON_CHART_GROUPS, OTHER_CHART_GROUPS, chart_option
from .layout import page_shell

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


def _format_uptime(uptime_sec: float | None) -> str:
    """`uptime_sec` as `[Dd] HH:MM:SS` -- the raw seconds count from the
    beacon field isn't something you can read at a glance once the
    satellite's been up for days.
    """
    if uptime_sec is None:
        return "?"
    total = int(uptime_sec)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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
                ).classes("text-caption text-grey")
            ui.label(row.get("friendly_message") or "").classes("text-sm italic")

        stats = [
            ("Satellite", row.get("satellite_name")),
            ("Uptime", _format_uptime(row.get("uptime_sec"))),
            ("Battery", f"{row.get('eps_battery_percent', '?')}%"),
            ("Battery Voltage", f"{row.get('eps_battery_voltage_V', '?')} V"),
            ("OBC Temp", f"{row.get('obc_temperature_C', '?')} °C"),
            ("EPS Mode", row.get("eps_mode")),
            ("CTS1 State", row.get("cts1_operation_state")),
        ]
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            for label, value in stats:
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-caption text-grey text-uppercase")
                    ui.label(str(value)).classes("text-base font-medium")


def _latest_extended_beacon_card(path: Path) -> None:
    """The extended-only fields (ADCS, extended EPS/OBC telemetry) only
    show up on `BEACON_EXTENDED` packets, which are interleaved with more
    frequent `BEACON_BASIC` ones -- so this looks up the latest *extended*
    beacon specifically, rather than reusing whatever `_latest_beacon_card`
    found.
    """
    latest = beacon_data.latest_beacons(path, n=1, packet_types=("BEACON_EXTENDED",))
    with ui.card().classes("w-full"):
        ui.label("Latest Extended Beacon").classes("text-lg font-bold")
        if latest.is_empty():
            ui.label("No extended beacon packets decoded yet.").classes(
                "text-caption text-grey"
            )
            return

        row = latest.to_dicts()[0]
        received_at = row["received_at"]
        ui.label(
            f"{received_at:%Y-%m-%d %H:%M:%S} UTC ({_age_str(received_at)})"
        ).classes("text-caption text-grey")

        stats = [
            ("ADC Battery Voltage", f"{row.get('obc_adc_battery_voltage_V', '?')} V"),
            (
                "MEMS Angular Rate Norm",
                f"{row.get('adcs_angular_rate_norm_deg_per_sec', '?')} deg/s",
            ),
            (
                "Angular Rate (x, y, z)",
                f"{row.get('adcs_estimated_rate_x_deg_per_sec', '?')}, "
                f"{row.get('adcs_estimated_rate_y_deg_per_sec', '?')}, "
                f"{row.get('adcs_estimated_rate_z_deg_per_sec', '?')} deg/s",
            ),
            ("ADCS Estimation Mode", row.get("adcs_attitude_estimation_mode")),
            ("ADCS Control Mode", row.get("adcs_control_mode")),
            (
                "Net Battery Power (avg)",
                f"{row.get('eps_total_avg_net_battery_power_W', '?')} W",
            ),
            (
                "Power Distributed (avg)",
                f"{row.get('eps_total_avg_power_distributed_W', '?')} W",
            ),
        ]
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            for label, value in stats:
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-caption text-grey text-uppercase")
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
            {"name": "uptime_sec", "label": "Uptime", "field": "uptime_sec"},
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
                "uptime_sec": _format_uptime(r.get("uptime_sec")),
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


def _window_label_for_hours(hours: float | None) -> str:
    for label, value in WINDOW_CHOICES.items():
        if value == hours:
            return label
    return "Last 24h"


def build_beacon_stats_page(parquet_path: Path, hours: float) -> None:
    state: dict[str, float | None] = {"hours": hours}

    # Split in two so the 30s auto-refresh only ever touches the cheap,
    # DOM-only status widgets -- the chart section (the expensive part,
    # once panels are open) is left alone unless the user explicitly asks
    # for it via the window selector or the "Refresh charts" button. Tearing
    # down and remounting ~47 ECharts instances every 30s regardless of
    # what's open was the main source of lag.
    @ui.refreshable
    def live_status() -> None:
        latest = beacon_data.latest_beacons(parquet_path, n=1)
        if latest.is_empty():
            ui.label(
                f"No beacon packets found at {parquet_path}. Run the "
                "pipeline (step_1, step_2, step_3) first."
            ).classes("text-lg text-warning")
            return
        _latest_beacon_card(parquet_path)
        _latest_extended_beacon_card(parquet_path)
        _recent_beacons_table(parquet_path)

    @ui.refreshable
    def chart_section() -> None:
        since = (
            datetime.now(UTC) - timedelta(hours=state["hours"])
            if state["hours"] is not None
            else None
        )
        _chart_groups(parquet_path, since=since)

    def _on_window_change(label: str) -> None:
        state["hours"] = WINDOW_CHOICES[label]
        chart_section.refresh()

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Beacon Stats").classes("text-2xl font-bold")
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
