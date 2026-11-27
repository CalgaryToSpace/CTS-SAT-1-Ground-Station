"""The "Satellite Events" page: every TX outage (step 12), then every OBC
reboot, EPS reboot, and uplinked-commands event detected from beacon
counters (step 4), newest first -- see `satellite_events` for the data layer
and `step_4_detect_satellite_events.pipeline`/
`step_12_detect_tx_outages.pipeline` for the detection logic itself.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_satellite_events_page", "format_chance"]

import math
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING

import polars as pl
from nicegui import ui

from cts1_mo_tools.cts1_processing_pipeline.step_4_detect_satellite_events.pipeline import (  # noqa: E501
    EVENT_SPECS,
)
from cts1_mo_tools.time_str import age_str, duration_str

from . import satellite_events
from .layout import page_shell

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

EVENT_TYPES = tuple(spec.event_type for spec in EVENT_SPECS)

# Uplinked-commands events are far more frequent (any uplink at all) and
# much less operationally interesting than a reboot -- start with them
# hidden so the table opens on the events worth looking at.
DEFAULT_ENABLED_EVENT_TYPES = frozenset(EVENT_TYPES) - {"Uplinked Commands"}


def _events_table(events: pl.DataFrame) -> None:
    if events.is_empty():
        ui.label(
            "No events detected yet -- run the pipeline (step_1..step_4) first."
        ).classes("text-caption text-grey mt-2")
        return

    columns = [
        {"name": "event_type", "label": "Event Type", "field": "event_type"},
        {
            "name": "detected_at",
            "label": "Detected At (UTC)",
            "field": "detected_at",
        },
        {
            "name": "estimated_event_at",
            "label": "Estimated Event At (UTC)",
            "field": "estimated_event_at",
        },
        {
            "name": "time_since_event_estimated",
            "label": "Time Since Event (Estimated)",
            "field": "time_since_event_estimated",
        },
        {
            "name": "event_to_beacon_dt",
            "label": "Event to Beacon ΔT",
            "field": "event_to_beacon_dt",
        },
        {
            "name": "obc_reboot_reason",
            "label": "OBC Reboot Reason",
            "field": "obc_reboot_reason",
        },
        {
            "name": "eps_reboot_reason",
            "label": "EPS Reboot Reason",
            "field": "eps_reboot_reason",
        },
        {
            "name": "detecting_packet_type",
            "label": "Detecting Packet",
            "field": "detecting_packet_type",
        },
    ]
    rows = [
        {
            "event_type": r["event_type"],
            "detected_at": f"{r['detected_at']:%Y-%m-%d %H:%M:%S}",
            "estimated_event_at": f"{r['estimated_event_at']:%Y-%m-%d %H:%M:%S}",
            "time_since_event_estimated": age_str(r["estimated_event_at"]),
            "event_to_beacon_dt": duration_str(
                r["time_since_event_when_detected_ms"] / 1000
            ),
            "obc_reboot_reason": r["obc_reboot_reason"] or "",
            "eps_reboot_reason": r["eps_reboot_reason"] or "",
            "detecting_packet_type": r["detecting_packet_type"],
        }
        for r in events.to_dicts()
    ]
    table = ui.table(columns=columns, rows=rows, row_key="detected_at").classes(
        "w-full"
    )
    table.add_slot(
        "body-cell-event_type",
        r"""
            <q-td :props="props">
                <q-badge :color="{
                    'OBC Reboot': 'negative',
                    'EPS Reboot': 'warning',
                    'Uplinked Commands': 'positive',
                }[props.value] || 'grey'">
                    {{ props.value }}
                </q-badge>
            </q-td>
        """,
    )
    ui.label(f"{events.height:,} event(s).").classes("text-caption text-grey mt-2")


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_chance(probability: float) -> str:
    """A probability as a percent down to 0.001%, then as its nearest power
    of ten with a superscript exponent, e.g. `10⁻²²`."""
    percent = probability * 100
    if percent == 0 or round(percent, 3) >= 0.001:  # noqa: PLR2004
        return f"{percent:.3f}%"
    exponent = round(math.log10(probability))
    return f"10{str(exponent).translate(_SUPERSCRIPT)}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return ""
    # duration_str rounds to whole seconds; a recovery beacon is often <1s.
    return f"{value:.1f}s" if value < 60 else duration_str(value)  # noqa: PLR2004


def _obc_and_eps(obc: str | None, eps: str | None) -> str:
    """OBC and EPS values stacked in one cell (one line each), or "" if neither."""
    return f"OBC: {obc}\nEPS: {eps}" if obc or eps else ""


def _fmt_time(value: datetime | None) -> str:
    return "" if value is None else f"{value:%Y-%m-%d %H:%M:%S}"


def _tx_outages_table(outages: pl.DataFrame) -> None:
    with ui.card().classes("w-full"):
        ui.label("TX Outages").classes("text-lg font-bold")
        ui.label(
            "Times the satellite seems to have stopped transmitting: 3h+ "
            "without a good packet, while 2+ stations that usually hear it "
            "listened and heard nothing. Chance is the probability that the "
            "detection is a mistake."
        ).classes("text-caption text-grey")

        if outages.is_empty():
            ui.label("No TX outages detected (or step_12 hasn't run yet).").classes(
                "text-caption text-grey mt-2"
            )
            return

        columns = [
            {"name": "status", "label": "Status", "field": "status"},
            {
                "name": "last_good",
                "label": "Last Good Packet (UTC)",
                "field": "last_good",
            },
            {
                "name": "last_uncorrectable",
                "label": "Last Uncorrectable Packet (UTC)",
                "field": "last_uncorrectable",
            },
            {
                "name": "first_good_after",
                "label": "First Good Packet After (UTC)",
                "field": "first_good_after",
            },
            {"name": "duration", "label": "Duration", "field": "duration"},
            {"name": "missed", "label": "Reliable Passes Missed", "field": "missed"},
            {"name": "chance", "label": "Chance", "field": "chance"},
            {
                "name": "recovery_beacon_dt",
                "label": "Recovery to Beacon ΔT",
                "field": "recovery_beacon_dt",
            },
            {
                "name": "recovery_uptime",
                "label": "Uptime at Recovery",
                "field": "recovery_uptime",
                "align": "left",
            },
            {
                "name": "recovery_reboot_reason",
                "label": "Reboot Reason at Recovery",
                "field": "recovery_reboot_reason",
                "align": "left",
            },
            {
                "name": "stations",
                "label": "Reliable Stations That Missed",
                "field": "stations",
                "align": "left",
            },
        ]
        # Long headers wrap, so the table fits the page instead of scrolling.
        for column in columns:
            column["headerStyle"] = "white-space: normal"
        rows = [
            {
                "status": "Ongoing" if r["is_ongoing"] else "Recovered",
                "last_good": _fmt_time(r["last_good_packet_at"]),
                "last_uncorrectable": _fmt_time(r["last_uncorrectable_packet_at"]),
                "first_good_after": _fmt_time(r["first_good_packet_after_at"]),
                "duration": (
                    f"{age_str(r['last_good_packet_at'])} so far"
                    if r["is_ongoing"]
                    else duration_str(r["gap_duration_ms"] / 1000)
                ),
                "missed": (
                    f"{r['missed_observation_count']} "
                    f"({r['missed_station_count']} stations)"
                ),
                "chance": format_chance(r["probability_by_chance"]),
                "recovery_beacon_dt": _fmt_seconds(
                    None
                    if r["recovery_beacon_at"] is None
                    else (
                        r["recovery_beacon_at"] - r["first_good_packet_after_at"]
                    ).total_seconds()
                ),
                "recovery_uptime": _obc_and_eps(
                    _fmt_seconds(r["recovery_obc_uptime_sec"]),
                    _fmt_seconds(r["recovery_eps_uptime_sec"]),
                ),
                "recovery_reboot_reason": _obc_and_eps(
                    r["recovery_obc_reboot_reason"], r["recovery_eps_reboot_reason"]
                ),
                "stations": ", ".join(r["missed_stations"]),
            }
            for r in outages.to_dicts()
        ]
        table = (
            ui.table(columns=columns, rows=rows, row_key="last_good")
            .classes("w-full")
            .props("dense")
        )
        # Browsers ignore a table cell's own width limits, so a fixed-width box.
        table.add_slot(
            "body-cell-stations",
            r"""
                <q-td :props="props">
                    <div style="width: 14rem; white-space: normal">
                        {{ props.value }}
                    </div>
                </q-td>
            """,
        )
        # One line each for OBC and EPS; an uptime never wraps, a reason may.
        for name, white_space in (
            ("recovery_uptime", "pre"),
            ("recovery_reboot_reason", "pre-line"),
        ):
            table.add_slot(
                f"body-cell-{name}",
                # No whitespace around the value: pre/pre-line would show it.
                f'<q-td :props="props" style="white-space: {white_space}">'
                "{{ props.value }}</q-td>",
            )
        table.add_slot(
            "body-cell-status",
            r"""
                <q-td :props="props">
                    <q-badge :color="props.value === 'Ongoing' ? 'negative' : 'grey'">
                        {{ props.value }}
                    </q-badge>
                </q-td>
            """,
        )
        ui.label(f"{outages.height:,} outage(s).").classes(
            "text-caption text-grey mt-2"
        )


def _event_type_filter(
    selected_types: dict[str, bool], *, on_change: Callable[[], object]
) -> None:
    with ui.row().classes("gap-4 flex-wrap items-center mt-2"):
        ui.label("Event type:").classes("text-sm font-medium")
        for event_type in EVENT_TYPES:
            ui.checkbox(
                event_type,
                value=selected_types[event_type],
                on_change=lambda e, et=event_type: (
                    selected_types.__setitem__(et, bool(e.value)),
                    on_change(),
                ),
            )


def build_satellite_events_page(data_dir: Path) -> None:
    selected_types: dict[str, bool] = {
        event_type: event_type in DEFAULT_ENABLED_EVENT_TYPES
        for event_type in EVENT_TYPES
    }

    @ui.refreshable
    def tx_outages() -> None:
        _tx_outages_table(satellite_events.load_tx_outages(data_dir))

    @ui.refreshable
    def content() -> None:
        events = satellite_events.load_events(data_dir)
        enabled = [et for et, checked in selected_types.items() if checked]
        if not events.is_empty():
            events = events.filter(pl.col("event_type").is_in(enabled))
        _events_table(events)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Satellite Events").classes("text-2xl font-bold")
            ui.button(
                "Refresh",
                icon="refresh",
                on_click=lambda: (tx_outages.refresh(), content.refresh()),
            )
        with ui.card().classes("w-full"):
            ui.label("Satellite Events").classes("text-lg font-bold")
            ui.label(
                "Every OBC reboot, EPS reboot, and uplinked-commands event, "
                "detected as the first beacon where the applicable onboard "
                "counter drops. The event itself is estimated to have "
                'happened "Event to Beacon ΔT" before that beacon.'
            ).classes("text-caption text-grey")
            _event_type_filter(selected_types, on_change=content.refresh)
            content()
        tx_outages()
