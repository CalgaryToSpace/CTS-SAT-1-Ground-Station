"""The "Satellite Events" page: every OBC reboot, EPS reboot, and uplinked-
commands event detected from beacon counters, newest first -- see
`satellite_events` for the data layer and
`step_4_detect_satellite_events.pipeline` for the detection logic itself.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_satellite_events_page"]

from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING

import polars as pl
from nicegui import ui

from cts1_mo_tools.cts1_processing_pipeline.step_4_detect_satellite_events.pipeline import (  # noqa: E501
    EVENT_SPECS,
)

from . import satellite_events
from .layout import page_shell

if TYPE_CHECKING:
    from collections.abc import Callable

EVENT_TYPES = tuple(spec.event_type for spec in EVENT_SPECS)

# Uplinked-commands events are far more frequent (any uplink at all) and
# much less operationally interesting than a reboot -- start with them
# hidden so the table opens on the events worth looking at.
DEFAULT_ENABLED_EVENT_TYPES = frozenset(EVENT_TYPES) - {"Uplinked Commands"}


def _duration_str(total_ms: int) -> str:
    total_sec = total_ms / 1000
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec:.1f}s"
    total_sec = int(total_sec)
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m {total_sec % 60}s"
    return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"


def _events_table(events: pl.DataFrame) -> None:
    with ui.card().classes("w-full"):
        ui.label("Satellite Events").classes("text-lg font-bold")
        ui.label(
            "Every OBC reboot, EPS reboot, and uplinked-commands event, "
            "detected as the first beacon where the applicable onboard "
            "counter drops -- the event itself is estimated to have "
            'happened "Event to Beacon ΔT" before that beacon.'
        ).classes("text-caption text-grey")

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
                "name": "time_since_event",
                "label": "Event to Beacon ΔT",
                "field": "time_since_event",
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
                "time_since_event": _duration_str(
                    r["time_since_event_when_detected_ms"]
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


def _event_type_filter(
    selected_types: dict[str, bool], *, on_change: Callable[[], object]
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Event type").classes("text-lg font-bold")
        with ui.row().classes("gap-4 flex-wrap mt-2"):
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
    def content() -> None:
        events = satellite_events.load_events(data_dir)
        enabled = [et for et, checked in selected_types.items() if checked]
        if not events.is_empty():
            events = events.filter(pl.col("event_type").is_in(enabled))
        _events_table(events)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Satellite Events").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=content.refresh)
        _event_type_filter(selected_types, on_change=content.refresh)
        content()
