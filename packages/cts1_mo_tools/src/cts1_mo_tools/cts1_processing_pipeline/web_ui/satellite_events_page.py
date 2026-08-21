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

from nicegui import ui

from . import satellite_events
from .layout import page_shell

if TYPE_CHECKING:
    import polars as pl


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
            'happened "Time Since Event" before that beacon.'
        ).classes("text-caption text-grey")

        if events.is_empty():
            ui.label(
                "No events detected yet -- run the pipeline (step_1..step_4) "
                "first."
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
                "label": "Time Since Event",
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
                "name": "eps_reset_count",
                "label": "EPS Reset Count",
                "field": "eps_reset_count",
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
                "eps_reset_count": r["eps_reset_count"],
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
                        'EPS Reboot': 'negative',
                        'Uplinked Commands': 'positive',
                    }[props.value] || 'grey'">
                        {{ props.value }}
                    </q-badge>
                </q-td>
            """,
        )
        ui.label(f"{events.height:,} event(s).").classes("text-caption text-grey mt-2")


def build_satellite_events_page(output_dir: Path) -> None:
    @ui.refreshable
    def content() -> None:
        _events_table(satellite_events.load_events(output_dir))

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Satellite Events").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=content.refresh)
        content()
