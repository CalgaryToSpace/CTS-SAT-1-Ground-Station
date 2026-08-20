"""The "Pipeline Status" page: an at-a-glance view of whether the
processing pipeline is up to date and working -- row counts and freshness
timestamps at every stage, a per-decoder-tool scoreboard, and the most
recently received packets -- see `pipeline_status` for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_pipeline_status_page"]

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere

from nicegui import ui

from . import pipeline_status as status_data
from .layout import page_shell

REFRESH_INTERVAL_SEC = 30.0

# Past these ages, a freshness timestamp is flagged yellow/red -- SatNOGS
# overpasses aren't constant, so a short gap is normal, but the pipeline
# genuinely stalling for hours is the thing this page exists to catch.
STALE_WARNING_AFTER = timedelta(minutes=90)
STALE_ERROR_AFTER = timedelta(hours=6)

RECENT_PACKETS_LIMIT = 25


def _age_str(value: datetime) -> str:
    delta = datetime.now(UTC) - value
    total_sec = int(delta.total_seconds())
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s ago"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m ago"
    if total_sec < 86400:  # noqa: PLR2004
        return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m ago"
    return f"{total_sec // 86400}d {(total_sec % 86400) // 3600}h ago"


def _freshness_row(label: str, value: datetime | None) -> None:
    with ui.row().classes("w-full items-center justify-between"):
        ui.label(label).classes("text-caption text-grey text-uppercase")
        if value is None:
            with ui.row().classes("items-center gap-1"):
                ui.icon("help", color="grey")
                ui.label("no data yet").classes("text-caption text-grey")
            return

        age = datetime.now(UTC) - value
        if age > STALE_ERROR_AFTER:
            color, icon = "negative", "error"
        elif age > STALE_WARNING_AFTER:
            color, icon = "warning", "warning"
        else:
            color, icon = "positive", "check_circle"
        with ui.row().classes("items-center gap-1"):
            ui.icon(icon, color=color)
            ui.label(f"{value:%Y-%m-%d %H:%M:%S} UTC ({_age_str(value)})").classes(
                f"text-{color}"
            )


def _stat_tile(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-caption text-grey text-uppercase")
        ui.label(value).classes("text-2xl font-bold")


def _counts_section(counts: status_data.PipelineCounts) -> None:
    with ui.card().classes("w-full"):
        ui.label("Pipeline Counts").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            _stat_tile("Observations", f"{counts.total_observations:,}")
            _stat_tile("Raw decodes", f"{counts.total_raw_decodes:,}")
            _stat_tile("Distinct packets", f"{counts.total_distinct_packets:,}")
            _stat_tile("Decoded packets", f"{counts.total_decoded_packets:,}")
        if counts.decode_backlog:
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.icon("warning", color="warning")
                ui.label(
                    f"{counts.decode_backlog:,} distinct packet(s) haven't been "
                    "run through step 3 (decoding) yet."
                ).classes("text-warning")

        ui.separator().classes("mt-2")
        ui.label("Freshness").classes("text-base font-medium mt-2")
        _freshness_row("Latest observation end (SatNOGS)", counts.latest_observation_end)
        _freshness_row("Latest packet received", counts.latest_packet_received_at)
        _freshness_row("Latest packet ingested locally", counts.latest_packet_ingested_at)
        _freshness_row("Latest decoder run", counts.latest_decoder_run_at)


def _decoder_tools_table(stats: list[status_data.DecoderToolStats]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Decoder Tool Scoreboard").classes("text-lg font-bold")
        ui.label(
            "\"Distinct Packets\" is each tool's share of every distinct "
            "(post-dedup) packet ever seen -- the meaningful score, since "
            "\"Raw Decodes\" double-counts the same packet across every "
            "ground station that caught it."
        ).classes("text-caption text-grey")
        if not stats:
            ui.label("No decoder runs recorded yet.").classes("text-caption text-grey")
            return

        columns = [
            {"name": "decoder", "label": "Decoder", "field": "decoder"},
            {
                "name": "distinct_packet_count",
                "label": "Distinct Packets",
                "field": "distinct_packet_count",
            },
            {"name": "share", "label": "Share", "field": "share"},
            {
                "name": "raw_decode_count",
                "label": "Raw Decodes",
                "field": "raw_decode_count",
            },
            {
                "name": "avg_runtime_ms",
                "label": "Avg Runtime (ms)",
                "field": "avg_runtime_ms",
            },
            {"name": "last_version", "label": "Version", "field": "last_version"},
            {"name": "last_run_at", "label": "Last Run", "field": "last_run_at"},
        ]
        rows = [
            {
                "decoder": s.decoder,
                "distinct_packet_count": f"{s.distinct_packet_count:,}",
                "share": f"{s.distinct_packet_share:.1%}",
                "raw_decode_count": f"{s.raw_decode_count:,}",
                "avg_runtime_ms": (
                    f"{s.avg_runtime_ms:,.0f}" if s.avg_runtime_ms is not None else ""
                ),
                "last_version": s.last_version or "",
                "last_run_at": (
                    f"{s.last_run_at:%Y-%m-%d %H:%M:%S} ({_age_str(s.last_run_at)})"
                    if s.last_run_at is not None
                    else "never"
                ),
            }
            for s in stats
        ]
        ui.table(columns=columns, rows=rows, row_key="decoder").classes("w-full")


def _decoders_display(decoders_json: str | None) -> str:
    """`["a","b"]` -> `"a, b"`, tolerant of a null/malformed cell."""
    if not decoders_json:
        return ""
    try:
        return ", ".join(json.loads(decoders_json))
    except (json.JSONDecodeError, TypeError):
        return decoders_json


def _recent_packets_table(decoded_packets_path: Path) -> None:
    recent = status_data.latest_packets(decoded_packets_path, n=RECENT_PACKETS_LIMIT)
    with ui.card().classes("w-full"):
        ui.label("Most Recent Packets").classes("text-lg font-bold")
        if recent.is_empty():
            ui.label("Nothing decoded yet.").classes("text-caption text-grey")
            return

        columns = [
            {"name": "received_at", "label": "Received", "field": "received_at"},
            {"name": "packet_type", "label": "Type", "field": "packet_type"},
            {"name": "csp_crc_valid", "label": "CRC OK?", "field": "csp_crc_valid"},
            {"name": "decoders", "label": "Decoders", "field": "decoders"},
            {"name": "general_message", "label": "Message", "field": "general_message"},
        ]
        rows = [
            {
                "received_at": f"{r['received_at']:%Y-%m-%d %H:%M:%S}",
                "packet_type": r.get("packet_type") or "",
                "csp_crc_valid": "yes" if r.get("csp_crc_valid") else "no",
                "decoders": _decoders_display(r.get("decoders")),
                "general_message": (r.get("general_message") or "")[:80],
            }
            for r in recent.to_dicts()
        ]
        table = ui.table(columns=columns, rows=rows, row_key="received_at").classes(
            "w-full"
        )
        table.add_slot(
            "body-cell-csp_crc_valid",
            r"""
                <q-td :props="props"
                    :class="props.value === 'yes' ? 'text-positive' : 'text-negative'">
                    {{ props.value }}
                </q-td>
            """,
        )


def build_pipeline_status_page(parquet_path: Path) -> None:
    output_dir = parquet_path.parent

    @ui.refreshable
    def content() -> None:
        counts = status_data.pipeline_counts(
            output_dir, decoded_packets_path=parquet_path
        )
        _counts_section(counts)
        _decoder_tools_table(status_data.decoder_tool_stats(output_dir))
        _recent_packets_table(parquet_path)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Pipeline Status").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=content.refresh)
        content()

    ui.timer(REFRESH_INTERVAL_SEC, content.refresh)
