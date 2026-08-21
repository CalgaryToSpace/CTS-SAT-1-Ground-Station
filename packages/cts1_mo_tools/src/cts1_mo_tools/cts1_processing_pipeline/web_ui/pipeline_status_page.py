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
from typing import TYPE_CHECKING

from nicegui import ui

from . import pipeline_status as status_data
from .layout import page_shell

if TYPE_CHECKING:
    import polars as pl

REFRESH_INTERVAL_SEC = 30.0

# Past these ages, a freshness timestamp is flagged yellow/red -- SatNOGS
# overpasses aren't constant, so a short gap is normal, but the pipeline
# genuinely stalling for hours is the thing this page exists to catch.
STALE_WARNING_AFTER = timedelta(minutes=90)
STALE_ERROR_AFTER = timedelta(hours=6)

RECENT_PACKETS_LIMIT = 25


def _duration_str(total_sec: int) -> str:
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m"
    if total_sec < 86400:  # noqa: PLR2004
        return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"
    return f"{total_sec // 86400}d {(total_sec % 86400) // 3600}h"


def _age_str(value: datetime) -> str:
    """ "X ago" for a past `value`, "in X" for a future one -- `value` can
    legitimately be in the future for `latest_observation_end`: step 1
    pulls every SatNOGS observation with `start < now`, not `end < now`,
    so an observation that's already started but hasn't finished yet
    (status "future"/in-progress) has an `end` still ahead of "now".
    """
    delta = (datetime.now(UTC) - value).total_seconds()
    if delta < 0:
        return f"in {_duration_str(int(-delta))}"
    return f"{_duration_str(int(delta))} ago"


def _source_icon(source: str) -> None:
    """A small info icon carrying a tooltip naming the pipeline step + table
    (+ column, where relevant) a value was pulled from, so a viewer can
    tell at a glance which stage to go investigate.
    """
    ui.icon("info", size="14px", color="grey").classes("cursor-help").tooltip(source)


def _freshness_row(label: str, value: datetime | None, *, source: str) -> None:
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-1"):
            ui.label(label).classes("text-caption text-grey text-uppercase")
            _source_icon(source)
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


def _plain_row(label: str, value: str | None, *, source: str) -> None:
    """Same layout as `_freshness_row`, but with no staleness coloring --
    for facts like "first packet ever" or "largest historical gap" that
    are just informational, not a live health signal to flag yellow/red.
    """
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-1"):
            ui.label(label).classes("text-caption text-grey text-uppercase")
            _source_icon(source)
        if value is None:
            with ui.row().classes("items-center gap-1"):
                ui.icon("help", color="grey")
                ui.label("no data yet").classes("text-caption text-grey")
            return
        ui.label(value)


def _first_packet_value(value: datetime | None) -> str | None:
    if value is None:
        return None
    return f"{value:%Y-%m-%d %H:%M:%S} UTC ({_age_str(value)})"


def _largest_gap_value(summary: status_data.PacketCompletenessSummary) -> str | None:
    if (
        summary.largest_gap is None
        or summary.largest_gap_from is None
        or summary.largest_gap_to is None
    ):
        return None
    return (
        f"{_duration_str(int(summary.largest_gap.total_seconds()))} "
        f"(from {summary.largest_gap_from:%Y-%m-%d %H:%M:%S} "
        f"to {summary.largest_gap_to:%Y-%m-%d %H:%M:%S} UTC)"
    )


def _counts_section(
    counts: status_data.PipelineCounts,
    completeness: status_data.PacketCompletenessSummary,
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Pipeline Counts").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            _stat_tile("Observations", f"{counts.total_observations:,}")
            _stat_tile("Raw decodes", f"{counts.total_raw_decodes:,}")
            _stat_tile("Distinct packets", f"{counts.total_distinct_packets:,}")
            _stat_tile("Decoded packets", f"{counts.total_decoded_packets:,}")
            _stat_tile("Satellite events", f"{counts.total_satellite_events:,}")
        if counts.decode_backlog:
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.icon("warning", color="warning")
                ui.label(
                    f"{counts.decode_backlog:,} distinct packet(s) haven't been "
                    "run through step 3 (decoding) yet."
                ).classes("text-warning")

        ui.separator().classes("mt-2")
        with ui.row().classes("w-full gap-8 flex-wrap items-start mt-2"):
            with ui.column().classes("flex-1 min-w-72 gap-0"):
                ui.label("Freshness").classes("text-base font-medium")
                _freshness_row(
                    "Latest observation end (SatNOGS)",
                    counts.latest_observation_end,
                    source="Step 1 · raw_observations.parquet · end",
                )
                _freshness_row(
                    "Latest packet received",
                    counts.latest_packet_received_at,
                    source="Step 1 · raw_packets.parquet · received_at",
                )
                _freshness_row(
                    "Latest packet ingested locally",
                    counts.latest_packet_ingested_at,
                    source="Step 1 · raw_packets.parquet · ingested_at",
                )
                _freshness_row(
                    "Latest decoder run",
                    counts.latest_decoder_run_at,
                    source="Step 1 · decoder_runs.parquet · run_at",
                )
                _freshness_row(
                    "Latest decode (step 3)",
                    counts.latest_decoded_at,
                    source=(
                        "Step 3 · everything_decoded.parquet · file "
                        "last-written time (rewritten whole on every run, "
                        "so no per-row timestamp)"
                    ),
                )
                _freshness_row(
                    "Latest satellite events run (step 4)",
                    counts.latest_satellite_events_at,
                    source=(
                        "Step 4 · satellite_events_from_beacons.parquet · "
                        "file last-written time (rewritten whole on every "
                        "run, so no per-row timestamp)"
                    ),
                )

            with ui.column().classes("flex-1 min-w-72 gap-0"):
                ui.label("Completeness").classes("text-base font-medium")
                _plain_row(
                    "First packet ever received",
                    _first_packet_value(completeness.first_received_at),
                    source="Step 3 · everything_decoded.parquet · received_at",
                )
                _plain_row(
                    "Largest gap (entire history)",
                    _largest_gap_value(completeness),
                    source="Step 3 · everything_decoded.parquet · received_at",
                )


def _decoder_tools_table(stats: list[status_data.DecoderToolStats]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Decoder Tool Scoreboard").classes("text-lg font-bold")
        ui.label(
            '"Distinct Packets" is each tool\'s share of every distinct '
            "(post-dedup) packet ever seen -- the meaningful score, since "
            '"Raw Decodes" double-counts the same packet across every '
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


def _recent_packets_table(recent: pl.DataFrame) -> None:
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


def build_pipeline_status_page(data_dir: Path) -> None:
    parquet_path = data_dir / status_data.DECODED_PACKETS_FILENAME

    @ui.refreshable
    def content() -> None:
        counts = status_data.pipeline_counts(data_dir)
        completeness = status_data.packet_completeness_summary(parquet_path)
        _counts_section(counts, completeness)
        _decoder_tools_table(status_data.decoder_tool_stats(data_dir))
        _recent_packets_table(
            status_data.latest_packets(parquet_path, n=RECENT_PACKETS_LIMIT)
        )

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Pipeline Status").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=content.refresh)
        content()

    ui.timer(REFRESH_INTERVAL_SEC, content.refresh)
