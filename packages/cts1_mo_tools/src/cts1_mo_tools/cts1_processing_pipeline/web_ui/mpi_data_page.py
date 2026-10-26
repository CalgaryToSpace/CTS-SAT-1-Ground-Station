"""The "MPI Data" page: when the satellite collected each MPI frame,
alongside its beacons and GNSS samples, by its own clock, and a summary of
each MPI collection -- see `collection_timeline` and `mpi_collections` for
the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_mpi_data_page"]

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING, Any, cast

from nicegui import run, ui

from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)
from cts1_mo_tools.time_str import duration_str

from . import collection_timeline, mpi_collections
from .charts import collection_timeline_option
from .layout import page_shell

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl


def _collection_timeline_card(timeline: pl.DataFrame | None) -> None:
    """`timeline` is None while it's still loading."""
    with ui.card().classes("w-full"):
        ui.label("Collection Timeline").classes("text-lg font-bold")

        if timeline is None:
            with ui.row().classes("items-center gap-2 p-2"):
                ui.spinner(size="md")
                ui.label("Loading...").classes("text-grey")
            return
        option = collection_timeline_option(
            timeline,
            sources=collection_timeline.TIMELINE_SOURCES,
            title="",
        )
        if option is None:
            ui.label(
                "Nothing collected yet -- run the pipeline (step_1..step_3, and "
                "step_6 for MPI frames) first."
            ).classes("text-caption text-grey mt-2")
            return
        ui.echart(option).classes("w-full h-80")


# (group label, columns) for each group of the collections table, each
# column being (field, label, numeric) -- the groups get their own header row
# above the columns', so it's clear which come from the satellite's on-board
# analysis of the file and which from the frames actually downlinked.
_COLLECTION_COLUMN_GROUPS: tuple[tuple[str, tuple[tuple[str, str, bool], ...]], ...] = (
    (
        "Collection",
        (
            ("collection_started_at", "Started (UTC)", False),
            ("collection_ended_at", "Ended (UTC)", False),
            ("collection_duration_ms", "Duration", True),
            ("collection_time_source", "Times From", False),
        ),
    ),
    (
        "Orbit (SGP4)",
        (
            ("_start_position", "Start Lat/Lon", False),
            ("_end_position", "End Lat/Lon", False),
            ("sunlit", "Sunlit", False),
        ),
    ),
    (
        "On-Board Analysis (analyze_mpi_data)",
        (
            ("analysis_file", "File", False),
            ("analysis_file_size", "File Size", True),
            ("analysis_frame_count", "Frames", True),
            ("analysis_strong_signal_counts", "Strong Signals", False),
            ("analysis_count", "Analyses", True),
            ("analysis_sha256", "SHA-256", False),
            ("analysis_received_at", "Received (UTC)", False),
        ),
    ),
    (
        "Frames Received (Downlinked)",
        (
            ("frames_present", "Present", True),
            ("frames_expected", "Expected", True),
            ("frames_missing", "Missing", True),
            ("integration_periods", "Integration Period(s)", True),
            ("time_sync_sources", "Time Sync Source(s)", True),
            ("_time_sync_age", "Time Sync Age", True),
            ("_temperature_range", "MPI Temp (°C)", True),
            ("first_frame_counter", "First Counter", True),
            ("last_frame_counter", "Last Counter", True),
            ("largest_gap_frames", "Largest Gap (Frames)", True),
            ("largest_gap_started_at", "Largest Gap Start (UTC)", False),
            ("largest_gap_duration_ms", "Largest Gap Duration", True),
        ),
    ),
)
_COLLECTION_COLUMNS = tuple(
    column for _, columns in _COLLECTION_COLUMN_GROUPS for column in columns
)

# The table's header: a row of group labels, each spanning its columns, then
# the usual row of column labels.
_COLLECTION_HEADER_SLOT = (
    '<q-tr :props="props">'
    + "".join(
        f'<q-th colspan="{len(columns)}" class="text-center text-weight-bold"'
        f' style="border-left: 1px solid rgba(128, 128, 128, 0.4)">{label}</q-th>'
        for label, columns in _COLLECTION_COLUMN_GROUPS
    )
    + "</q-tr>"
    '<q-tr :props="props">'
    '<q-th v-for="col in props.cols" :key="col.name" :props="props">'
    "{{ col.label }}</q-th>"
    "</q-tr>"
)


def _seconds(duration_ms: int) -> str:
    return f"{duration_ms / 1000:,.2f} s"


# How the columns that need more than `str()` are shown.
_CELL_FORMATTERS: dict[str, Callable[[Any], Any]] = {
    "collection_duration_ms": lambda duration_ms: duration_str(duration_ms // 1000),
    "largest_gap_duration_ms": _seconds,
    "analysis_file_size": lambda size: f"{size:,} B",
    "analysis_sha256": lambda sha256: sha256[:12],
    "analysis_strong_signal_counts": mpi_collections.strong_signal_counts_text,
    "integration_periods": mpi_collections.integration_periods_text,
    "time_sync_sources": mpi_collections.time_sync_sources_text,
}


def _cell(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"{value:%Y-%m-%d %H:%M:%S.%f}"[:-3]
    formatter = _CELL_FORMATTERS.get(field)
    return formatter(value) if formatter else value


def _position_cell(row: dict[str, Any], prefix: str) -> str:
    return mpi_collections.position_text(
        row.get(f"{prefix}_latitude_deg"), row.get(f"{prefix}_longitude_deg")
    )


def _temperature_range_cell(row: dict[str, Any]) -> str:
    low, high = row.get("board_temperature_min_c"), row.get("board_temperature_max_c")
    if low is None or high is None:
        return ""
    return f"{low:.1f}" if low == high else f"{low:.1f} to {high:.1f}"


def _time_sync_age_cell(row: dict[str, Any]) -> str:
    """How long since the clock was last set, at the recording's start --
    e.g. "16m 5s" -- flagged if it was resynced during the recording, when
    the age is at the first frame received instead if that's where it was.
    """
    age_ms = row.get("ms_since_time_sync_at_start")
    if age_ms is None:
        return ""
    age = duration_str(age_ms // 1000)
    resyncs = row.get("time_resyncs_during") or 0
    if not resyncs:
        return age
    synced_at, started_at = row.get("time_synced_at"), row["collection_started_at"]
    resynced_first = (
        synced_at is not None
        and synced_at - started_at > step_8_pipeline.RESYNC_TOLERANCE
    )
    at = "first frame" if resynced_first else "start"
    times = "once" if resyncs == 1 else f"{resyncs} times"
    return f"⚠ resynced {times} during; {age} at {at}"


def _time_source_cell(row: dict[str, Any]) -> str:
    """ "Times From": the analysis's full action name when it's from one --
    e.g. "analyze_mpi_data_v1".
    """
    source = row.get("collection_time_source") or ""
    action = row.get("analysis_action")
    return action if source == "analysis" and action else source


def _property_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _analysis_details(analysis: dict[str, Any]) -> None:
    """Every property of one analysis response, as sent, with its strong
    signals (when there are any) as their own table.
    """
    properties = mpi_collections.response_properties(analysis["analysis_response_text"])
    strong_signals = properties.pop("strong_signals", None)
    rows = [
        {
            "property": "(telecommand sent)",
            "value": _cell("", analysis["analysis_tcmd_sent_at"]),
        },
        {
            "property": "(response received)",
            "value": _cell("", analysis["analysis_received_at"]),
        },
        *(
            {"property": key, "value": _property_value(value)}
            for key, value in properties.items()
        ),
    ]
    ui.table(
        columns=[
            {
                "name": "property",
                "label": "Property",
                "field": "property",
                "align": "left",
            },
            {"name": "value", "label": "Value", "field": "value", "align": "left"},
        ],
        rows=rows,
        row_key="property",
    ).classes("w-full").props("dense flat")
    if isinstance(strong_signals, list) and strong_signals:
        ui.label(f"Strong signals ({len(strong_signals):,})").classes("font-bold mt-2")
        signals = cast("list[Any]", strong_signals)
        downlinked = mpi_collections.strong_signals_downlinked_text(
            signals, analysis.get("analysis_strong_signal_frames_downlinked")
        )
        if all(isinstance(signal, dict) for signal in signals):
            keys = list(dict.fromkeys(k for signal in signals for k in signal))
            ui.table(
                columns=[
                    {"name": k, "label": k, "field": k, "align": "left"} for k in keys
                ]
                + [
                    {
                        "name": "_downlinked",
                        "label": "Downlinked",
                        "field": "_downlinked",
                        "align": "left",
                    }
                ],
                rows=[
                    {k: _property_value(signal.get(k)) for k in keys}
                    | {"_i": i, "_downlinked": downlinked[i]}
                    for i, signal in enumerate(signals)
                ],
                row_key="_i",
            ).classes("w-full").props("dense flat")
        else:
            ui.code(json.dumps(signals, indent=2), language="json").classes("w-full")


def _show_analyses(collection: dict[str, Any], analyses: pl.DataFrame) -> None:
    """A dialog with every distinct analysis of `collection`'s file."""
    matching = mpi_collections.analyses_of(analyses, collection)
    if not matching:
        ui.notify(
            "No on-board analysis for this collection."
            if collection.get("analysis_action") is None
            else "No analysis details -- rerun step 8 to write mpi_analyses.parquet."
        )
        return
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-5xl"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(collection.get("analysis_file") or "(unnamed file)").classes(
                "text-lg font-bold"
            )
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")
        ui.label(
            f"{len(matching)} distinct analysis response(s) of this file, the "
            "one the collection uses first (the latest version, then the "
            "largest file, then the earliest received)."
        ).classes("text-caption text-grey")
        for i, analysis in enumerate(matching):
            title = " · ".join(
                part
                for part in (
                    analysis["analysis_action"],
                    f"received {_cell('', analysis['analysis_received_at'])}",
                    f"≥{analysis['analysis_strong_threshold_dn']} DN"
                    if analysis["analysis_strong_threshold_dn"] is not None
                    else "",
                    f"{analysis['analysis_file_size']:,} B"
                    if analysis["analysis_file_size"] is not None
                    else "",
                    "used for this collection" if i == 0 else "",
                )
                if part
            )
            with ui.expansion(title, value=i == 0).classes("w-full"):
                _analysis_details(analysis)
    dialog.open()


def _export_buttons(collections: pl.DataFrame) -> None:
    """Download the whole table -- every column of `mpi_collections.parquet`,
    including the ones the page hides.
    """
    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Export CSV",
            icon="download",
            on_click=lambda: ui.download.content(
                mpi_collections.collections_csv_bytes(collections),
                filename="mpi_collections.csv",
            ),
        ).props("dense")
        ui.button(
            "Export Excel",
            icon="download",
            on_click=lambda: ui.download.content(
                mpi_collections.collections_xlsx_bytes(collections),
                filename="mpi_collections.xlsx",
            ),
        ).props("dense outline")


def _collections_card(
    collections: pl.DataFrame | None, analyses: pl.DataFrame | None
) -> None:
    """`collections`/`analyses` are None while they're still loading."""
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Collections").classes("text-lg font-bold")
            if collections is not None and not collections.is_empty():
                _export_buttons(collections)

        if collections is None or analyses is None:
            with ui.row().classes("items-center gap-2 p-2"):
                ui.spinner(size="md")
                ui.label("Loading...").classes("text-grey")
            return
        if collections.is_empty():
            ui.label(
                "No MPI collections yet -- run the pipeline (step_1..step_3, "
                "then step_6 through step_8) first."
            ).classes("text-caption text-grey mt-2")
            return

        columns = [
            {
                "name": field,
                "label": label,
                "field": field,
                "align": "right" if numeric else "left",
            }
            for field, label, numeric in _COLLECTION_COLUMNS
        ]
        collection_rows = collections.to_dicts()
        rows = [
            # `.get`: an `mpi_collections.parquet` from before analyses were merged in
            # has no analysis columns.
            {field: _cell(field, row.get(field)) for field, _, _ in _COLLECTION_COLUMNS}
            | {
                "collection_time_source": _time_source_cell(row),
                "_start_position": _position_cell(row, "start"),
                "_end_position": _position_cell(row, "end"),
                "_temperature_range": _temperature_range_cell(row),
                "_time_sync_age": _time_sync_age_cell(row),
                "_row_index": i,
            }
            for i, row in enumerate(collection_rows)
        ]
        table = ui.table(columns=columns, rows=rows, row_key="_row_index").classes(
            "w-full cursor-pointer"
        )
        table.props("dense flat")
        table.add_slot("header", _COLLECTION_HEADER_SLOT)
        # Its args are the click event, the row, and its index.
        table.on(
            "rowClick",
            lambda e: _show_analyses(
                collection_rows[e.args[1]["_row_index"]], analyses
            ),
        )
        ui.label(f"{collections.height:,} collection(s).").classes(
            "text-caption text-grey mt-2"
        )


@dataclass(frozen=True, slots=True)
class _PageData:
    timeline: pl.DataFrame
    collections: pl.DataFrame
    analyses: pl.DataFrame


def _load_page_data(data_dir: Path) -> _PageData:
    """Blocking (it may fetch the launch date) -- run it via `run.io_bound`."""
    return _PageData(
        timeline=collection_timeline.load_collection_timeline(data_dir),
        collections=mpi_collections.load_collections(data_dir),
        analyses=mpi_collections.load_analyses(data_dir),
    )


def build_mpi_data_page(data_dir: Path) -> None:
    @ui.refreshable
    def content(page_data: _PageData | None) -> None:
        _collection_timeline_card(page_data and page_data.timeline)
        _collections_card(
            page_data and page_data.collections, page_data and page_data.analyses
        )

    async def _reload() -> None:
        # Load in a thread, then rebuild synchronously -- see
        # `pipeline_status_page.build_pipeline_status_page._reload`. Loading
        # can also wait on SatNOGS DB, for the launch date.
        page_data = await run.io_bound(_load_page_data, data_dir)
        if page_data is None:  # Only when the server's shutting down.
            return
        content.refresh(page_data)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Data").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=_reload)
        content(None)

    ui.timer(0, _reload, once=True)
