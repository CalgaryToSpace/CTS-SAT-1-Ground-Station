"""The "Export Data" page: a flow chart of every table the pipeline
produces, a picker for which of them to export, common filters (packet
type, received-at range), and a choice of output format -- see
`export_tables` for the actual filtering/export logic.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_export_page"]

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere

from nicegui import ui

from . import export_tables
from .export_raw import raw_export_url
from .layout import page_shell

# A flowchart of every table the pipeline produces and the step that
# produces it -- kept in sync with `export_tables.TABLE_SPECS` by hand
# (there are only 3 steps/5 tables total, and the step boundaries rarely
# change), rather than generated from it, since the diagram also needs to
# show the SatNOGS API as the ultimate source, which isn't a table at all.
PIPELINE_FLOWCHART = """
flowchart LR
    satnogs[("SatNOGS API")] --> step1["Step 1\\nDownload & Demodulate"]
    step1 --> raw_observations[("raw_observations")]
    step1 --> raw_packets[("raw_packets")]
    step1 --> decoder_runs[("decoder_runs")]
    raw_observations --> step2["Step 2\\nDeduplicate Packets"]
    raw_packets --> step2
    step2 --> distinct_packets[("distinct_packets_over_time")]
    distinct_packets --> step3["Step 3\\nDecode Packets"]
    step3 --> everything_decoded[("everything_decoded")]
"""

RANGE_INPUT_MASK = "####-##-## ##:##:##"
RANGE_INPUT_PLACEHOLDER = "YYYY-MM-DD HH:MM:SS"
RANGE_INPUT_FORMAT = "%Y-%m-%d %H:%M:%S"

# label -> lookback duration; "All time" clears the range entirely.
TIME_PRESETS: dict[str, timedelta | None] = {
    "All time": None,
    "Last 12h": timedelta(hours=12),
    "Last 24h": timedelta(hours=24),
    "Last 3d": timedelta(days=3),
    "Last 7d": timedelta(days=7),
    "Last 30d": timedelta(days=30),
}


def _format_range_input(value: datetime) -> str:
    return value.strftime(RANGE_INPUT_FORMAT)


def _parse_range_input(value: str | None) -> datetime | None:
    """Parse a `RANGE_INPUT_MASK`-shaped value ("YYYY-MM-DD HH:MM:SS"),
    treated as UTC -- see `file_reassembler_page._parse_range_input` for
    why a masked text input is used instead of a native datetime picker.

    Returns None for anything that doesn't (yet) parse.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _tables_section(selected: dict[str, bool]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Tables").classes("text-lg font-bold")
        ui.label(
            "Pick one or more tables to export below, ordered by which "
            "pipeline step produces them. Each table's ⚡ link streams "
            "its parquet file exactly as the pipeline wrote it, straight "
            "off disk -- no filtering, no format conversion, no row limit."
        ).classes("text-caption text-grey")

        current_step: str | None = None
        for spec in export_tables.TABLE_SPECS:
            if spec.step != current_step:
                current_step = spec.step
                ui.label(spec.step).classes(
                    "text-caption text-grey text-uppercase mt-3"
                )
            with ui.row().classes("items-start gap-2 w-full flex-nowrap"):
                ui.checkbox(
                    value=selected[spec.key],
                    on_change=lambda e, key=spec.key: selected.__setitem__(
                        key, e.value
                    ),
                ).classes("mt-1")
                with ui.column().classes("gap-0"):
                    with ui.row().classes("items-baseline gap-2"):
                        ui.label(spec.key).classes("font-mono font-medium")
                        ui.link(
                            "⚡ direct parquet download", raw_export_url(spec.key)
                        ).classes("text-caption")
                    ui.label(spec.description).classes("text-caption text-grey")


def _packet_type_filter(selected_types: dict[str, bool]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Packet type filter").classes("text-lg font-bold")
        ui.label(
            "Only applies to tables with a packet_type column (currently "
            "just everything_decoded) -- ignored for every other table. "
            "Leave every box checked (or uncheck them all) to include "
            "every packet type."
        ).classes("text-caption text-grey")
        with ui.row().classes("gap-4 flex-wrap mt-2"):
            for packet_type in export_tables.PACKET_TYPES:
                ui.checkbox(
                    packet_type,
                    value=selected_types[packet_type],
                    on_change=lambda e, pt=packet_type: selected_types.__setitem__(
                        pt, e.value
                    ),
                )


def _time_filter_section(time_state: dict[str, str | None]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Time range filter").classes("text-lg font-bold")
        ui.label(
            "Applied to each table's own timestamp column -- received_at "
            "for most tables, start for raw_observations, run_at for "
            "decoder_runs."
        ).classes("text-caption text-grey")

        start_input: ui.input
        end_input: ui.input

        def _apply_preset(delta: timedelta | None) -> None:
            if delta is None:
                time_state["start"] = None
                time_state["end"] = None
            else:
                now = datetime.now(UTC)
                time_state["start"] = _format_range_input(now - delta)
                time_state["end"] = _format_range_input(now)
            start_input.value = time_state["start"] or ""
            end_input.value = time_state["end"] or ""

        with ui.row().classes("gap-2 flex-wrap mt-2"):
            for label, delta in TIME_PRESETS.items():
                ui.button(
                    label, on_click=lambda _, delta=delta: _apply_preset(delta)
                ).props("outline dense")

        with ui.row().classes("items-center gap-2 mt-3"):
            start_input = (
                ui.input(
                    "Start (UTC)",
                    value=time_state["start"] or "",
                    placeholder=RANGE_INPUT_PLACEHOLDER,
                    on_change=lambda e: time_state.__setitem__("start", e.value),
                )
                .props(f'mask="{RANGE_INPUT_MASK}"')
                .classes("w-56")
            )
            end_input = (
                ui.input(
                    "End (UTC)",
                    value=time_state["end"] or "",
                    placeholder=RANGE_INPUT_PLACEHOLDER,
                    on_change=lambda e: time_state.__setitem__("end", e.value),
                )
                .props(f'mask="{RANGE_INPUT_MASK}"')
                .classes("w-56")
            )


def _options_section(options_state: dict[str, object]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Export options").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-6 flex-wrap mt-2"):
            ui.select(
                list(export_tables.EXPORT_FORMATS),
                value=options_state["format"],
                label="Format",
                on_change=lambda e: options_state.__setitem__("format", e.value),
            ).classes("w-40")
            ui.checkbox(
                "Zip output",
                value=options_state["zip"],
                on_change=lambda e: options_state.__setitem__("zip", e.value),
            )
            ui.checkbox(
                "Drop irrelevant columns (all-null after filtering)",
                value=options_state["drop_null"],
                on_change=lambda e: options_state.__setitem__("drop_null", e.value),
            )
        ui.label(
            "CSV and Parquet write one file per table -- selecting more "
            'than one of them requires "Zip output". Excel/SQLite/DuckDB '
            "always bundle every selected table into a single file."
        ).classes("text-caption text-grey mt-2")


def build_export_page(parquet_path: Path) -> None:
    output_dir = parquet_path.parent

    selected_tables: dict[str, bool] = {
        spec.key: spec.key == export_tables.TABLE_SPECS[-1].key
        for spec in export_tables.TABLE_SPECS
    }
    selected_packet_types: dict[str, bool] = dict.fromkeys(
        export_tables.PACKET_TYPES, True
    )
    time_state: dict[str, str | None] = {"start": None, "end": None}
    options_state: dict[str, object] = {
        "format": export_tables.EXPORT_FORMATS[0],
        "zip": False,
        "drop_null": True,
    }

    def _run_export() -> None:
        specs = [
            spec for spec in export_tables.TABLE_SPECS if selected_tables[spec.key]
        ]
        if not specs:
            ui.notify("Pick at least one table to export.", type="warning")
            return

        start = _parse_range_input(time_state["start"])
        end = _parse_range_input(time_state["end"])
        checked_types = [pt for pt, checked in selected_packet_types.items() if checked]
        packet_types = (
            checked_types
            if checked_types and len(checked_types) < len(export_tables.PACKET_TYPES)
            else None
        )

        try:
            result = export_tables.export_selected_tables(
                specs,
                output_dir,
                start=start,
                end=end,
                packet_types=packet_types,
                drop_all_null_columns=bool(options_state["drop_null"]),
                export_format=options_state["format"],  # type: ignore[arg-type]
                zip_output=bool(options_state["zip"]),
            )
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return

        ui.download.content(result.data, filename=result.filename)
        ui.notify(f"Exported {result.filename} ({len(result.data):,} bytes).", type="positive")

    with page_shell():
        ui.label("Export Data").classes("text-2xl font-bold")
        ui.label(
            "Export any of the pipeline's tables to a file, with optional "
            "time-range and packet-type filtering."
        ).classes("text-caption text-grey")

        with ui.card().classes("w-full"):
            ui.label("Pipeline Overview").classes("text-lg font-bold")
            ui.mermaid(PIPELINE_FLOWCHART).classes("w-full")

        _tables_section(selected_tables)
        _packet_type_filter(selected_packet_types)
        _time_filter_section(time_state)
        _options_section(options_state)

        ui.button("Export", icon="download", on_click=_run_export).classes("mt-2")
