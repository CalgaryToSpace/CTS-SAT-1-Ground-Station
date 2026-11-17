"""The "Export Data" page: a picker for which of the pipeline's tables to
export, common filters (packet type, received-at range), a choice of output
format, and (at the bottom) a flow chart of every table the pipeline
produces -- see `export_tables` for the actual filtering/export logic.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_export_page"]

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere

import polars as pl
from nicegui import run, ui

from . import export_tables
from .export_raw import raw_export_url, raw_export_urls
from .layout import page_shell

# A flowchart of every table the pipeline produces and the step that
# produces it -- kept in sync with `export_tables.TABLE_SPECS` and
# `daemon.STEP_NAMES` by hand (checked by `test_export_page`), rather than
# generated from them, since the diagram also needs to show each step's
# inputs and the external sources (SatNOGS, Hugging Face), which aren't
# tables at all. Laid out top-to-bottom: there are too many steps for a
# left-to-right layout to stay readable at page width.
PIPELINE_FLOWCHART = """
flowchart TD
    satnogs[("SatNOGS API")] --> step0["Step 0\\nList Observations"]
    step0 --> raw_observations[("raw_observations")]
    step0 --> observation_listing_windows[("observation_listing_windows")]
    step0 --> observation_listing_history[("observation_listing_history")]
    raw_observations --> step1["Step 1\\nDownload & Demodulate"]
    satnogs --> step1
    step1 --> raw_packets[("raw_packets")]
    step1 --> decoder_runs[("decoder_runs")]
    raw_observations --> step2["Step 2\\nDeduplicate Packets"]
    raw_packets --> step2
    step2 --> distinct_packets[("distinct_packets_over_time")]
    distinct_packets --> step3["Step 3\\nDecode Packets"]
    step3 --> everything_decoded[("everything_decoded")]
    everything_decoded --> step4["Step 4\\nDetect Satellite Events"]
    step4 --> satellite_events[("satellite_events_from_beacons")]
    everything_decoded --> step5["Step 5\\nReassemble TCMD Responses"]
    step5 --> reassembled_tcmd_responses[("reassembled_tcmd_responses")]
    everything_decoded --> step6["Step 6\\nExtract MPI Frames"]
    step6 --> mpi_frames[("mpi_frames")]
    huggingface[("Hugging Face\\nSpace-Track TLEs")] --> step7["Step 7\\nFetch TLEs"]
    step7 --> tle_history[("tle_history")]
    mpi_frames --> step8["Step 8\\nSummarize MPI Collections"]
    tle_history --> step8
    reassembled_tcmd_responses --> step8
    step8 --> mpi_collections[("mpi_collections")]
    step8 --> mpi_analyses[("mpi_analyses")]
    mpi_frames --> step9["Step 9\\nClean MPI Frames"]
    step9 --> mpi_frames_cleaned[("mpi_frames_cleaned")]
    reassembled_tcmd_responses --> step10["Step 10\\nList Satellite Files"]
    step10 --> satellite_files[("satellite_files")]
    step10 --> satellite_file_mentions[("satellite_file_mentions")]
    everything_decoded --> step11["Step 11\\nReassemble Downlinked Files"]
    satellite_files --> step11
    satellite_file_mentions --> step11
    step11 --> file_downlink_time_periods[("file_downlink_time_periods")]
    step11 --> downlinked_files[("downlinked_files")]
    %% Invisible links holding step 7 (which only step 8 reads) up top with
    %% step 0 -- the layout otherwise pulls it down, right next to step 8.
    huggingface ~~~ step0
    step7 ~~~ raw_observations
    %% Colours: external sources grey, steps blue, output tables green.
    classDef source fill:#eceff1,stroke:#607d8b,color:#263238
    classDef step fill:#bbdefb,stroke:#1565c0,color:#0d47a1,font-weight:bold
    classDef output fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    class satnogs,huggingface source
    class step0,step1,step2,step3,step4,step5,step6,step7,step8,step9,step10,step11 step
    class raw_observations,observation_listing_windows output
    class observation_listing_history,raw_packets,decoder_runs output
    class distinct_packets,everything_decoded output
    class satellite_events,reassembled_tcmd_responses,mpi_frames,tle_history output
    class mpi_collections,mpi_analyses,mpi_frames_cleaned output
    class satellite_files,satellite_file_mentions output
    class file_downlink_time_periods,downlinked_files output
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


def _tables_section(selected: dict[str, bool], selected_types: dict[str, bool]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Tables").classes("text-lg font-bold")
        ui.label(
            "Pick tables to export. Each table's ⚡ link streams "
            "the entire table as a parquet file."
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
                        key, bool(e.value)
                    ),
                ).classes("mt-1")
                with ui.column().classes("gap-0"):
                    with ui.row().classes("items-baseline gap-2"):
                        ui.label(spec.key).classes("font-mono font-medium")
                        ui.link(
                            "⚡ direct parquet download", raw_export_url(spec.key)
                        ).classes("text-caption")
                    ui.label(spec.description).classes("text-caption text-grey")
                    if spec.has_packet_type:
                        _packet_type_filter(selected_types)


def _packet_type_filter(selected_types: dict[str, bool]) -> None:
    """Checkboxes picking which packet types a table with a `packet_type`
    column is filtered to, with one button to select or deselect them all.
    """
    checkboxes: list[ui.checkbox] = []

    def _all_selected() -> bool:
        return all(selected_types.values())

    def _on_change(packet_type: str, value: bool) -> None:  # noqa: FBT001
        selected_types[packet_type] = value
        toggle_all.set_text("Deselect all" if _all_selected() else "Select all")

    def _toggle_all() -> None:
        value = not _all_selected()
        for checkbox in checkboxes:
            checkbox.value = value  # Updates `selected_types`, via `_on_change`.

    with ui.column().classes("gap-1 mt-2"):
        with ui.row().classes("items-center gap-2"):
            ui.label("Packet type filter").classes("text-sm font-medium")
            toggle_all = ui.button(
                "Deselect all" if _all_selected() else "Select all",
                on_click=_toggle_all,
            ).props("dense flat no-caps size=sm")
        with ui.row().classes("gap-x-4 gap-y-0 flex-wrap"):
            checkboxes.extend(
                ui.checkbox(
                    packet_type,
                    value=selected_types[packet_type],
                    on_change=lambda e, pt=packet_type: _on_change(pt, bool(e.value)),
                ).props("dense")
                for packet_type in export_tables.PACKET_TYPES
            )


def _time_filter_section(time_state: dict[str, str | None]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Time range filter").classes("text-lg font-bold")

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


def duckdb_bootstrap_sql(
    base_url: str,
    columns_by_table: dict[export_tables.TableSpec, list[str]],
    urls_by_table: dict[str, list[str]],
) -> str:
    """DuckDB statements creating a view per table over its raw parquet
    download URLs (`urls_by_table`, by table key), then selecting from each,
    with a `-- <step>` comment above each step's tables. An empty column
    list selects `*`.
    """
    views: list[str] = []
    selects: list[str] = []
    current_step: str | None = None
    for spec, columns in columns_by_table.items():
        if spec.step != current_step:
            current_step = spec.step
            if views:  # A blank line between steps' groups.
                views.append("")
                selects.append("")
            views.append(f"-- {spec.step}")
            selects.append(f"-- {spec.step}")
        views.append(
            f"CREATE OR REPLACE VIEW {spec.key} AS SELECT * FROM "  # noqa: S608 -- shown, never run
            "read_parquet(["
            + ", ".join(
                f"'{base_url.rstrip('/')}{url}'" for url in urls_by_table[spec.key]
            )
            + "]);"
        )
        selects.append(
            "SELECT "  # noqa: S608 -- shown, never run
            + (", ".join(f'"{c}"' for c in columns) if columns else "*")
            + f" FROM {spec.key};"
        )
    return "\n".join([*views, "", *selects])


def _collapsible(title: str) -> ui.element:
    """A collapsed-by-default section, its header styled like the cards' titles.

    A native <details> rather than `ui.expansion`, whose open state lives on
    the server -- every toggle waited on a websocket round trip.
    """
    details = ui.element("details").classes("w-full group")
    with (
        details,
        ui.element("summary").classes(
            "text-lg font-bold cursor-pointer list-none flex items-center gap-1"
        ),
    ):
        ui.icon("chevron_right").classes("transition-transform group-open:rotate-90")
        ui.label(title)
    return details


def _duckdb_section(data_dir: Path) -> None:
    request = ui.context.client.request
    # Behind a TLS-terminating reverse proxy the app itself only sees http.
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    base_url = f"{scheme}://{request.url.netloc}"
    # Only tables already produced -- the rest would 404. Reads parquet footers only.
    columns = {
        spec: pl.scan_parquet(data_dir / spec.filename).collect_schema().names()
        for spec in export_tables.TABLE_SPECS
        if (data_dir / spec.filename).exists()
    }
    urls = {spec.key: raw_export_urls(spec.key, data_dir) for spec in columns}

    with ui.card().classes("w-full"), _collapsible("DuckDB Bootstrap"):
        ui.label(
            "Paste into DuckDB to query every table straight from this "
            "server (reads the ⚡ direct parquet download links)."
        ).classes("text-caption text-grey")
        list_columns = ui.checkbox("List all columns instead of *")

        @ui.refreshable
        def _code() -> None:
            ui.code(
                duckdb_bootstrap_sql(
                    base_url,
                    columns if list_columns.value else {spec: [] for spec in columns},
                    urls,
                ),
                language="sql",
            ).classes("w-full overflow-x-auto")

        _code()
        list_columns.on_value_change(_code.refresh)


@dataclass
class _ExportOptions:
    """The (mutable) export-format/zip/drop-nulls/lineage state `_options_section`
    edits -- typed fields instead of a `dict[str, object]` so each widget's
    `value=`/`on_change=` can stay bool/str-typed all the way through,
    rather than every read needing an `object` cast.
    """

    export_format: export_tables.ExportFormat = export_tables.EXPORT_FORMATS[0]
    zip_output: bool = False
    drop_null_columns: bool = True
    include_lineage_columns: bool = False


def _options_section(options: _ExportOptions) -> ui.card:
    """Returns the card, so the caller can put the Export button inside it."""
    with ui.card().classes("w-full") as card:
        ui.label("Export options").classes("text-lg font-bold")
        with ui.row().classes("items-center gap-6 flex-wrap mt-2"):
            ui.select(
                list(export_tables.EXPORT_FORMATS),
                value=options.export_format,
                label="Format",
                on_change=lambda e: setattr(options, "export_format", e.value),
            ).classes("w-40")
            ui.checkbox(
                "Zip output",
                value=options.zip_output,
                on_change=lambda e: setattr(options, "zip_output", bool(e.value)),
            ).tooltip(
                "CSV and Parquet write one file per table -- selecting more "
                "than one of them requires this. Excel/SQLite/DuckDB always "
                "bundle every selected table into a single file."
            )
            ui.checkbox(
                "Drop irrelevant columns (all-null after filtering)",
                value=options.drop_null_columns,
                on_change=lambda e: setattr(
                    options, "drop_null_columns", bool(e.value)
                ),
            )
            ui.checkbox(
                "Include lineage columns",
                value=options.include_lineage_columns,
                on_change=lambda e: setattr(
                    options, "include_lineage_columns", bool(e.value)
                ),
            ).tooltip(
                "Keep the huge per-packet traceability columns (sources, "
                "observation_ids) in distinct_packets_over_time and "
                "everything_decoded."
            )
    return card


def build_export_page(data_dir: Path) -> None:
    selected_tables: dict[str, bool] = {
        spec.key: spec.key == "everything_decoded" for spec in export_tables.TABLE_SPECS
    }
    selected_packet_types: dict[str, bool] = dict.fromkeys(
        export_tables.PACKET_TYPES, True
    )
    time_state: dict[str, str | None] = {"start": None, "end": None}
    options = _ExportOptions()

    async def _run_export() -> None:
        specs = [
            spec for spec in export_tables.TABLE_SPECS if selected_tables[spec.key]
        ]
        if not specs:
            ui.notify("Pick at least one table to export.", type="warning")
            return

        start = _parse_range_input(time_state["start"])
        end = _parse_range_input(time_state["end"])
        checked_types = [pt for pt, checked in selected_packet_types.items() if checked]
        if not checked_types and any(spec.has_packet_type for spec in specs):
            ui.notify("Pick at least one packet type to export.", type="warning")
            return
        packet_types = (
            checked_types
            if checked_types and len(checked_types) < len(export_tables.PACKET_TYPES)
            else None
        )

        # Off the event loop: a big export can take long enough that the
        # websocket misses its heartbeats, and NiceGUI shows "Reconnecting"
        # (or reloads the page outright) while it waits.
        export_button.disable()
        export_spinner.set_visibility(True)
        try:
            result = await run.io_bound(
                export_tables.export_selected_tables,
                specs,
                data_dir,
                start=start,
                end=end,
                packet_types=packet_types,
                drop_all_null_columns=options.drop_null_columns,
                include_lineage_columns=options.include_lineage_columns,
                export_format=options.export_format,
                zip_output=options.zip_output,
            )
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return
        finally:
            export_button.enable()
            export_spinner.set_visibility(False)
        if result is None:  # Only when the server's shutting down.
            return

        ui.download.content(result.data, filename=result.filename)
        ui.notify(
            f"Exported {result.filename} ({len(result.data):,} bytes).", type="positive"
        )

    with page_shell():
        ui.label("Export Data").classes("text-2xl font-bold")
        ui.label(
            "Export any of the pipeline's tables to a file, with optional "
            "time-range and packet-type filtering."
        ).classes("text-caption text-grey")

        _tables_section(selected_tables, selected_packet_types)
        _time_filter_section(time_state)
        with _options_section(options), ui.row().classes("items-center gap-2 mt-2"):
            export_button = ui.button("Export", icon="download", on_click=_run_export)
            export_spinner = ui.spinner(size="md")
            export_spinner.set_visibility(False)

        _duckdb_section(data_dir)

        with ui.card().classes("w-full"), _collapsible("Pipeline Overview"):
            ui.mermaid(PIPELINE_FLOWCHART).classes("w-full")
