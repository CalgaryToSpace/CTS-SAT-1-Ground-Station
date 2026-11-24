"""The "MPI Data" page: when the satellite collected each MPI frame,
alongside its beacons and GNSS samples, by its own clock, and a summary of
each MPI collection -- see `collection_timeline` and `mpi_collections` for
the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_mpi_data_page"]

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING, Any, cast

import fracturedjson
from nicegui import run, ui

from . import collection_timeline, mpi_collections
from . import mpi_collection_fields as fields
from .charts import collection_timeline_option
from .layout import page_shell
from .mpi_strip_view import collection_key, vis_page_url

if TYPE_CHECKING:
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
            bin_label=f"{collection_timeline.BIN_SECONDS} sec",
        )
        if option is None:
            ui.label(
                "Nothing collected yet -- run the pipeline (step_1..step_3, and "
                "step_6 for MPI frames) first."
            ).classes("text-caption text-grey mt-2")
            return
        ui.echart(option).classes("w-full h-80")


# The collections table's groups: a link to the strip, then the shared ones
# -- each group gets its own header row above its columns', so it's clear
# which come from the satellite's on-board analysis of the file and which
# from the frames actually downlinked.
_COLLECTION_COLUMN_GROUPS = (
    fields.ColumnGroup("", (fields.Column("_strip", "Vis"),)),
    *fields.COLUMN_GROUPS,
)
_COLLECTION_COLUMNS = tuple(
    column for group in _COLLECTION_COLUMN_GROUPS for column in group.columns
)

# The leading column's button opens a collection's analyses -- a button
# rather than a row click, so the cells' text can be selected and copied.
_DETAILS_COLUMN = "_details"
_DETAILS_CELL_SLOT = r"""
    <q-td :props="props" auto-width>
        <q-btn
            flat round dense size="sm" icon="info"
            @click="$parent.$emit('show_analyses', props.row._row_index)"
        >
            <q-tooltip>On-board analysis details</q-tooltip>
        </q-btn>
    </q-td>
"""

# The table's header: a row of group labels, each spanning its columns, then
# the usual row of column labels.
_COLLECTION_HEADER_SLOT = (
    '<q-tr :props="props">'
    "<q-th></q-th>"
    + "".join(
        f'<q-th colspan="{len(group.columns)}" class="text-center text-weight-bold"'
        f' style="border-left: 1px solid rgba(128, 128, 128, 0.4)">{group.label}</q-th>'
        for group in _COLLECTION_COLUMN_GROUPS
    )
    + "</q-tr>"
    '<q-tr :props="props">'
    '<q-th v-for="col in props.cols" :key="col.name" :props="props">'
    "{{ col.label }}</q-th>"
    "</q-tr>"
)

# A link to the collection's frames as a strip, on the MPI Vis page (see
# `mpi_strip_view`), for a collection with any frames downlinked.
_STRIP_CELL_SLOT = (
    '<q-td :props="props">'
    '<q-btn v-if="props.row.frames_present" flat round dense icon="view_week" '
    'type="a" :href="props.row._strip_href">'
    "<q-tooltip>View its frames in MPI Vis</q-tooltip></q-btn>"
    "</q-td>"
)


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
            "value": fields.cell_text("", analysis["analysis_tcmd_sent_at"]),
        },
        {
            "property": "(response received)",
            "value": fields.cell_text("", analysis["analysis_received_at"]),
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
            ui.code(
                fracturedjson.dumps(signals, indent=2, line_length=60), language="json"
            ).classes("w-full")


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
                    "received "
                    + fields.cell_text("", analysis["analysis_received_at"]),
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
    """Download the whole table as the page shows it, grouped and
    labelled alike -- every field of `mpi_collections.parquet`, including
    the ones the page hides (as in the MPI Vis page's metadata panel).
    """
    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Export CSV",
            icon="download",
            on_click=lambda: ui.download.content(
                fields.collections_csv_bytes(collections),
                filename="mpi_collections.csv",
            ),
        ).props("dense")
        ui.button(
            "Export Excel",
            icon="download",
            on_click=lambda: ui.download.content(
                fields.collections_xlsx_bytes(collections),
                filename="mpi_collections.xlsx",
            ),
        ).props("dense outline")


def _collections_card(
    collections: pl.DataFrame | None,
    analyses: pl.DataFrame | None,
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

        columns = [{"name": _DETAILS_COLUMN, "label": "", "field": _DETAILS_COLUMN}] + [
            {
                "name": column.field,
                "label": column.label,
                "field": column.field,
                "align": "right" if column.numeric else "left",
            }
            for column in _COLLECTION_COLUMNS
        ]
        collection_rows = collections.to_dicts()
        rows = [
            fields.collection_cells(row)
            | {
                "_strip_href": vis_page_url(
                    collection_key(row["collection_started_at"])
                ),
                "_row_index": i,
            }
            for i, row in enumerate(collection_rows)
        ]
        table = ui.table(columns=columns, rows=rows, row_key="_row_index").classes(
            "w-full"
        )
        table.props("dense flat")
        table.add_slot("header", _COLLECTION_HEADER_SLOT)
        table.add_slot(f"body-cell-{_DETAILS_COLUMN}", _DETAILS_CELL_SLOT)
        table.add_slot("body-cell-_strip", _STRIP_CELL_SLOT)
        # The event's arg is the row's `_row_index`.
        table.on(
            "show_analyses",
            lambda e: _show_analyses(collection_rows[e.args], analyses),
        )
        ui.label(f"{collections.height:,} collection(s).").classes(
            "text-caption text-grey mt-2"
        )


@dataclass(frozen=True, slots=True)
class _CollectionsData:
    collections: pl.DataFrame
    analyses: pl.DataFrame


def _load_collections_data(data_dir: Path) -> _CollectionsData:
    """Blocking -- run it via `run.io_bound`."""
    return _CollectionsData(
        collections=mpi_collections.load_collections(data_dir),
        analyses=mpi_collections.load_analyses(data_dir),
    )


def build_mpi_data_page(data_dir: Path) -> None:

    # The timeline and the collections table load and render independently,
    # so the (small, quick) table doesn't wait on the (big) timeline.
    @ui.refreshable
    def timeline_content(timeline: pl.DataFrame | None) -> None:
        _collection_timeline_card(timeline)

    @ui.refreshable
    def collections_content(collections_data: _CollectionsData | None) -> None:
        _collections_card(
            collections_data and collections_data.collections,
            collections_data and collections_data.analyses,
        )

    # Each loads in a thread, then rebuilds synchronously -- see
    # `pipeline_status_page.build_pipeline_status_page._reload`. A None
    # result is only when the server's shutting down.
    async def _reload_timeline() -> None:
        # Can also wait on SatNOGS DB, for the launch date.
        timeline = await run.io_bound(
            collection_timeline.load_collection_timeline, data_dir
        )
        if timeline is not None:
            timeline_content.refresh(timeline)

    async def _reload_collections() -> None:
        collections_data = await run.io_bound(_load_collections_data, data_dir)
        if collections_data is None:
            return
        collections_content.refresh(collections_data)

    async def _reload() -> None:
        await asyncio.gather(_reload_timeline(), _reload_collections())

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Data").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=_reload)
        timeline_content(None)
        collections_content(None)

    ui.timer(0, _reload, once=True)
