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
import base64
import functools
import itertools
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING, Any, cast

import fracturedjson
from nicegui import run, ui

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    beacons as beacon_summary,
)
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)

from . import collection_timeline, mpi_collections, mpi_strip
from . import mpi_collection_fields as fields
from .attitude_sparkline import attitude_sparkline_svg
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


# The page-only plot columns -- pictures, not values the exports or the
# metadata panel could show. The attitude plot is an inline SVG (see
# `attitude_sparkline`); the raw MPI one a thumbnail PNG of the frames,
# linking to them in MPI Vis.
_ATTITUDE_PLOT_COLUMN = "_beacon_attitude_plot"
_THUMBNAIL_WIDTH = 140  # px, as wide as the attitude plot.
_RAW_MPI_PLOT_COLUMN = "_raw_mpi_plot"
_ATTITUDE_PLOT_CELL_SLOT = (
    '<q-td :props="props" style="padding-top: 0; padding-bottom: 0">'
    '<span v-html="props.row._beacon_attitude_plot"></span></q-td>'
)
_RAW_MPI_PLOT_CELL_SLOT = (
    '<q-td :props="props" style="padding-top: 0; padding-bottom: 0">'
    '<a v-if="props.row._raw_mpi_plot" :href="props.row._strip_href">'
    '<img :src="props.row._raw_mpi_plot" alt="Raw MPI frames" '
    # `max-width: none`: the page's base CSS caps images at their cell's width.
    f'style="display: block; width: {_THUMBNAIL_WIDTH}px; max-width: none; '
    'height: 36px">'
    "<q-tooltip>Raw MPI frames (pixels top to bottom, time left to right)"
    " -- view them in MPI Vis</q-tooltip></a></q-td>"
)

# The collections table's groups: the buttons and plots, then the shared
# ones -- each group gets its own header row above its columns', so it's
# clear which come from the satellite's on-board analysis of the file and
# which from the frames actually downlinked.
_COLLECTION_COLUMN_GROUPS = (
    fields.ColumnGroup(
        "",
        (
            fields.Column("_strip", "Vis"),
            fields.Column(_RAW_MPI_PLOT_COLUMN, "Raw MPI"),
            fields.Column(_ATTITUDE_PLOT_COLUMN, "Attitude Plot"),
        ),
    ),
    *fields.COLUMN_GROUPS,
)
_COLLECTION_COLUMNS = tuple(
    column for group in _COLLECTION_COLUMN_GROUPS for column in group.columns
)

# The leading column's button opens a collection's analyses -- a button
# rather than a row click, so the cells' text can be selected and copied.
_DETAILS_COLUMN = "_details"
_DETAILS_CELL_SLOT = r"""
    <q-td :props="props">
        <q-btn
            flat round dense size="sm" icon="info"
            @click="$parent.$emit('show_analyses', props.row._row_index)"
        >
            <q-tooltip>On-board analysis details</q-tooltip>
        </q-btn>
    </q-td>
"""

# The columns that stay put when the table scrolls sideways -- the buttons
# and plots, leftmost -- each with its (fixed) width in px.
_PINNED_WIDTHS = {
    _DETAILS_COLUMN: 44,
    "_strip": 52,
    _RAW_MPI_PLOT_COLUMN: _THUMBNAIL_WIDTH + 16,  # Plus the cell's padding.
    _ATTITUDE_PLOT_COLUMN: _THUMBNAIL_WIDTH + 16,
}
# Each one's offset from the left: the widths of those before it.
_PINNED_LEFTS = dict(
    zip(
        _PINNED_WIDTHS,
        itertools.accumulate(_PINNED_WIDTHS.values(), initial=0),
        strict=False,  # `accumulate` gives one more: the total.
    )
)
_PINNED_CLASS = "mpi-pinned"
# Opaque, so what scrolls under them doesn't show through.
_PINNED_CSS = f"""
.{_PINNED_CLASS} {{ background: #fff; }}
.body--dark .{_PINNED_CLASS} {{ background: var(--q-dark); }}
"""


def _pinned_style(field: str, *, header: bool = False) -> str:
    """The CSS pinning `field`'s cells at their place from the left, the
    last pinned column edged so it's clear the rest scrolls under it.
    """
    width = _PINNED_WIDTHS[field]
    edge = (
        "; box-shadow: inset -1px 0 0 rgba(128, 128, 128, 0.5)"
        if field == list(_PINNED_WIDTHS)[-1]
        else ""
    )
    return (
        f"position: sticky; left: {_PINNED_LEFTS[field]}px; "
        f"z-index: {3 if header else 2}; "
        f"width: {width}px; min-width: {width}px; max-width: {width}px{edge}"
    )


def _column_def(field: str, label: str, *, numeric: bool = False) -> dict[str, Any]:
    column: dict[str, Any] = {
        "name": field,
        "label": label,
        "field": field,
        "align": "right" if numeric else "left",
    }
    if field in _PINNED_WIDTHS:
        column |= {
            "classes": _PINNED_CLASS,
            "headerClasses": _PINNED_CLASS,
            "style": _pinned_style(field),
            "headerStyle": _pinned_style(field, header=True),
        }
    return column


# The table's header: a row of group labels, each spanning its columns, then
# the usual row of column labels. The buttons-and-plots group is pinned like
# its columns.
_LEADING_GROUP_WIDTH = sum(_PINNED_WIDTHS.values()) - _PINNED_WIDTHS[_DETAILS_COLUMN]
_COLLECTION_HEADER_SLOT = (
    '<q-tr :props="props">'
    f'<q-th class="{_PINNED_CLASS}" '
    f'style="{_pinned_style(_DETAILS_COLUMN, header=True)}"></q-th>'
    f'<q-th colspan="{len(_COLLECTION_COLUMN_GROUPS[0].columns)}" '
    f'class="{_PINNED_CLASS}" style="position: sticky; z-index: 3; '
    f"left: {_PINNED_WIDTHS[_DETAILS_COLUMN]}px; width: {_LEADING_GROUP_WIDTH}px; "
    'box-shadow: inset -1px 0 0 rgba(128, 128, 128, 0.5)"></q-th>'
    + "".join(
        f'<q-th colspan="{len(group.columns)}" class="text-center text-weight-bold"'
        f' style="border-left: 1px solid rgba(128, 128, 128, 0.4)">{group.label}</q-th>'
        for group in _COLLECTION_COLUMN_GROUPS[1:]
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
    plots: list[dict[str, str]] | None,
) -> None:
    """`collections`/`analyses`/`plots` (each collection's plot columns'
    cells, in order) are None while they're still loading.
    """
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Collections").classes("text-lg font-bold")
            if collections is not None and not collections.is_empty():
                _export_buttons(collections)

        if collections is None or analyses is None or plots is None:
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

        columns = [_column_def(_DETAILS_COLUMN, "")] + [
            _column_def(column.field, column.label, numeric=column.numeric)
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
            | plots[i]
            for i, row in enumerate(collection_rows)
        ]
        table = ui.table(columns=columns, rows=rows, row_key="_row_index").classes(
            "w-full"
        )
        table.props("dense flat")
        table.add_slot("header", _COLLECTION_HEADER_SLOT)
        table.add_slot(f"body-cell-{_DETAILS_COLUMN}", _DETAILS_CELL_SLOT)
        table.add_slot("body-cell-_strip", _STRIP_CELL_SLOT)
        table.add_slot(f"body-cell-{_ATTITUDE_PLOT_COLUMN}", _ATTITUDE_PLOT_CELL_SLOT)
        table.add_slot(f"body-cell-{_RAW_MPI_PLOT_COLUMN}", _RAW_MPI_PLOT_CELL_SLOT)
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
    # Each collection's plot columns' cells (an SVG, or a PNG's data URI; ""
    # for none), in `collections` order.
    plots: list[dict[str, str]]


@functools.lru_cache(maxsize=1)
def _raw_mpi_plots(
    data_dir: Path, frames_mtime: float, collections_mtime: float
) -> list[str]:
    """Each collection's raw frames thumbnail as a data URI (or "") -- see
    `mpi_strip.raw_thumbnails`. Slow-ish (it decodes every frame), so kept
    until either file changes (the mtimes are only the cache key).
    """
    del frames_mtime, collections_mtime
    return [
        f"data:image/png;base64,{base64.b64encode(png).decode()}" if png else ""
        for png in mpi_strip.raw_thumbnails(
            data_dir, mpi_collections.load_collections(data_dir), _THUMBNAIL_WIDTH
        )
    ]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _load_collections_data(data_dir: Path) -> _CollectionsData:
    """Blocking -- run it via `run.io_bound`."""
    collections = mpi_collections.load_collections(data_dir)
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    attitude_plots = [""] * collections.height
    if decoded_path.exists() and not collections.is_empty():
        beacons = beacon_summary.load_beacons(decoded_path, collections)
        attitude_plots = [
            attitude_sparkline_svg(beacons, start, end)
            for start, end in collections.select(
                "collection_started_at", "collection_ended_at"
            ).iter_rows()
        ]
    raw_mpi_plots = _raw_mpi_plots(
        data_dir,
        _mtime(data_dir / step_6_pipeline.OUTPUT_FILENAME),
        _mtime(data_dir / step_8_pipeline.COLLECTIONS_OUTPUT_FILENAME),
    )
    if len(raw_mpi_plots) != collections.height:  # Rewritten in between.
        raw_mpi_plots = [""] * collections.height
    return _CollectionsData(
        collections=collections,
        analyses=mpi_collections.load_analyses(data_dir),
        plots=[
            {_ATTITUDE_PLOT_COLUMN: attitude, _RAW_MPI_PLOT_COLUMN: raw}
            for attitude, raw in zip(attitude_plots, raw_mpi_plots, strict=True)
        ],
    )


def build_mpi_data_page(data_dir: Path) -> None:
    ui.add_css(_PINNED_CSS)

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
            collections_data and collections_data.plots,
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
