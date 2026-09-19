"""The "Browse Packets" page: a wide, Excel-like grid over every column of
`everything_decoded.parquet`, packet-type/time-range/message-substring
filters and sorting applied server-side, a details panel for whichever row
was clicked (every field, each copyable), plus CSV/Excel exports of the
filtered set -- see `packet_browser` for the actual querying/paging/export
logic.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_packet_browser_page"]

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING

import polars as pl
from nicegui import ui

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

from . import packet_browser
from .layout import page_shell

if TYPE_CHECKING:
    from collections.abc import Callable

RANGE_INPUT_MASK = "####-##-## ##:##:##"
RANGE_INPUT_PLACEHOLDER = "YYYY-MM-DD HH:MM:SS"

PAGE_SIZE_OPTIONS = [50, 100, 200, 500]

# Pinned to the left and always visible, whatever the column finder says --
# a row is hard to make sense of without them.
ALWAYS_VISIBLE_COLUMNS = ("received_at", "packet_type")

# Auto-sized to their contents, but capped so one long general_message
# can't push every other column off-screen -- the details panel shows the
# full value.
MAX_COLUMN_WIDTH_PX = 420

# `navigator.clipboard` only exists in a secure context (HTTPS or
# localhost) -- fall back to the old hidden-textarea trick so copy still
# works when the dashboard is reached over plain HTTP on the LAN.
_COPY_JS = """
(async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return true;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  return ok;
})(%s)
"""

# Every non-numeric/boolean/string column (Datetime, in practice) gets
# stringified before it's handed to the grid -- AG Grid's JSON payload has
# no native datetime type, and this keeps its formatting consistent with
# the rest of the dashboard's `%Y-%m-%d %H:%M:%S` timestamps.
_GRID_PASSTHROUGH_DTYPES = (pl.Boolean, pl.String)


def _grid_ready(df: pl.DataFrame) -> list[dict[str, object]]:
    exprs = [
        pl.col(name).dt.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(dtype, pl.Datetime)
        else pl.col(name).cast(pl.String)
        for name, dtype in df.schema.items()
        if dtype not in _GRID_PASSTHROUGH_DTYPES and not dtype.is_numeric()
    ]
    if exprs:
        df = df.with_columns(exprs)
    return df.to_dicts()


def _parse_range_input(value: str | None) -> datetime | None:
    """Parse a `RANGE_INPUT_MASK`-shaped value ("YYYY-MM-DD HH:MM:SS"),
    treated as UTC -- see `file_reassembler_page._parse_range_input` for why
    a masked text input is used instead of a native datetime picker.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _copy_to_clipboard(text: str, *, what: str) -> None:
    ui.run_javascript(_COPY_JS % json.dumps(text))
    ui.notify(f"Copied {what}.", type="positive", timeout=1500)


def _column_matches(column: str, query: str) -> bool:
    """Whether the column finder should show `column` -- any of the query's
    whitespace-separated terms appearing in it (case-insensitively) counts.
    """
    terms = query.lower().split()
    return (
        not terms
        or column in ALWAYS_VISIBLE_COLUMNS
        or any(term in column.lower() for term in terms)
    )


def _pretty_json(value: object) -> str | None:
    """`value` pretty-printed, if it's a string holding a JSON list/object
    (the step 2 traceability columns, `sources` etc.) -- None otherwise.
    """
    if not isinstance(value, str) or not value.startswith(("[", "{")):
        return None
    try:
        return json.dumps(json.loads(value), indent=2)
    except ValueError:
        return None


@dataclass
class _FilterState:
    """The (mutable) filter/sort/paging state every widget on this page edits."""

    packet_types: list[str] = field(default_factory=list)  # empty = every type
    start: str | None = None
    end: str | None = None
    message_substring: str = ""
    case_sensitive: bool = False
    page_size: int = PAGE_SIZE_OPTIONS[1]
    offset: int = 0
    sort: packet_browser.PacketSort = field(default_factory=packet_browser.PacketSort)
    column_query: str = ""

    def to_filters(self) -> packet_browser.PacketBrowserFilters:
        return packet_browser.PacketBrowserFilters(
            packet_types=tuple(self.packet_types) or None,
            start=_parse_range_input(self.start),
            end=_parse_range_input(self.end),
            message_substring=self.message_substring or None,
            case_sensitive=self.case_sensitive,
        )


def _filters_section(
    state: _FilterState, packet_types: list[str], *, on_search: Callable[[], None]
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Filters").classes("text-lg font-bold")
        ui.label(
            "Applied server-side before any rows are sent to the browser. "
            "Press Enter in any text box (or Search) to apply; packet type "
            "and rows/page apply immediately."
        ).classes("text-caption text-grey")

        def _set_and_search(attr: str, value: object) -> None:
            setattr(state, attr, value)
            on_search()

        with ui.row().classes("w-full items-end gap-4 flex-wrap mt-2"):
            ui.select(
                packet_types,
                multiple=True,
                label="Packet type(s) (all if empty)",
                value=state.packet_types,
                on_change=lambda e: _set_and_search("packet_types", e.value or []),
            ).classes("w-64").props("use-chips dense clearable")

            ui.input(
                "Start (UTC)",
                value=state.start or "",
                placeholder=RANGE_INPUT_PLACEHOLDER,
                on_change=lambda e: setattr(state, "start", e.value),
            ).props(f'mask="{RANGE_INPUT_MASK}" clearable').classes("w-52").on(
                "keydown.enter", on_search
            )
            ui.input(
                "End (UTC)",
                value=state.end or "",
                placeholder=RANGE_INPUT_PLACEHOLDER,
                on_change=lambda e: setattr(state, "end", e.value),
            ).props(f'mask="{RANGE_INPUT_MASK}" clearable').classes("w-52").on(
                "keydown.enter", on_search
            )

            ui.input(
                "Message contains",
                value=state.message_substring,
                on_change=lambda e: setattr(state, "message_substring", e.value or ""),
            ).props("clearable").classes("w-64").on("keydown.enter", on_search)
            ui.checkbox(
                "Case-sensitive",
                value=state.case_sensitive,
                on_change=lambda e: setattr(state, "case_sensitive", bool(e.value)),
            )
            ui.select(
                PAGE_SIZE_OPTIONS,
                label="Rows/page",
                value=state.page_size,
                on_change=lambda e: _set_and_search("page_size", e.value),
            ).classes("w-28")

            ui.button("Search", icon="search", on_click=on_search)


def _field_row(name: str, value: object) -> None:
    """One field of the details panel: name, full (selectable) value, and a
    copy button.
    """
    text = "" if value is None else str(value)
    pretty = _pretty_json(value)
    with ui.row().classes("w-full items-start gap-2 flex-nowrap py-1"):
        with ui.column().classes("gap-0 flex-1 min-w-0"):
            ui.label(name).classes("text-caption text-grey font-mono")
            if pretty is not None:
                ui.code(pretty, language="json").classes(
                    "w-full text-xs max-h-64 overflow-auto"
                )
            else:
                ui.label(text).classes(
                    "font-mono text-sm break-all whitespace-pre-wrap select-text"
                )
        ui.button(
            icon="content_copy",
            on_click=lambda: _copy_to_clipboard(text, what=name),
        ).props("flat dense round size=sm").tooltip(f"Copy {name}")


def _notify_export(*, row_count: int, truncated: bool) -> None:
    if row_count == 0:
        ui.notify("No packets match these filters.", type="warning")
        return
    message = f"Exported {row_count:,} packet(s)."
    if truncated:
        message += (
            f" Capped at {packet_browser.MAX_EXPORT_ROWS:,} rows -- narrow "
            "the filters for a complete export."
        )
    ui.notify(message, type="warning" if truncated else "positive")


def build_packet_browser_page(data_dir: Path) -> None:  # noqa: C901, PLR0915
    parquet_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    packet_types = packet_browser.packet_type_options(parquet_path)
    state = _FilterState()
    # The grid currently on screen (replaced on every refresh), and the row
    # the details panel is showing -- kept across page turns/searches, since
    # it's a snapshot of that one row rather than a view into the grid.
    grid_ref: dict[str, ui.aggrid | None] = {"grid": None}
    detail: dict[str, dict[str, object] | None] = {"row": None}
    detail_query: dict[str, str] = {"text": ""}

    @ui.refreshable
    def grid_section() -> None:
        grid_ref["grid"] = None
        filters = state.to_filters()
        page = packet_browser.load_page(
            parquet_path,
            filters,
            offset=state.offset,
            limit=state.page_size,
            sort=state.sort,
        )
        if page.total_rows == 0:
            ui.label("No packets match these filters.").classes(
                "text-caption text-grey"
            )
            return

        total_pages = -(-page.total_rows // state.page_size)  # ceil div
        current_page = state.offset // state.page_size + 1
        has_prev = state.offset > 0
        has_next = state.offset + state.page_size < page.total_rows

        with ui.row().classes("w-full items-center justify-between"):
            ui.label(
                f"{state.offset + 1:,}-"
                f"{min(state.offset + state.page_size, page.total_rows):,} of "
                f"{page.total_rows:,} packet(s), sorted by {state.sort.column} "
                f"({'desc' if state.sort.descending else 'asc'}) -- "
                f"{len(page.columns)} column(s) (all-null columns hidden for "
                "this filter)."
            ).classes("text-caption text-grey")
            with ui.row().classes("items-center gap-1"):
                for icon, target, enabled, tip in (
                    ("first_page", 1, has_prev, "First page"),
                    ("chevron_left", current_page - 1, has_prev, "Previous page"),
                ):
                    ui.button(
                        icon=icon, on_click=lambda _, t=target: _go_to_page(t)
                    ).props("flat dense round").tooltip(tip).set_enabled(enabled)
                ui.label(f"Page {current_page:,} / {total_pages:,}").classes("mx-1")
                for icon, target, enabled, tip in (
                    ("chevron_right", current_page + 1, has_next, "Next page"),
                    ("last_page", total_pages, has_next, "Last page"),
                ):
                    ui.button(
                        icon=icon, on_click=lambda _, t=target: _go_to_page(t)
                    ).props("flat dense round").tooltip(tip).set_enabled(enabled)

        column_defs = [
            {
                "field": col,
                # AG Grid auto-title-cases the header when headerName isn't
                # given -- these are raw column names (snake_case, units
                # baked in like "_C"/"_V"), not prose, so title-casing them
                # just mangles them.
                "headerName": col,
                "headerTooltip": col,
                "pinned": "left" if col in ALWAYS_VISIBLE_COLUMNS else None,
                "hide": not _column_matches(col, state.column_query),
                "sort": (
                    ("desc" if state.sort.descending else "asc")
                    if col == state.sort.column
                    else None
                ),
            }
            for col in page.columns
        ]
        grid = (
            ui.aggrid(
                {
                    "columnDefs": column_defs,
                    "rowData": _grid_ready(page.rows),
                    "defaultColDef": {
                        "sortable": True,
                        "resizable": True,
                        "filter": False,
                        "maxWidth": MAX_COLUMN_WIDTH_PX,
                    },
                    "autoSizeStrategy": {"type": "fitCellContents"},
                    # Plain mouse text selection inside cells, so any value
                    # can be highlighted and Ctrl+C'd straight off the grid.
                    "enableCellTextSelection": True,
                    "ensureDomOrder": True,
                    "rowSelection": {
                        "mode": "singleRow",
                        "checkboxes": False,
                        "enableClickSelection": True,
                    },
                    # Only one column sorts at a time, server-side --
                    # shift-click multi-sort would be silently ignored.
                    "multiSortKey": "none",
                    "rowHeight": 26,
                    "headerHeight": 30,
                    "tooltipShowDelay": 300,
                    "suppressFieldDotNotation": True,
                },
                auto_size_columns=False,
            )
            .classes("w-full")
            .style("height: 75vh")
        )
        # Explicit arg lists: NiceGUI otherwise serializes every event field,
        # and AG Grid's `context` is a circular object that JSON.stringify
        # chokes on -- so the event never reaches the server at all.
        grid.on("rowClicked", lambda e: _show_detail(e.args.get("data")), ["data"])
        grid.on("sortChanged", _on_sort_changed, [])
        grid_ref["grid"] = grid

    @ui.refreshable
    def detail_panel() -> None:
        row = detail["row"]
        if row is None:
            return
        present = {k: v for k, v in row.items() if v is not None and v != ""}
        with (
            ui.card()
            .classes("w-[30rem] shrink-0 gap-2")
            .style("height: calc(75vh + 3rem); overflow-y: auto")
        ):
            with ui.row().classes("w-full items-center justify-between flex-nowrap"):
                ui.label("Packet details").classes("text-lg font-bold")
                with ui.row().classes("items-center gap-1"):
                    ui.button(
                        "JSON",
                        icon="content_copy",
                        on_click=lambda: _copy_to_clipboard(
                            json.dumps(present, indent=2), what="row as JSON"
                        ),
                    ).props("flat dense").tooltip("Copy every field as JSON")
                    ui.button(icon="close", on_click=lambda: _show_detail(None)).props(
                        "flat dense round"
                    ).tooltip("Close")
            ui.input(
                "Filter fields",
                value=detail_query["text"],
                on_change=lambda e: _set_detail_query(e.value or ""),
            ).props("dense clearable").classes("w-full")
            detail_fields()

    @ui.refreshable
    def detail_fields() -> None:
        row = detail["row"] or {}
        terms = detail_query["text"].lower().split()
        shown = 0
        with ui.column().classes("w-full gap-0 divide-y"):
            for name, value in row.items():
                if value is None or value == "":
                    continue
                haystack = f"{name} {value}".lower()
                if terms and not all(term in haystack for term in terms):
                    continue
                _field_row(name, value)
                shown += 1
        if shown == 0:
            ui.label("No fields match.").classes("text-caption text-grey")

    def _show_detail(row: dict[str, object] | None) -> None:
        detail["row"] = row
        detail_panel.refresh()

    def _set_detail_query(text: str) -> None:
        detail_query["text"] = text
        detail_fields.refresh()

    async def _on_sort_changed() -> None:
        grid = grid_ref["grid"]
        if grid is None:
            return
        column_state = await grid.run_grid_method("getColumnState")
        sorted_cols = [c for c in column_state or [] if c.get("sort")]
        new_sort = (
            packet_browser.PacketSort(
                column=sorted_cols[0]["colId"],
                descending=sorted_cols[0]["sort"] == "desc",
            )
            if sorted_cols
            else packet_browser.PacketSort()
        )
        if new_sort == state.sort:
            return  # e.g. the grid echoing back the sort it was built with
        state.sort = new_sort
        state.offset = 0
        grid_section.refresh()

    def _apply_column_query(query: str) -> None:
        state.column_query = query
        grid = grid_ref["grid"]
        if grid is None:
            return
        columns = [c["field"] for c in grid.options["columnDefs"]]
        shown = [c for c in columns if _column_matches(c, query)]
        hidden = [c for c in columns if c not in shown]
        # AG Grid's JS API takes the visibility flag positionally.
        grid.run_grid_method("setColumnsVisible", shown, True)  # noqa: FBT003
        if hidden:
            grid.run_grid_method("setColumnsVisible", hidden, False)  # noqa: FBT003

    def _go_to_page(page_number: int) -> None:
        state.offset = max(0, (page_number - 1) * state.page_size)
        grid_section.refresh()

    def _search() -> None:
        state.offset = 0
        grid_section.refresh()

    def _export_csv() -> None:
        result = packet_browser.export_filtered_csv(parquet_path, state.to_filters())
        if result.row_count:
            ui.download.content(result.csv_bytes, filename="packets.csv")
        _notify_export(row_count=result.row_count, truncated=result.truncated)

    def _export_excel() -> None:
        result = packet_browser.export_filtered_excel(parquet_path, state.to_filters())
        if result.row_count:
            ui.download.content(result.xlsx_bytes, filename="packets.xlsx")
        _notify_export(row_count=result.row_count, truncated=result.truncated)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Browse Packets").classes("text-2xl font-bold")
            with ui.row().classes("items-center gap-2"):
                ui.button("Export CSV", icon="download", on_click=_export_csv)
                ui.button(
                    "Export Excel", icon="download", on_click=_export_excel
                ).props("outline")
        ui.label(
            "Every decoded packet, every column -- filtering and sorting "
            "happen server-side, so this stays responsive no matter how much "
            "history has piled up. Drag to select text in any cell to copy "
            "it, or click a row to see (and copy) all of its fields."
        ).classes("text-caption text-grey")

        _filters_section(state, packet_types, on_search=_search)

        ui.input(
            'Find columns (e.g. "temp volt")',
            on_change=lambda e: _apply_column_query(e.value or ""),
        ).props("dense clearable").classes("w-80").tooltip(
            "Show only columns whose name contains any of these words "
            "(received_at and packet_type always stay visible)."
        )

        with ui.row().classes("w-full items-start gap-4 flex-nowrap"):
            with ui.column().classes("flex-1 min-w-0 gap-2"):
                grid_section()
            detail_panel()
