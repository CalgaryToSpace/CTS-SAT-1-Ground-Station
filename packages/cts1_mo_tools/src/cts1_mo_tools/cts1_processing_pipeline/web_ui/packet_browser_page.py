"""The "Browse Packets" page: a wide, Excel-like grid over every column of
`everything_decoded.parquet`, packet-type/time-range/message-substring
filters applied server-side, plus a CSV export of the filtered set -- see
`packet_browser` for the actual querying/paging/export logic.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_packet_browser_page"]

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


@dataclass
class _FilterState:
    """The (mutable) filter/paging state every widget on this page edits."""

    packet_types: list[str] = field(default_factory=list)  # empty = every type
    start: str | None = None
    end: str | None = None
    message_substring: str = ""
    case_sensitive: bool = False
    page_size: int = PAGE_SIZE_OPTIONS[1]
    offset: int = 0

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
            "Every filter below is applied server-side before any rows are "
            "sent to the browser."
        ).classes("text-caption text-grey")

        with ui.row().classes("w-full items-end gap-4 flex-wrap mt-2"):
            ui.select(
                packet_types,
                multiple=True,
                label="Packet type(s) (all if empty)",
                value=state.packet_types,
                on_change=lambda e: setattr(state, "packet_types", e.value or []),
            ).classes("w-64").props("use-chips dense")

            ui.input(
                "Start (UTC)",
                value=state.start or "",
                placeholder=RANGE_INPUT_PLACEHOLDER,
                on_change=lambda e: setattr(state, "start", e.value),
            ).props(f'mask="{RANGE_INPUT_MASK}"').classes("w-48")
            ui.input(
                "End (UTC)",
                value=state.end or "",
                placeholder=RANGE_INPUT_PLACEHOLDER,
                on_change=lambda e: setattr(state, "end", e.value),
            ).props(f'mask="{RANGE_INPUT_MASK}"').classes("w-48")

            ui.input(
                "Message contains",
                value=state.message_substring,
                on_change=lambda e: setattr(state, "message_substring", e.value or ""),
            ).classes("w-64")
            ui.checkbox(
                "Case-sensitive",
                value=state.case_sensitive,
                on_change=lambda e: setattr(state, "case_sensitive", bool(e.value)),
            )
            ui.select(
                PAGE_SIZE_OPTIONS,
                label="Rows/page",
                value=state.page_size,
                on_change=lambda e: setattr(state, "page_size", e.value),
            ).classes("w-28")

            ui.button("Search", icon="search", on_click=on_search)


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


def build_packet_browser_page(data_dir: Path) -> None:
    parquet_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    packet_types = packet_browser.packet_type_options(parquet_path)
    state = _FilterState()

    @ui.refreshable
    def grid_section() -> None:
        filters = state.to_filters()
        page = packet_browser.load_page(
            parquet_path, filters, offset=state.offset, limit=state.page_size
        )
        if page.total_rows == 0:
            ui.label("No packets match these filters.").classes(
                "text-caption text-grey"
            )
            return

        total_pages = -(-page.total_rows // state.page_size)  # ceil div
        current_page = state.offset // state.page_size + 1

        with ui.row().classes("w-full items-center justify-between"):
            ui.label(
                f"{state.offset + 1:,}-"
                f"{min(state.offset + state.page_size, page.total_rows):,} of "
                f"{page.total_rows:,} packet(s) -- {len(page.columns)} column(s) "
                "shown (all-null columns hidden for this filter)."
            ).classes("text-caption text-grey")
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="chevron_left", on_click=lambda: _turn_page(-1)).props(
                    "flat dense round"
                ).set_enabled(state.offset > 0)
                ui.label(f"Page {current_page} / {total_pages}")
                ui.button(icon="chevron_right", on_click=lambda: _turn_page(1)).props(
                    "flat dense round"
                ).set_enabled(state.offset + state.page_size < page.total_rows)

        column_defs = [
            {
                "field": col,
                # AG Grid auto-title-cases the header when headerName isn't
                # given -- these are raw column names (snake_case, units
                # baked in like "_C"/"_V"), not prose, so title-casing them
                # just mangles them.
                "headerName": col,
                "pinned": "left" if col == "received_at" else None,
            }
            for col in page.columns
        ]
        ui.aggrid(
            {
                "columnDefs": column_defs,
                "rowData": _grid_ready(page.rows),
                "defaultColDef": {
                    "sortable": True,
                    "resizable": True,
                    "filter": False,
                },
                "rowHeight": 24,
                "headerHeight": 28,
                "suppressFieldDotNotation": True,
            },
            auto_size_columns=False,
        ).classes("w-full").style("height: 75vh")

    def _turn_page(delta: int) -> None:
        state.offset = max(0, state.offset + delta * state.page_size)
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
            "Every decoded packet, every column -- filtering happens "
            "server-side, so this stays responsive no matter how much "
            "history has piled up."
        ).classes("text-caption text-grey")

        _filters_section(state, packet_types, on_search=_search)
        grid_section()
