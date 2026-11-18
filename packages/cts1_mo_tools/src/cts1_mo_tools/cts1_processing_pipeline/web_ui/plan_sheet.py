"""`PlanSheet`: the Plan page's spreadsheet-style table -- see `plan_sheet.js`.

Its library (Jspreadsheet CE, MIT) and stylesheets load from jsDelivr.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["PlanSheet"]

from typing import Any

from nicegui import ui

_STYLESHEETS = (
    "https://cdn.jsdelivr.net/npm/jspreadsheet-ce@5.0.4/dist/jspreadsheet.css",
    "https://cdn.jsdelivr.net/npm/jspreadsheet-ce@5.0.4/dist/jspreadsheet.themes.css",
    # The version jsDelivr's jspreadsheet-ce@5.0.4 bundle imports.
    "https://cdn.jsdelivr.net/npm/jsuites@5.12.0/dist/jsuites.css",
)

# Monospaced, like the Excel download; colours follow NiceGUI's dark mode.
_CSS = """
.plan-sheet .jss_worksheet { font-family: ui-monospace, Consolas, monospace; }
.plan-sheet .jss_content { box-shadow: none !important; }
.body--dark .plan-sheet .jss_worksheet td.readonly { color: #999; }
.body--dark .plan-sheet {
    --border_color: #555;
    --header_background: #2a2a2a;
    --header_color: #ddd;
    --header_background_highlighted: #3c3c3c;
    --header_color_highlighted: #fff;
    --content_background: #1d1d1d;
    --content_color: #ddd;
    --selection: rgba(255, 255, 255, 0.08);
    --border_color_highlighted: #90caf9;
}
"""


class PlanSheet(ui.element, component="plan_sheet.js"):
    """A spreadsheet of rows (lists of cell values, a hidden id first).

    `columns` are Jspreadsheet column options (each with a `name`), the first
    `{"type": "hidden", "name": "id"}`. Edits arrive as a `cells_changed`
    event: a list of `{id, row, field, value}`, `id` null for a new row.
    """

    def __init__(self, columns: list[dict[str, Any]]) -> None:
        super().__init__()
        for href in _STYLESHEETS:
            ui.add_head_html(f'<link rel="stylesheet" href="{href}">')
        ui.add_css(_CSS)
        self.classes("plan-sheet w-full")
        self.props["columns"] = columns
        self.props["rows"] = []
        self.rows: list[list[Any]] = []

    def set_rows(self, rows: list[list[Any]]) -> None:
        self.rows = rows
        self.props["rows"] = rows
        self.update()

    async def selected_ids(self) -> list[int]:
        """Ids of the rows with a selected cell."""
        return await self.run_method("selectedIds")
