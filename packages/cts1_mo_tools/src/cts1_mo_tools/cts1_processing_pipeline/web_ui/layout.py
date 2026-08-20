"""Shared page-chrome helpers used by every page in this web UI."""

from __future__ import annotations

__all__ = ["page_shell"]

from nicegui import ui


def page_shell() -> ui.column:
    """The column every page's content sits in: centered, capped width."""
    return ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4")
