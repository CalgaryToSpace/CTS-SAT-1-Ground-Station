"""A small "pipeline running" indicator for the navbar: a spinner plus the
daemon's progress through its current step, shown only while the daemon is
actually processing, and linking to the Pipeline Status page for details.

Reads the same `daemon_status.json` heartbeat the Pipeline Status page
does -- see `cts1_processing_pipeline.daemon_signals`.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_nav_pipeline_indicator"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nicegui import ui

from cts1_mo_tools.cts1_processing_pipeline import daemon_signals

if TYPE_CHECKING:
    from pathlib import Path

PIPELINE_STATUS_PATH = "/pipeline-status"

# Matched to the daemon's own heartbeat: polling faster than it writes
# would only ever re-read the same file.
POLL_INTERVAL_SEC = daemon_signals.HEARTBEAT_INTERVAL_SEC

# Classes for the indicator's wordier parts ("Pipeline running", the
# progress bar): hidden on phones, and also from `xl` up, where the navbar's
# full link row appears (see `main._nav`) and leaves no room for them --
# just the spinner and count show there.
_WIDE_ONLY = "max-md:hidden xl:hidden"


@dataclass(frozen=True, slots=True)
class _View:
    """Everything the indicator renders -- compared between polls so the
    navbar is only rebuilt (and re-sent to the browser) when it changes.
    """

    detail: str | None
    progress: daemon_signals.Progress | None


def _read_view(data_dir: Path) -> _View | None:
    """What to show, or None to show nothing (daemon not processing)."""
    status = daemon_signals.read_status(data_dir)
    if status is None or not status.is_processing:
        return None
    return _View(detail=status.detail, progress=status.progress)


def build_nav_pipeline_indicator(data_dir: Path) -> None:
    """Add the indicator at the current position (inside the header), and
    keep it updated on a timer for as long as the page is open.
    """

    @ui.refreshable
    def indicator(view: _View | None) -> None:
        if view is None:
            return
        tooltip_parts = ["Pipeline is running"]
        if view.detail:
            tooltip_parts.append(view.detail)
        if view.progress is not None:
            tooltip_parts.append(str(view.progress))
        with ui.link(target=PIPELINE_STATUS_PATH).classes("text-white no-underline"):
            with ui.row().classes("items-center no-wrap gap-2"):
                ui.spinner(size="1.25em", color="white")
                # Full wording where the navbar has room for it; just the
                # spinner and count where it doesn't -- see `_WIDE_ONLY`.
                ui.label("Pipeline running").classes(_WIDE_ONLY + " text-body2")
                if view.progress is not None:
                    ui.label(f"{view.progress.done}/{view.progress.total}").classes(
                        "text-body2"
                    )
                    ui.linear_progress(
                        value=view.progress.fraction,
                        show_value=False,
                        color="white",
                    ).classes(_WIDE_ONLY + " w-16")
            ui.tooltip(" · ".join(tooltip_parts))

    last_view = _read_view(data_dir)
    indicator(last_view)

    def _poll() -> None:
        nonlocal last_view
        view = _read_view(data_dir)
        if view != last_view:
            last_view = view
            indicator.refresh(view)

    ui.timer(POLL_INTERVAL_SEC, _poll)
