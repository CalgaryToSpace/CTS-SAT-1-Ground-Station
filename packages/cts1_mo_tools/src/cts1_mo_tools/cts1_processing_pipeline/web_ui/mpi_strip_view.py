"""The MPI Data page's strip view of one MPI collection: a dialog with the
collection's frames as a long strip -- see `mpi_strip` for the image
itself -- with a readout of whatever frame, pixel and housekeeping value
is under the mouse.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["show_mpi_strip"]

import contextlib
import html
import math
from typing import TYPE_CHECKING, Any

from nicegui import run, ui

from . import mpi_strip
from .preview_image_route import preview_image_url

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from nicegui.events import GenericEventArguments

# Image pixels across per frame -- or "fit": the whole strip scaled to the
# dialog's width (horizontally only, so the detector rows stay legible).
_FIT = "fit"
_ZOOMS: dict[int | str, str] = {_FIT: "Fit width", 1: "1x", 2: "2x", 4: "4x", 8: "8x"}
_DEFAULT_ZOOM = _FIT
# "Fit width" renders the strip at least this wide (if it can), so the
# browser scales it down to fit, averaging columns, rather than blurring up.
_FIT_RENDER_WIDTH_PX = 2560
# Roughly how many time-axis labels a fit-width strip gets.
_FIT_TICK_COUNT = 10
# Time-axis labels closer than this fraction of their usual spacing are
# skipped, so they don't overlap.
_MIN_TICK_GAP_FRACTION = 0.7
_MODES: dict[mpi_strip.IntensityMode, str] = {
    "raw": "Raw DN",
    "minus_median": "Minus each pixel's median",
}
_RANGES: dict[mpi_strip.IntensityRange, str] = {
    "typical": "Scale to typical frames",
    "full": "Scale to all frames",
}
# Roughly how far apart (in screen pixels) the time axis's labels are.
_TICK_SPACING_PX = 160
_LABEL_COLUMN_WIDTH = "17rem"


def _time_text(value: Any) -> str:
    return f"{value:%H:%M:%S.%f}"[:-3]


def _band_value_text(band: mpi_strip.Band, value: Any) -> str:
    if value is None:
        return f"{band.label}: ?"
    unit = f" {band.unit}" if band.unit else ""
    return f"{band.label}: {value:{band.format}}{unit}"


def _readout_lines(
    strip: mpi_strip.MpiStrip,
    frames: list[dict[str, Any]],
    mode: mpi_strip.IntensityMode,
    image_x: float,
    image_y: float,
) -> list[tuple[str, bool]] | None:
    """What's under the mouse at (`image_x`, `image_y`), for display: one
    `(text, is_under_mouse)` per line, the detector pixel or band the mouse
    is directly over marked. Lines stay in a fixed order (and the pixel line
    is always there), so the list doesn't jump about as the mouse moves.
    """
    column = strip.column_at(image_x)
    if column is None:
        return None
    if column.kind == "separator":
        return [("Between two recordings merged into this collection.", False)]
    if column.kind == "missing" or column.frame_index is None:
        return [(f"Frame {column.frame_counter}: not received.", False)]
    frame = frames[column.frame_index]
    lines = [
        (f"Frame {column.frame_counter}", False),
        (
            f"{frame['collected_at']:%Y-%m-%d} {_time_text(frame['collected_at'])} UTC",
            False,
        ),
    ]
    pixel = strip.pixel_index_at(image_y)
    if pixel is None:
        lines.append(("Pixel: -", False))
    else:
        value = strip.pixel_value(column.frame_index, pixel)
        if value is None:
            text = f"Pixel {pixel}: none"
        elif mode == "minus_median":
            text = f"Pixel {pixel}: {value:+.0f} DN vs its median"
        else:
            text = f"Pixel {pixel}: {value:.0f} DN"
        lines.append((text, True))
    band = strip.band_at(image_y)
    lines += [
        (
            _band_value_text(b, frame.get(b.column)),
            band is not None and band.band == b,
        )
        for b in mpi_strip.BANDS
    ]
    lines.append(
        (f"{frame['timing_method']} (sync {frame['time_sync_source']})", False)
    )
    return lines


def _readout_html(lines: list[tuple[str, bool]]) -> str:
    """`lines` (see `_readout_lines`) as a bulleted list, the one under the
    mouse in bold.
    """
    items = "".join(
        f"<li><b>{html.escape(text)}</b></li>"
        if bold
        else f"<li>{html.escape(text)}</li>"
        for text, bold in lines
    )
    return f'<ul class="list-disc pl-5 m-0">{items}</ul>'


def _legend(gradient_css: str, low: str, high: str) -> None:
    with ui.row().classes("items-center gap-1 no-wrap"):
        ui.label(low).classes("text-xs")
        ui.element("div").style(
            f"background: {gradient_css}; width: 8rem; height: 0.75rem; "
            "border-radius: 2px"
        )
        ui.label(high).classes("text-xs")


def _labels_column(strip: mpi_strip.MpiStrip) -> None:
    """The strip's row labels, at the same heights as the rows themselves."""

    def label(text: str, top: float, height: float, classes: str = "") -> None:
        ui.label(text).classes(f"absolute text-xs truncate w-full {classes}").style(
            f"top: {top}px; height: {height}px; line-height: {height}px"
        )

    last_pixel = strip.first_pixel_index + strip.pixel_count - 1
    with (
        ui.element("div")
        .classes("relative flex-none")
        .style(f"width: {_LABEL_COLUMN_WIDTH}; height: {strip.height}px")
    ):
        label(f"Pixel {strip.first_pixel_index}", 0, 16)
        label("Detector pixels", strip.detector_height / 2 - 8, 16, "font-bold")
        label(f"Pixel {last_pixel}", strip.detector_height - 16, 16)
        for band in strip.bands:
            text = f"{band.band.label} ({band.range_text()})"
            label(text, band.top, mpi_strip.BAND_HEIGHT)


def _time_axis(
    strip: mpi_strip.MpiStrip, frames: list[dict[str, Any]], *, fit: bool
) -> None:
    """Times (UTC) under the strip, every `_TICK_SPACING_PX` or so (or
    `_FIT_TICK_COUNT` labels across, for a fit-width strip) -- each the time
    of the first frame received at or after its position. A label that would
    land too close to the previous one (e.g. right after a run of missing
    frames) is skipped, so labels never overlap.
    """
    column_count = len(strip.columns)
    if fit:
        every = max(1, math.ceil(column_count / _FIT_TICK_COUNT))
    else:
        every = max(1, math.ceil(_TICK_SPACING_PX / strip.zoom))
    width = "100%" if fit else f"{strip.width}px"
    min_gap = every * _MIN_TICK_GAP_FRACTION
    last_labelled: int | None = None
    with (
        ui.element("div").classes("relative").style(f"width: {width}; height: 1.25rem")
    ):
        for start in range(0, len(strip.columns), every):
            column = next(
                (
                    (i, c)
                    for i, c in enumerate(strip.columns[start : start + every], start)
                    if c.frame_index is not None
                ),
                None,
            )
            if column is None:
                continue
            i, c = column
            assert c.frame_index is not None
            if last_labelled is not None and i - last_labelled < min_gap:
                continue
            last_labelled = i
            left = f"{i / column_count * 100:.3f}%" if fit else f"{i * strip.zoom}px"
            ui.label(_time_text(frames[c.frame_index]["collected_at"])[:8]).classes(
                "absolute text-xs text-grey whitespace-nowrap"
            ).style(f"left: {left}; border-left: 1px solid; padding-left: 2px")


def _strip_body(
    strip: mpi_strip.MpiStrip,
    frames: list[dict[str, Any]],
    mode: mpi_strip.IntensityMode,
    filename: str,
    *,
    fit: bool,
) -> ui.element:
    """Build the strip, its legends and readout; returns the strip's
    horizontally-scrolling container.
    """
    url = preview_image_url(strip.png, "image/png", filename)
    with ui.row().classes("w-full items-center gap-6"):
        suffix = " vs median" if mode == "minus_median" else ""
        _legend(
            mpi_strip.INTENSITY_GRADIENT_CSS,
            f"{strip.intensity_low:,.0f} DN{suffix}",
            f"{strip.intensity_high:,.0f} DN{suffix}",
        )
        _legend(mpi_strip.VALUE_GRADIENT_CSS, "Band's lowest", "highest")
        with ui.row().classes("items-center gap-1"):
            ui.element("div").style(
                "background: rgba(128, 128, 128, 0.35); width: 0.75rem; height: 0.75rem"
            )
            ui.label("Frame not received").classes("text-xs")
        ui.button(
            "Download PNG",
            icon="download",
            on_click=lambda: ui.download.from_url(url, filename),
        ).props("dense outline")

    with ui.row().classes("w-full no-wrap gap-2 items-start"):
        _labels_column(strip)
        with (
            ui.element("div")
            .classes("overflow-hidden" if fit else "overflow-x-auto")
            .style("flex: 1 1 auto; min-width: 0") as scroller
        ):
            # Sized up front (its height is always 1:1), so the layout -- and
            # the dialog's vertical scroll -- doesn't collapse while it loads.
            # Fit-width stretches it horizontally only; the mouse readout
            # still maps back to image pixels on each axis separately.
            width = "100%" if fit else f"{strip.width}px"
            image = ui.interactive_image(
                url, events=["mousemove"], cross="rgba(255, 255, 255, 0.5)"
            ).style(f"width: {width}; height: {strip.height}px; max-width: none")
            _time_axis(strip, frames, fit=fit)

    # Tall enough for a full readout (frame, time, pixel, every band, timing),
    # so the dialog's height -- and so its vertical scroll -- doesn't change
    # as the readout does, nor when a re-render resets it.
    readout_lines = 4 + len(mpi_strip.BANDS)
    readout = (
        ui.html("Hover over the strip for each frame's values.")
        .classes("font-mono text-sm w-full")
        .style(f"min-height: {readout_lines * 1.25}rem")
    )

    def on_mouse(e: GenericEventArguments) -> None:
        lines = _readout_lines(
            strip, frames, mode, e.args["image_x"], e.args["image_y"]
        )
        if lines is not None:
            readout.set_content(_readout_html(lines))

    image.on("mouse", on_mouse, ["image_x", "image_y"], throttle=0.05)
    return scroller


async def _scroll_center_fraction(scroller: ui.element) -> float | None:
    """How far along `scroller`'s content (0 to 1) the middle of its visible
    part is, or None if unknown (e.g. not mounted, or the client's gone) or
    it doesn't scroll at all (e.g. fit-width), so there's no position to keep.

    A fraction rather than a pixel offset, since the strip's width changes
    with the zoom.
    """
    with contextlib.suppress(TimeoutError):
        fraction = await ui.run_javascript(
            f"""(() => {{
                const el = getHtmlElement({scroller.id});
                if (!el || el.scrollWidth <= el.clientWidth) return null;
                return (el.scrollLeft + el.clientWidth / 2) / el.scrollWidth;
            }})()""",
            timeout=2.0,
        )
        if isinstance(fraction, (int, float)):
            return float(fraction)
    return None


def _scroll_to_center_fraction(scroller: ui.element, fraction: float) -> None:
    """Scroll `scroller` so `fraction` of its content is mid-view, once the
    browser has mounted it (it's usually brand new).
    """
    ui.run_javascript(
        f"""(() => {{
            const start = performance.now();
            const tick = () => {{
                const el = getHtmlElement({scroller.id});
                if (el) {{
                    el.scrollLeft = {fraction} * el.scrollWidth - el.clientWidth / 2;
                }} else if (performance.now() - start < 2000) {{
                    requestAnimationFrame(tick);
                }}
            }};
            tick();
        }})()"""
    )


def _summary_text(collection: dict[str, Any], frames: pl.DataFrame) -> str:
    recordings = frames["collection_started_at"].n_unique()
    first, last = frames["collected_at"].min(), frames["collected_at"].max()
    expected = collection.get("frames_expected")
    parts = [
        f"{frames.height:,} frame(s) received"
        + (f" of {expected:,} expected by their counters" if expected else ""),
        f"{first:%Y-%m-%d} {_time_text(first)} to {_time_text(last)} UTC",
    ]
    if recordings > 1:
        parts.append(f"{recordings} recordings, each starting its counter at 0")
    return " · ".join(parts)


async def show_mpi_strip(
    data_dir: Path, collection: dict[str, Any], collections: pl.DataFrame
) -> None:
    """A dialog with `collection`'s frames (one row of `mpi_collections.parquet`,
    `collections` being all of them) as a strip.
    """
    started_at = collection["collection_started_at"]
    filename = f"mpi_collection_{started_at:%Y-%m-%d_%H%M%S}Z.png"
    with ui.dialog().props("maximized") as dialog, ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label(
                f"MPI Collection Started {started_at:%Y-%m-%d %H:%M:%S} UTC"
            ).classes("text-lg font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")
        summary = ui.label().classes("text-caption text-grey")
        with ui.row().classes("items-center gap-4"):
            zoom_toggle = ui.toggle(_ZOOMS, value=_DEFAULT_ZOOM).props("dense")
            mode_toggle = ui.toggle(_MODES, value="raw").props("dense")
            range_toggle = ui.toggle(_RANGES, value="typical").props("dense")
        body = ui.column().classes("w-full gap-2")
        with body, ui.row().classes("items-center gap-2 p-2"):
            ui.spinner(size="md")
            ui.label("Loading frames...").classes("text-grey")
    dialog.open()

    frames = await run.io_bound(
        mpi_strip.load_collection_frames, data_dir, collection, collections
    )
    if frames is None:  # Only when the server's shutting down.
        return
    if frames.is_empty():
        body.clear()
        with body:
            ui.label("No frames of this collection have been downlinked.").classes(
                "text-grey"
            )
        return
    summary.set_text(_summary_text(collection, frames))
    frame_rows = frames.drop("pixels").to_dicts()
    fit_zoom = max(
        1, math.ceil(_FIT_RENDER_WIDTH_PX / max(1, mpi_strip.column_count(frames)))
    )
    scroller: ui.element | None = None

    async def render() -> None:
        nonlocal scroller
        mode = mode_toggle.value
        fit = zoom_toggle.value == _FIT
        strip = await run.io_bound(
            mpi_strip.render_strip,
            frames,
            zoom=fit_zoom if fit else zoom_toggle.value,
            mode=mode,
            intensity_range=range_toggle.value,
        )
        if strip is None:
            return
        # Read just before swapping strips, so scrolling done while this one
        # rendered isn't lost -- keeps the same time centered in view.
        center = None if scroller is None else await _scroll_center_fraction(scroller)
        body.clear()
        with body:
            scroller = _strip_body(strip, frame_rows, mode, filename, fit=fit)
        if center is not None:
            _scroll_to_center_fraction(scroller, center)

    for toggle in (zoom_toggle, mode_toggle, range_toggle):
        toggle.on_value_change(render)
    await render()
