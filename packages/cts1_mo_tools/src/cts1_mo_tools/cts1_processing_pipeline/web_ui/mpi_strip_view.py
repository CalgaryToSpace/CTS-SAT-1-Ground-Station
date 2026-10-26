"""The MPI Data page's strip view of one MPI collection: a dialog with the
collection's frames as a long strip -- see `mpi_strip` for the image
itself -- with a readout of whatever frame, pixel and housekeeping value
is under the mouse.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["show_mpi_strip"]

import math
from typing import TYPE_CHECKING, Any

from nicegui import run, ui

from . import mpi_strip
from .preview_image_route import preview_image_url

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from nicegui.events import GenericEventArguments

# Image pixels across per frame.
_ZOOMS = {1: "1x", 2: "2x", 4: "4x", 8: "8x"}
_DEFAULT_ZOOM = 2
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


def _readout_text(
    strip: mpi_strip.MpiStrip,
    frames: list[dict[str, Any]],
    mode: mpi_strip.IntensityMode,
    image_x: float,
    image_y: float,
) -> str | None:
    """What's under the mouse at (`image_x`, `image_y`), for display."""
    column = strip.column_at(image_x)
    if column is None:
        return None
    if column.kind == "separator":
        return "Between two recordings merged into this collection."
    if column.kind == "missing" or column.frame_index is None:
        return f"Frame {column.frame_counter}: not received."
    frame = frames[column.frame_index]
    parts = [
        f"Frame {column.frame_counter}",
        f"{frame['collected_at']:%Y-%m-%d} {_time_text(frame['collected_at'])} UTC",
    ]
    pixel = strip.pixel_index_at(image_y)
    if pixel is not None:
        value = strip.pixel_value(column.frame_index, pixel)
        if value is None:
            parts.append(f"pixel {pixel}: none")
        elif mode == "minus_median":
            parts.append(f"pixel {pixel}: {value:+.0f} DN vs its median")
        else:
            parts.append(f"pixel {pixel}: {value:.0f} DN")
    band = strip.band_at(image_y)
    # The band under the mouse first, then the rest.
    bands = sorted(mpi_strip.BANDS, key=lambda b: band is None or b != band.band)
    parts += [_band_value_text(b, frame.get(b.column)) for b in bands]
    parts.append(f"{frame['timing_method']} (sync {frame['time_sync_source']})")
    return " · ".join(parts)


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


def _time_axis(strip: mpi_strip.MpiStrip, frames: list[dict[str, Any]]) -> None:
    """Times (UTC) under the strip, every `_TICK_SPACING_PX` or so -- each the
    time of the first frame received at or after its position.
    """
    every = max(1, math.ceil(_TICK_SPACING_PX / strip.zoom))
    with (
        ui.element("div")
        .classes("relative")
        .style(f"width: {strip.width}px; height: 1.25rem")
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
            ui.label(_time_text(frames[c.frame_index]["collected_at"])[:8]).classes(
                "absolute text-xs text-grey whitespace-nowrap"
            ).style(
                f"left: {i * strip.zoom}px; border-left: 1px solid; padding-left: 2px"
            )


def _strip_body(
    strip: mpi_strip.MpiStrip,
    frames: list[dict[str, Any]],
    mode: mpi_strip.IntensityMode,
    filename: str,
) -> None:
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
            .classes("overflow-x-auto")
            .style("flex: 1 1 auto; min-width: 0")
        ):
            image = ui.interactive_image(
                url, events=["mousemove"], cross="rgba(255, 255, 255, 0.5)"
            ).style(f"width: {strip.width}px; max-width: none")
            _time_axis(strip, frames)

    readout = ui.label("Hover over the strip for each frame's values.").classes(
        "font-mono text-sm w-full"
    )

    def on_mouse(e: GenericEventArguments) -> None:
        text = _readout_text(strip, frames, mode, e.args["image_x"], e.args["image_y"])
        if text is not None:
            readout.set_text(text)

    image.on("mouse", on_mouse, ["image_x", "image_y"], throttle=0.05)


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

    async def render() -> None:
        mode = mode_toggle.value
        strip = await run.io_bound(
            mpi_strip.render_strip,
            frames,
            zoom=zoom_toggle.value,
            mode=mode,
            intensity_range=range_toggle.value,
        )
        if strip is None:
            return
        body.clear()
        with body:
            _strip_body(strip, frame_rows, mode, filename)

    for toggle in (zoom_toggle, mode_toggle, range_toggle):
        toggle.on_value_change(render)
    await render()
