"""The "MPI Vis" page: a list of every MPI collection on the left, and
the selected one's frames as a long strip -- see `mpi_strip` for the image
itself -- with a readout of whatever frame, pixel and housekeeping value is
under the mouse.

The selected collection is kept in the page's URL (`?collection=<start
time>`, see `collection_key`), so a link to it opens it directly, and the
browser's back/forward buttons step between the collections viewed.

It's its own page, rather than part of the MPI Data page, so it doesn't
wait on (or compete for the CPU with) that page's collection timeline --
the slowest thing on it, especially while the pipeline's running.

Each collection's frames are decoded once (`mpi_strip.prepare_frames`),
and each strip image rendered once per value setting at 1 image pixel per
frame: the fixed zooms just stretch it in the browser (pixelated), so
changing the zoom never waits on the server.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = [
    "COLLECTION_QUERY_PARAM",
    "MPI_VIS_PATH",
    "build_mpi_vis_page",
    "collection_key",
    "vis_page_url",
]

import contextlib
import html
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from nicegui import run, ui

from . import mpi_collections, mpi_strip
from .layout import page_shell
from .preview_image_route import preview_image_url

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime
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
# Marks the strip's horizontally-scrolling container, for the wheel handler.
_SCROLLER_CLASS = "mpi-strip-scroller"
# Marks the strip's image (its tiles, side by side) inside that container.
_CANVAS_CLASS = "mpi-strip-canvas"
# Emitted on the strip's container by Ctrl+scroll -- see `_install_wheel_handler`.
_WHEEL_ZOOM_EVENT = "strip_wheel_zoom"
# Where a re-rendered strip should be scrolled to: `(fraction, viewport)`
# puts `fraction` of the way along the strip at `viewport` of the way across
# the visible part (0.5 = centered).
_ScrollAnchor = tuple[float, float]

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
    strip: mpi_strip.MpiStrip, frames: list[dict[str, Any]], zoom: int | None
) -> None:
    """Times (UTC) under the strip, every `_TICK_SPACING_PX` or so at `zoom`
    screen pixels per column (or `_FIT_TICK_COUNT` labels across, for a
    fit-width strip, `zoom` None) -- each the time
    of the first frame received at or after its position. A label that would
    land too close to the previous one (e.g. right after a run of missing
    frames) is skipped, so labels never overlap.
    """
    column_count = len(strip.columns)
    if zoom is None:
        every = max(1, math.ceil(column_count / _FIT_TICK_COUNT))
    else:
        every = max(1, math.ceil(_TICK_SPACING_PX / zoom))
    width = "100%" if zoom is None else f"{column_count * zoom}px"
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
            left = f"{i / column_count * 100:.3f}%" if zoom is None else f"{i * zoom}px"
            ui.label(_time_text(frames[c.frame_index]["collected_at"])[:8]).classes(
                "absolute text-xs text-grey whitespace-nowrap"
            ).style(f"left: {left}; border-left: 1px solid; padding-left: 2px")


def _strip_canvas(
    strip: mpi_strip.MpiStrip, filename: str, zoom: int | None
) -> ui.element:
    """The strip's image -- its tiles side by side -- at `zoom` screen pixels
    per column (or fit to the width, for None), with a crosshair that follows
    the mouse (drawn in the browser, so it never waits on the server).
    """
    fit = zoom is None
    # Sized up front (its height is always 1:1), so the layout -- and the
    # page's vertical scroll -- doesn't collapse while it loads. Stretched
    # horizontally only; the mouse readout still maps back to image pixels
    # on each axis separately.
    width = "100%" if fit else f"{len(strip.columns) * zoom}px"
    # Nearest-neighbour, so each frame stays a crisp column. (A fit-width
    # strip's shrunk to about its shown width by the server, averaging
    # columns, so it's barely resampled either way.)
    rendering = "auto" if fit else "pixelated"
    stem = filename.removesuffix(".png")
    with (
        ui.element("div")
        .classes(f"{_CANVAS_CLASS} relative flex no-wrap")
        .style(f"width: {width}; height: {strip.height}px; cursor: crosshair")
    ) as canvas:
        for i, (png, tile_width) in enumerate(
            zip(strip.tiles, strip.tile_widths, strict=True)
        ):
            url = preview_image_url(png, "image/png", f"{stem}_{i}.png")
            ui.element("img").props(f'src="{url}" draggable="false"').style(
                f"display: block; flex: none; height: 100%; "
                f"width: {tile_width / strip.width * 100:.6f}%; "
                f"image-rendering: {rendering}; pointer-events: none"
            )
        line = "position: absolute; pointer-events: none; display: none; "
        line += "background: rgba(255, 255, 255, 0.5)"
        cross_x = ui.element("div").style(f"{line}; top: 0; bottom: 0; width: 1px")
        cross_y = ui.element("div").style(f"{line}; left: 0; right: 0; height: 1px")
    canvas.on(
        "mousemove",
        js_handler=f"""(e) => {{
            const r = getHtmlElement({canvas.id}).getBoundingClientRect();
            const x = getHtmlElement({cross_x.id}), y = getHtmlElement({cross_y.id});
            x.style.left = (e.clientX - r.left) + 'px';
            y.style.top = (e.clientY - r.top) + 'px';
            x.style.display = y.style.display = 'block';
        }}""",
    )
    canvas.on(
        "mouseleave",
        js_handler=f"""() => {{
            getHtmlElement({cross_x.id}).style.display = 'none';
            getHtmlElement({cross_y.id}).style.display = 'none';
        }}""",
    )
    return canvas


def _strip_body(  # noqa: PLR0913
    strip: mpi_strip.MpiStrip,
    frames: list[dict[str, Any]],
    mode: mpi_strip.IntensityMode,
    filename: str,
    zoom: int | None,
    on_download: Callable[[], Awaitable[None]],
) -> ui.element:
    """Build the strip, its legends and readout, at `zoom` screen pixels per
    column -- the image stretched to that, pixelated -- or fit to the
    width (None); returns the strip's horizontally-scrolling container.
    """
    fit = zoom is None
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
        ui.button("Download PNG", icon="download", on_click=on_download).props(
            "dense outline"
        )

    with ui.row().classes("w-full no-wrap gap-2 items-start"):
        _labels_column(strip)
        with (
            ui.element("div")
            .classes(
                f"{_SCROLLER_CLASS} {'overflow-hidden' if fit else 'overflow-x-auto'}"
            )
            .style("flex: 1 1 auto; min-width: 0") as scroller
        ):
            canvas = _strip_canvas(strip, filename, zoom)
            _time_axis(strip, frames, zoom)

    # Tall enough for a full readout (frame, time, pixel, every band, timing),
    # so the page's height -- and so its vertical scroll -- doesn't change
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

    # Only where the mouse is, in image pixels -- the readout's built here,
    # from data the browser doesn't have.
    canvas.on(
        "mousemove",
        on_mouse,
        throttle=0.05,
        js_handler=f"""(e) => {{
            const r = getHtmlElement({canvas.id}).getBoundingClientRect();
            if (!r.width || !r.height) return;
            emit({{
                image_x: (e.clientX - r.left) * {strip.width} / r.width,
                image_y: (e.clientY - r.top) * {strip.height} / r.height,
            }});
        }}""",
    )
    return scroller


async def _scroll_center_anchor(scroller: ui.element) -> _ScrollAnchor | None:
    """The part of the strip mid-view in `scroller`, as an anchor to restore
    after a re-render -- or None if unknown (e.g. not mounted, or the
    client's gone) or it doesn't scroll at all (e.g. fit-width), so there's
    no position to keep.

    A fraction of the strip's width rather than a pixel offset, since the
    width changes with the zoom.
    """
    with contextlib.suppress(TimeoutError):
        fraction = await scroller.client.run_javascript(
            f"""(() => {{
                const el = getHtmlElement({scroller.id});
                const img = el && el.querySelector('.{_CANVAS_CLASS}');
                if (!img || !img.clientWidth || el.scrollWidth <= el.clientWidth) {{
                    return null;
                }}
                return (el.scrollLeft + el.clientWidth / 2) / img.clientWidth;
            }})()""",
            timeout=2.0,
        )
        if isinstance(fraction, (int, float)):
            return float(fraction), 0.5
    return None


def _scroll_to_anchor(scroller: ui.element, anchor: _ScrollAnchor) -> None:
    """Scroll `scroller` to `anchor` (see `_ScrollAnchor`), once the browser
    has mounted it (it's usually brand new).
    """
    fraction, viewport = anchor
    # Via the element's client, not `ui.`: works from any task (e.g. a timer).
    scroller.client.run_javascript(
        f"""(() => {{
            const start = performance.now();
            const tick = () => {{
                const el = getHtmlElement({scroller.id});
                const img = el && el.querySelector('.{_CANVAS_CLASS}');
                if (img && img.clientWidth) {{
                    el.scrollLeft =
                        {fraction} * img.clientWidth - {viewport} * el.clientWidth;
                }} else if (performance.now() - start < 2000) {{
                    requestAnimationFrame(tick);
                }}
            }};
            tick();
        }})()"""
    )


def _install_wheel_handler(container: ui.element) -> None:
    """Over the strip (anywhere in `container`): the scroll wheel pans it
    sideways (when it's wider than the view), and Ctrl+scroll dispatches
    `_WHEEL_ZOOM_EVENT` on `container`, its `detail` `{direction, fraction,
    viewport, client_width}` -- `direction` 1 to zoom
    in or -1 to zoom out, at the point under the cursor (as a
    `_ScrollAnchor`) -- instead of zooming the whole page.

    Installed on `container`, which outlives every re-rendered strip.
    Installing it twice on the same element is a no-op.
    """
    container.client.run_javascript(
        f"""(() => {{
            const start = performance.now();
            const install = () => {{
                const root = getHtmlElement({container.id});
                if (!root) {{
                    if (performance.now() - start < 5000) {{
                        requestAnimationFrame(install);
                    }}
                    return;
                }}
                if (root.dataset.wheelInstalled) return;
                root.dataset.wheelInstalled = '1';
                let pending = 0;
                let lastZoomAt = 0;
                root.addEventListener('wheel', (e) => {{
                    const el = e.target.closest('.{_SCROLLER_CLASS}');
                    const img = el && el.querySelector('.{_CANVAS_CLASS}');
                    if (!img || !img.clientWidth) return;
                    const scale = e.deltaMode === 1 ? 16
                        : e.deltaMode === 2 ? el.clientWidth : 1;
                    const dy = e.deltaY * scale;
                    if (e.ctrlKey || e.metaKey) {{
                        // Never the browser's own page zoom, over the strip.
                        e.preventDefault();
                        const now = performance.now();
                        // One step per notch, or per ~0.35 s of trackpad pinch.
                        if (now - lastZoomAt < 350) {{ pending = 0; return; }}
                        pending += dy;
                        if (Math.abs(pending) < 40) return;
                        const x = e.clientX - el.getBoundingClientRect().left;
                        // A DOM event: `container` is a plain element, not
                        // a Vue component with `$emit`.
                        root.dispatchEvent(new CustomEvent('{_WHEEL_ZOOM_EVENT}', {{
                            detail: {{
                            direction: pending < 0 ? 1 : -1,
                            fraction: (el.scrollLeft + x) / img.clientWidth,
                            viewport: x / el.clientWidth,
                            client_width: el.clientWidth,
                        }} }}));
                        pending = 0;
                        lastZoomAt = now;
                    }} else if (
                        Math.abs(dy) > Math.abs(e.deltaX * scale)
                        && el.scrollWidth > el.clientWidth
                    ) {{
                        e.preventDefault();
                        el.scrollLeft += dy;
                    }}
                }}, {{ passive: false }});
            }};
            install();
        }})()"""
    )


def _next_zoom(
    current: int | str, *, direction: int, client_width: float, column_count: int
) -> int | str | None:
    """The `_ZOOMS` level one step in (`direction` > 0) or out from `current`,
    by on-screen pixels per frame -- a fit-width strip's depends on how wide
    it's shown, so it can sit anywhere among the fixed zooms. None at either
    end.
    """
    fit_scale = client_width / max(1, column_count)
    scales = {z: fit_scale if z == _FIT else float(z) for z in _ZOOMS}
    now = scales[current]
    if direction > 0:
        bigger = [z for z in _ZOOMS if scales[z] > now]
        return min(bigger, key=lambda z: scales[z]) if bigger else None
    smaller = [z for z in _ZOOMS if scales[z] < now]
    return max(smaller, key=lambda z: scales[z]) if smaller else None


def _labelled(label: str) -> ui.column:
    """A column with a small caption `label`, for a control placed in it."""
    with ui.column().classes("gap-0") as column:
        ui.label(label).classes("text-caption text-grey")
    return column


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


# The page URL's query parameter naming the collection being viewed.
COLLECTION_QUERY_PARAM = "collection"
MPI_VIS_PATH = "/mpi-vis"
# Dispatched on the page's URL-sync element when the browser's back/forward
# buttons change the URL -- see `_MpiStripView`.
_URL_CHANGE_EVENT = "strip_url_change"
# The page's strip area fills the window under the header and the page's
# title row, so the collections list and the strip scroll separately.
_VIEW_HEIGHT = "calc(100vh - 9rem)"


def collection_key(started_at: datetime) -> str:
    """`started_at` (a collection's `collection_started_at`) as it appears in
    the URL, e.g. "2026-10-30T20:17:01.123Z" -- unique per collection.
    """
    return f"{started_at:%Y-%m-%dT%H:%M:%S}.{started_at.microsecond // 1000:03d}Z"


def vis_page_url(key: str) -> str:
    """The MPI Vis page open at the collection `key` (see `collection_key`)."""
    return f"{MPI_VIS_PATH}?{COLLECTION_QUERY_PARAM}={quote(key, safe='')}"


def _collection_item_caption(row: dict[str, Any]) -> str:
    present, expected = row.get("frames_present") or 0, row.get("frames_expected")
    return f"{present:,} of {expected:,} frames" if expected else f"{present:,} frames"


@dataclass(frozen=True, slots=True)
class _LoadedCollection:
    """A collection's frames, loaded and decoded -- everything the strip
    needs that doesn't depend on the view's settings.
    """

    prepared: mpi_strip.PreparedFrames
    # Each frame's fields bar its pixels, for the readout and time axis.
    frame_rows: list[dict[str, Any]]
    summary: str


def _load_collection(
    data_dir: Path, row: dict[str, Any], collections: pl.DataFrame
) -> _LoadedCollection | None:
    """Blocking -- run it via `run.io_bound`. None when none of the
    collection's frames have been downlinked.
    """
    frames = mpi_strip.load_collection_frames(data_dir, row, collections)
    prepared = mpi_strip.prepare_frames(frames)
    if prepared is None:
        return None
    return _LoadedCollection(
        prepared=prepared,
        frame_rows=frames.drop("pixels").to_dicts(),
        summary=_summary_text(row, frames),
    )


class _MpiStripView:
    """The page's collections list and strip view. `set_collections` (once
    loaded) gives it the collections to list and look keys up in; `show`
    switches it to one.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._collections: pl.DataFrame | None = None
        self._rows_by_key: dict[str, dict[str, Any]] = {}
        self._items: dict[str, ui.item] = {}
        self._active_key: str | None = None
        # Bumped per `show`, so a slow load for a collection the user has
        # since moved on from is dropped instead of displayed.
        self._load_generation = 0

        # Listens for back/forward; registered now, before it reaches the
        # browser.
        url_sync = ui.element("div").classes("hidden")
        url_sync.on(
            _URL_CHANGE_EVENT,
            lambda e: self._on_url_change(e.args["detail"]),
            ["detail"],
        )
        ui.add_body_html(
            f"""<script>
            window.addEventListener('popstate', () => {{
                const el = getHtmlElement({url_sync.id});
                if (el) el.dispatchEvent(new CustomEvent('{_URL_CHANGE_EVENT}', {{
                    detail: new URL(location.href).searchParams.get(
                        '{COLLECTION_QUERY_PARAM}') || ''
                }}));
            }});
            </script>"""
        )

        with (
            ui.row()
            .classes("w-full no-wrap gap-4 min-h-0")
            .style(f"height: {_VIEW_HEIGHT}")
        ):
            with ui.column().classes(
                "w-64 flex-none h-full overflow-y-auto gap-0 border-r pr-2"
            ):
                ui.label("MPI Collections").classes("text-caption text-grey")
                self._list = ui.list().props("dense separator").classes("w-full")
                with self._list, ui.row().classes("items-center gap-2 p-2"):
                    ui.spinner(size="sm")
                    ui.label("Loading...").classes("text-grey")
            with ui.column().classes("grow min-w-0 h-full overflow-y-auto gap-2"):
                self._title = ui.label().classes("text-lg font-bold")
                self._summary = ui.label().classes("text-caption text-grey")
                with ui.row().classes("items-end gap-4"):
                    with _labelled("Time Scale"):
                        zoom_toggle = ui.toggle(_ZOOMS, value=_DEFAULT_ZOOM)
                    with _labelled("Value Processing"):
                        mode_toggle = ui.toggle(_MODES, value="raw")
                    with _labelled("Value Scale"):
                        range_toggle = ui.toggle(_RANGES, value="typical")
                    for toggle in (zoom_toggle, mode_toggle, range_toggle):
                        toggle.props("dense")
                    ui.label(
                        "Over the strip: scroll to pan, Ctrl+scroll to zoom."
                    ).classes("text-caption text-grey")
                self._body = ui.column().classes("w-full gap-2")
        self._viewer = _StripViewer(
            body=self._body,
            zoom_toggle=zoom_toggle,
            mode_toggle=mode_toggle,
            range_toggle=range_toggle,
        )
        _install_wheel_handler(self._body)

    def set_collections(self, collections: pl.DataFrame) -> None:
        """(Re)list `collections` (`mpi_collections.parquet`), newest first."""
        self._collections = collections
        rows = collections.sort("collection_started_at", descending=True).to_dicts()
        self._rows_by_key = {
            collection_key(r["collection_started_at"]): r for r in rows
        }
        self._items = {}
        self._list.clear()
        with self._list:
            if not rows:
                ui.label(
                    "No MPI collections yet -- run the pipeline (step_1..step_3, "
                    "then step_6 through step_8) first."
                ).classes("text-caption text-grey p-2")
            for key, row in self._rows_by_key.items():
                # NiceGUI passes the click event first; the key's bound after.
                with (
                    ui.item(on_click=lambda _e, k=key: self.show(k)) as item,
                    ui.item_section(),
                ):
                    ui.item_label(
                        f"{row['collection_started_at']:%Y-%m-%d %H:%M:%S} UTC"
                    )
                    ui.item_label(_collection_item_caption(row)).props("caption")
                if not row.get("frames_present"):
                    item.props("disable")
                self._items[key] = item
        self._highlight(self._active_key)

    def notify_missing(self, key: str) -> None:
        with self._body:  # A UI context, even when called from a timer.
            ui.notify(f"No MPI collection starting at {key}.", type="warning")

    def has_collection(self, key: str) -> bool:
        return key in self._rows_by_key

    def newest_with_frames(self) -> str | None:
        return next(
            (k for k, r in self._rows_by_key.items() if r.get("frames_present")), None
        )

    async def show(self, key: str, *, url: str | None = "push") -> None:
        """Switch to the collection `key` (see `collection_key`), and set the
        URL to it: `url` "push" as a new history entry, "replace" in place
        of the current one, or None not at all (e.g. when the URL's what
        asked for it).
        """
        row = self._rows_by_key.get(key)
        if row is None or self._collections is None:
            self.notify_missing(key)
            return
        self._active_key = key
        self._highlight(key)
        if url is not None:
            _set_url_collection(self._body, key, replace=url == "replace")

        self._load_generation += 1
        generation = self._load_generation
        started_at = row["collection_started_at"]
        self._title.set_text(
            f"MPI Collection Started {started_at:%Y-%m-%d %H:%M:%S} UTC"
        )
        self._summary.set_text("")
        self._viewer.clear_strip()
        with self._body, ui.row().classes("items-center gap-2 p-2"):
            ui.spinner(size="md")
            ui.label("Loading frames...").classes("text-grey")

        loaded = await run.io_bound(
            _load_collection, self._data_dir, row, self._collections
        )
        if generation != self._load_generation:
            return  # Another collection's been picked meanwhile.
        if loaded is None:
            self._body.clear()
            with self._body:
                ui.label("No frames of this collection have been downlinked.").classes(
                    "text-grey"
                )
            return
        self._summary.set_text(loaded.summary)
        await self._viewer.show(
            loaded, filename=f"mpi_collection_{started_at:%Y-%m-%d_%H%M%S}Z.png"
        )

    def _highlight(self, key: str | None) -> None:
        for item_key, item in self._items.items():
            if item_key == key:
                item.props("active")
                item.client.run_javascript(
                    f"getHtmlElement({item.id})?.scrollIntoView({{block: 'nearest'}})"
                )
            else:
                item.props(remove="active")

    async def _on_url_change(self, key: str) -> None:
        """The browser's back/forward buttons moved to a URL viewing `key`
        (a `collection_key`), or none (`""`) -- the newest collection, then.
        """
        if not key:
            newest = self.newest_with_frames()
            if newest is not None and newest != self._active_key:
                await self.show(newest, url="replace")
        elif key != self._active_key:
            await self.show(key, url=None)


def _set_url_collection(element: ui.element, key: str, *, replace: bool) -> None:
    """Set the page URL's `COLLECTION_QUERY_PARAM` to `key`, as a new history
    entry (or, if `replace`, in place of the current one) -- unless it's
    already that. `element` is any element on the page, for its client.
    """
    method = "replaceState" if replace else "pushState"
    element.client.run_javascript(
        f"""(() => {{
            const url = new URL(location.href);
            const key = {json.dumps(key)};
            if (url.searchParams.get('{COLLECTION_QUERY_PARAM}') === key) return;
            url.searchParams.set('{COLLECTION_QUERY_PARAM}', key);
            history.{method}(null, '', url);
        }})()"""
    )


# The strip images kept per collection: one per (value processing, value
# scale, fit-width or not) -- every fixed zoom shares one image, stretched.
_StripKey = tuple[str, str, bool]


class _StripViewer:
    """Renders a collection's strip into `body` per the toggles' settings,
    re-rendering when they change, and handles the wheel over it (see
    `_install_wheel_handler`). `show` switches it to another collection.
    """

    def __init__(
        self,
        *,
        body: ui.element,
        zoom_toggle: ui.toggle,
        mode_toggle: ui.toggle,
        range_toggle: ui.toggle,
    ) -> None:
        self._body = body
        self._zoom_toggle = zoom_toggle
        self._mode_toggle = mode_toggle
        self._range_toggle = range_toggle
        self._loaded: _LoadedCollection | None = None
        self._strips: dict[_StripKey, mpi_strip.MpiStrip] = {}
        self._filename = ""
        self._column_count = 0
        self._fit_zoom = 1
        self._scroller: ui.element | None = None
        # Set by a Ctrl+scroll zoom, for the render it triggers to scroll to.
        self._pending_anchor: _ScrollAnchor | None = None
        self._rendering = False

        for toggle in (zoom_toggle, mode_toggle, range_toggle):
            toggle.on_value_change(self.render)
        # Registered now, before `body` reaches the browser: a listener added
        # to an element it already has isn't reliably wired up there.
        body.on(
            _WHEEL_ZOOM_EVENT,
            lambda e: self._on_wheel_zoom(e.args["detail"]),
            ["detail"],
        )

    def clear_strip(self) -> None:
        """Drop the current collection (e.g. while the next one loads)."""
        self._loaded = None
        self._strips = {}
        self._scroller = None
        self._body.clear()

    async def show(self, loaded: _LoadedCollection, *, filename: str) -> None:
        """Switch to (and render) a collection, keeping the current view
        settings; it starts scrolled to the start.
        """
        self._loaded = loaded
        self._strips = {}
        self._filename = filename
        self._column_count = len(loaded.prepared.columns)
        self._fit_zoom = max(
            1, math.ceil(_FIT_RENDER_WIDTH_PX / max(1, self._column_count))
        )
        self._scroller = None
        self._pending_anchor = None
        await self.render()

    async def render(self) -> None:
        if self._loaded is None:
            return
        self._rendering = True
        try:
            await self._render(self._loaded)
        finally:
            self._rendering = False

    async def _render(self, loaded: _LoadedCollection) -> None:
        mode = self._mode_toggle.value
        zoom = None if self._zoom_toggle.value == _FIT else self._zoom_toggle.value
        intensity_range = self._range_toggle.value
        key = (mode, intensity_range, zoom is None)
        strip = self._strips.get(key)
        if strip is None:
            strip = await run.io_bound(
                mpi_strip.render_prepared,
                loaded.prepared,
                zoom=self._fit_zoom if zoom is None else 1,
                mode=mode,
                intensity_range=intensity_range,
                # Fit-width: no wider than it's worth sending, as the browser
                # would only shrink it.
                max_width=_FIT_RENDER_WIDTH_PX if zoom is None else None,
                tile_width=mpi_strip.TILE_WIDTH,
            )
            if strip is None or loaded is not self._loaded:
                return  # Server shutting down, or switched collections meanwhile.
            self._strips[key] = strip
        # A Ctrl+scroll zoom keeps the point under the cursor in place;
        # anything else keeps whatever's mid-view there. Read just before
        # swapping strips, so scrolling done while this one rendered isn't lost.
        anchor, self._pending_anchor = self._pending_anchor, None
        if anchor is None and self._scroller is not None:
            anchor = await _scroll_center_anchor(self._scroller)
            if loaded is not self._loaded:
                return
        self._body.clear()
        with self._body:
            self._scroller = _strip_body(
                strip,
                loaded.frame_rows,
                mode,
                self._filename,
                zoom,
                on_download=lambda: self._download(loaded, mode, intensity_range),
            )
        if anchor is not None:
            _scroll_to_anchor(self._scroller, anchor)

    async def _download(
        self,
        loaded: _LoadedCollection,
        mode: mpi_strip.IntensityMode,
        intensity_range: mpi_strip.IntensityRange,
    ) -> None:
        """Download the strip as one PNG, 1 image pixel per column -- only
        rendered when asked for, as the page itself shows it in tiles.
        """
        strip = await run.io_bound(
            mpi_strip.render_prepared,
            loaded.prepared,
            mode=mode,
            intensity_range=intensity_range,
        )
        if strip is not None:
            with self._body:  # A UI context, for the client to download on.
                ui.download.content(strip.png, self._filename, "image/png")

    def _on_wheel_zoom(self, detail: dict[str, Any]) -> None:
        """Step the zoom in/out at the cursor -- `detail` as dispatched by
        `_install_wheel_handler`.
        """
        if self._rendering or self._loaded is None:
            return  # Don't stack up slow renders from one long gesture.
        target = _next_zoom(
            self._zoom_toggle.value,
            direction=detail["direction"],
            client_width=detail["client_width"],
            column_count=self._column_count,
        )
        if target is None:
            return
        self._pending_anchor = (detail["fraction"], detail["viewport"])
        self._zoom_toggle.value = target  # Re-renders, via `on_value_change`.


def build_mpi_vis_page(data_dir: Path, *, collection: str | None = None) -> None:
    """`collection`: a `collection_key` to open at, from the URL -- else the
    newest collection with any frames downlinked.
    """
    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("MPI Vis").classes("text-2xl font-bold")
            ui.link("All collections, in detail: MPI Data", "/mpi-data").classes(
                "text-caption"
            )
        view = _MpiStripView(data_dir)

    async def _load() -> None:
        collections = await run.io_bound(mpi_collections.load_collections, data_dir)
        if collections is None:
            return  # Server shutting down.
        view.set_collections(collections)
        if collection and view.has_collection(collection):
            await view.show(collection, url=None)  # The URL already names it.
            return
        if collection:
            view.notify_missing(collection)
        newest = view.newest_with_frames()
        if newest is not None:
            await view.show(newest, url="replace")

    ui.timer(0, _load, once=True)
