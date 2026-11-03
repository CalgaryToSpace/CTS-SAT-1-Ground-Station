"""The MPI Data page's strip view of MPI collections: a dialog with a list of
every collection on the left, and the selected one's frames as a long strip
-- see `mpi_strip` for the image itself -- with a readout of whatever frame,
pixel and housekeeping value is under the mouse.

The selected collection is kept in the page's URL (`?collection=<start
time>`, see `collection_key`), so a link to it opens it directly, and the
browser's back/forward buttons step between the collections viewed.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["COLLECTION_QUERY_PARAM", "MpiStripDialog", "collection_key"]

import contextlib
import html
import json
import math
from typing import TYPE_CHECKING, Any

from nicegui import run, ui

from . import mpi_strip
from .preview_image_route import preview_image_url

if TYPE_CHECKING:
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
            .classes(
                f"{_SCROLLER_CLASS} {'overflow-hidden' if fit else 'overflow-x-auto'}"
            )
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
                const img = el && el.querySelector('img');
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
                const img = el && el.querySelector('img');
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

    Installed on `container`, which outlives every re-rendered strip -- but
    not a closed dialog (its content is re-created on each open), so call
    this on every open. Installing it twice on the same element is a no-op.
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
                    const img = el && el.querySelector('img');
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
# Dispatched on the dialog's URL-sync element when the browser's back/forward
# buttons change the URL -- see `MpiStripDialog`.
_URL_CHANGE_EVENT = "strip_url_change"


def collection_key(started_at: datetime) -> str:
    """`started_at` (a collection's `collection_started_at`) as it appears in
    the URL, e.g. "2026-10-30T20:17:01.123Z" -- unique per collection.
    """
    return f"{started_at:%Y-%m-%dT%H:%M:%S}.{started_at.microsecond // 1000:03d}Z"


def _collection_item_caption(row: dict[str, Any]) -> str:
    present, expected = row.get("frames_present") or 0, row.get("frames_expected")
    return f"{present:,} of {expected:,} frames" if expected else f"{present:,} frames"


class MpiStripDialog:
    """The strip-view dialog, built once per page (call from the page's build
    function). `show` opens it at a collection; `set_collections` (once
    loaded) gives it the collections to list and look keys up in.
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
        # When the dialog is closed because the URL changed (back button),
        # the URL is already right -- don't push another entry.
        self._closing_from_url = False

        # Listens for back/forward. Outside the dialog, so it's always in
        # the page; registered now, before it reaches the browser.
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
            ui.dialog().props("maximized") as self._dialog,
            ui.card().classes("w-full h-full no-wrap gap-2"),
        ):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                self._title = ui.label().classes("text-lg font-bold")
                ui.button(icon="close", on_click=self._dialog.close).props(
                    "flat round dense"
                )
            with ui.row().classes("w-full grow no-wrap gap-4 min-h-0"):
                with ui.column().classes(
                    "w-64 flex-none h-full overflow-y-auto gap-0 border-r pr-2"
                ):
                    ui.label("MPI Collections").classes("text-caption text-grey")
                    self._list = ui.list().props("dense separator").classes("w-full")
                with ui.column().classes("grow min-w-0 h-full overflow-y-auto gap-2"):
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
        self._dialog.on("hide", self._on_hide)
        self._viewer = _StripViewer(
            body=self._body,
            zoom_toggle=zoom_toggle,
            mode_toggle=mode_toggle,
            range_toggle=range_toggle,
        )

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

    def has_collection(self, key: str) -> bool:
        return key in self._rows_by_key

    async def show(self, key: str, *, push_url: bool = True) -> None:
        """Open the dialog (if it isn't already) at the collection `key` (see
        `collection_key`), and -- unless `push_url` is False, e.g. when the
        URL's what asked for it -- push it to the URL as a new history entry.
        """
        row = self._rows_by_key.get(key)
        if row is None or self._collections is None:
            with self._dialog:  # A UI context, even when called from a timer.
                ui.notify(f"No MPI collection starting at {key}.", type="warning")
            return
        self._active_key = key
        self._highlight(key)
        if push_url:
            _push_url_collection(self._dialog, key)
        if not self._dialog.value:
            self._dialog.open()
        _install_wheel_handler(self._body)

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

        frames = await run.io_bound(
            mpi_strip.load_collection_frames, self._data_dir, row, self._collections
        )
        if frames is None or generation != self._load_generation:
            return  # Server shutting down, or another collection's been picked.
        if frames.is_empty():
            self._body.clear()
            with self._body:
                ui.label("No frames of this collection have been downlinked.").classes(
                    "text-grey"
                )
            return
        self._summary.set_text(_summary_text(row, frames))
        await self._viewer.show(
            frames, filename=f"mpi_collection_{started_at:%Y-%m-%d_%H%M%S}Z.png"
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

    def _on_hide(self) -> None:
        self._active_key = None
        self._highlight(None)
        if self._closing_from_url:
            self._closing_from_url = False
        else:
            _push_url_collection(self._dialog, None)

    async def _on_url_change(self, key: str) -> None:
        """The browser's back/forward buttons moved to a URL viewing `key`
        (a `collection_key`), or none (`""`).
        """
        if key and key != self._active_key:
            await self.show(key, push_url=False)
        elif not key and self._dialog.value:
            self._closing_from_url = True
            self._dialog.close()


def _push_url_collection(element: ui.element, key: str | None) -> None:
    """Push the page URL with its `COLLECTION_QUERY_PARAM` set to `key` (or
    removed, for None) as a new history entry -- unless it's already that.
    `element` is any element on the page, for its client.
    """
    element.client.run_javascript(
        f"""(() => {{
            const url = new URL(location.href);
            const key = {json.dumps(key)};
            if (url.searchParams.get('{COLLECTION_QUERY_PARAM}') === key) return;
            if (key === null) url.searchParams.delete('{COLLECTION_QUERY_PARAM}');
            else url.searchParams.set('{COLLECTION_QUERY_PARAM}', key);
            history.pushState(null, '', url);
        }})()"""
    )


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
        self._frames: pl.DataFrame | None = None
        self._frame_rows: list[dict[str, Any]] = []
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
        self._frames = None
        self._scroller = None
        self._body.clear()

    async def show(self, frames: pl.DataFrame, *, filename: str) -> None:
        """Switch to (and render) a collection's `frames`, keeping the
        current view settings; it starts scrolled to the start.
        """
        self._frames = frames
        self._frame_rows = frames.drop("pixels").to_dicts()
        self._filename = filename
        self._column_count = mpi_strip.column_count(frames)
        self._fit_zoom = max(
            1, math.ceil(_FIT_RENDER_WIDTH_PX / max(1, self._column_count))
        )
        self._scroller = None
        self._pending_anchor = None
        await self.render()

    async def render(self) -> None:
        if self._frames is None:
            return
        self._rendering = True
        try:
            await self._render(self._frames)
        finally:
            self._rendering = False

    async def _render(self, frames: pl.DataFrame) -> None:
        mode = self._mode_toggle.value
        fit = self._zoom_toggle.value == _FIT
        strip = await run.io_bound(
            mpi_strip.render_strip,
            frames,
            zoom=self._fit_zoom if fit else self._zoom_toggle.value,
            mode=mode,
            intensity_range=self._range_toggle.value,
        )
        if strip is None or frames is not self._frames:
            return  # Nothing to show, or switched collections meanwhile.
        # A Ctrl+scroll zoom keeps the point under the cursor in place;
        # anything else keeps whatever's mid-view there. Read just before
        # swapping strips, so scrolling done while this one rendered isn't lost.
        anchor, self._pending_anchor = self._pending_anchor, None
        if anchor is None and self._scroller is not None:
            anchor = await _scroll_center_anchor(self._scroller)
        self._body.clear()
        with self._body:
            self._scroller = _strip_body(
                strip, self._frame_rows, mode, self._filename, fit=fit
            )
        if anchor is not None:
            _scroll_to_anchor(self._scroller, anchor)

    def _on_wheel_zoom(self, detail: dict[str, Any]) -> None:
        """Step the zoom in/out at the cursor -- `detail` as dispatched by
        `_install_wheel_handler`.
        """
        if self._rendering or self._frames is None:
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
