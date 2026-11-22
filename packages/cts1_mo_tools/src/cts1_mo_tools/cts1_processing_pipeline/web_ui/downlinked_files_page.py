"""The "Satellite Files" page: one table of every file known to be on the
satellite or reassembled from bulk downlinks (step 11's `downlinked_files`),
and -- with `?file=<file key>` -- one reassembled file's viewer: its
image/text/hex preview, byte coverage map, and every downlink of it. Plus
the "Image Gallery" page, of the images among them. See `downlinked_files`
for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = [
    "IMAGE_GALLERY_PATH",
    "SATELLITE_FILES_PATH",
    "build_image_gallery_page",
    "build_satellite_files_page",
    "file_page_url",
]

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime elsewhere
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import orjson
from nicegui import run, ui

from . import downlinked_files
from .file_reassembler_page import coverage_map
from .layout import page_shell

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

SATELLITE_FILES_PATH = "/satellite-files"
IMAGE_GALLERY_PATH = "/image-gallery"
GLOBE_VIEWER_URL = "https://jonathe0.github.io/Sat-1-Photos/globe/"

# How much of a file the viewer shows as text or hex.
TEXT_PREVIEW_BYTES = 64 * 1024
HEX_PREVIEW_BYTES = 4 * 1024
HEX_BYTES_PER_LINE = 16

# Rendering a coverage map is one pixel per byte; past this it's more
# scrolling than information.
MAX_COVERAGE_MAP_BYTES = 4 * 1024 * 1024

TABLE_ROWS_PER_PAGE = 50
GALLERY_THUMBNAIL_HEIGHT_PX = 160

_STATUS_COLORS = {
    "Verified": "positive",
    "Complete": "positive",
    "SHA-256 mismatch": "negative",
    "Conflicting": "warning",
    "Partial": "grey",
    "Not downlinked": "grey",
}

_LISTING_STATUS_TEXT = {
    "listed": "Listed",
    "not_in_latest_listing": "Gone from latest listing",
    "newer_than_latest_listing": "Newer than latest listing",
    "never_listed": "Never listed",
}

# How each chunk was tied to its file -- see `step_11_reassemble_downlinked_files`.
_ATTRIBUTION_TEXT = {
    "header": "header",
    "beacon": "beacon",
    "next_beacon": "next beacon",
    "time_window": "nearby (filled a gap)",
    "unattributed": "unattributed",
}

_SOURCE_TEXT = {
    "listing": "listing",
    "bulk_downlink_header": "downlink header",
    "adcs_file_header": "ADCS file header",
    "file_map": "file map",
    "mpi_analysis": "MPI analysis",
    "other": "other",
}


def file_page_url(file_key: str) -> str:
    """The viewer for one reassembled file."""
    return f"{SATELLITE_FILES_PATH}?{urlencode({'file': file_key})}"


def _when(value: datetime | None) -> str:
    return "" if value is None else f"{value:%Y-%m-%d %H:%M:%S}"


def _size(value: int | None) -> str:
    return "" if value is None else f"{value:,} B"


def _json_list_text(value: str | None, names: dict[str, str] | None = None) -> str:
    items = [str(item) for item in (orjson.loads(value) if value else [])]
    return ", ".join((names or {}).get(item, item) for item in items)


def _loading() -> None:
    with ui.row().classes("items-center gap-2 p-2"):
        ui.spinner(size="md")
        ui.label("Loading...").classes("text-grey")


def _not_run_yet(step: str) -> None:
    ui.label(f"Nothing yet -- run the pipeline (through {step}) first.").classes(
        "text-caption text-grey mt-2"
    )


def _search_input(table: ui.table, placeholder: str) -> None:
    ui.input(placeholder=placeholder).props("dense clearable outlined").classes(
        "w-72"
    ).bind_value_to(table, "filter")


def _link_cell_slot(href_field: str, icon: str, tooltip: str) -> str:
    return (
        f'<q-td :props="props" auto-width><q-btn v-if="props.row.{href_field}" '
        f'flat round dense size="sm" icon="{icon}" type="a" '
        f':href="props.row.{href_field}"><q-tooltip>{tooltip}</q-tooltip>'
        "</q-btn></q-td>"
    )


def _status_cell_slot() -> str:
    colors = ", ".join(f"'{k}': '{v}'" for k, v in _STATUS_COLORS.items())
    return (
        f'<q-td :props="props"><q-badge v-if="props.value" '
        f":color=\"({{{colors}}})[props.value] || 'grey'\" "
        ':label="props.value" /></q-td>'
    )


# -- Gallery -------------------------------------------------------------------


def _gallery_card(images: pl.DataFrame | None, order: str, *, descending: bool) -> None:
    with ui.card().classes("w-full"):
        if images is None:
            _loading()
            return
        if images.is_empty():
            _not_run_yet("step_11")
            return

        with ui.element("div").classes(
            "w-full grid gap-3 grid-cols-[repeat(auto-fill,minmax(180px,1fr))]"
        ):
            for row in downlinked_files.sort_gallery_images(
                images, order, descending=descending
            ).to_dicts():
                _gallery_tile(row)
        ui.label(f"{len(images):,} image(s).").classes("text-caption text-grey")


def _gallery_tile(row: dict[str, Any]) -> None:
    status = downlinked_files.status_text(row)
    with (
        ui.link(target=file_page_url(row["file_key"])).classes(
            "no-underline text-inherit"
        ),
        ui.card().tight().classes("w-full h-full hover:shadow-lg"),
    ):
        ui.image(downlinked_files.file_preview_url(row["file_key"])).props(
            'loading="lazy" fit="contain" no-spinner'
        ).classes("w-full bg-grey-2").style(f"height: {GALLERY_THUMBNAIL_HEIGHT_PX}px")
        with ui.column().classes("p-2 gap-0 w-full"):
            ui.label(row["file_path"]).classes("text-sm font-medium break-all")
            if timestamp := row["file_timestamp"]:
                ui.label(f"{_when(timestamp)} UTC").classes("text-caption")
            with ui.row().classes("items-center gap-2"):
                ui.badge(status, color=_STATUS_COLORS.get(status, "grey"))
                ui.label(f"{row['received_fraction']:.0%}").classes(
                    "text-caption text-grey"
                )


# -- Files table ---------------------------------------------------------------

_FILES_COLUMNS: tuple[dict[str, Any], ...] = (
    {"name": "_open", "label": "", "field": "_open"},
    {"name": "file_path", "label": "File", "field": "file_path", "align": "left"},
    {"name": "content_kind", "label": "Kind", "field": "content_kind", "align": "left"},
    {
        "name": "file_timestamp",
        "label": "File Timestamp (UTC)",
        "field": "file_timestamp",
    },
    {"name": "status", "label": "Downlinked", "field": "status", "align": "left"},
    {
        "name": "received_fraction",
        "label": "Received",
        "field": "received_fraction",
        ":format": "v => v == null ? '' : (v * 100).toFixed(1) + '%'",
    },
    {
        "name": "size",
        "label": "Size",
        "field": "size",
        ":format": "v => v == null ? '' : v.toLocaleString() + ' B'",
    },
    {"name": "listing", "label": "Listing", "field": "listing", "align": "left"},
    {"name": "downlink_count", "label": "Downlinks", "field": "downlink_count"},
    {
        "name": "rejected_downlink_count",
        "label": "Rejected",
        "field": "rejected_downlink_count",
    },
    {"name": "sources", "label": "Seen In", "field": "sources", "align": "left"},
    {"name": "mention_count", "label": "Mentions", "field": "mention_count"},
    {"name": "last_seen", "label": "Last Seen (UTC)", "field": "last_seen"},
    {"name": "last_listed", "label": "Last Listed (UTC)", "field": "last_listed"},
    {
        "name": "last_downlinked",
        "label": "Last Downlinked (UTC)",
        "field": "last_downlinked",
    },
    {"name": "sha256", "label": "SHA-256", "field": "sha256", "align": "left"},
)

_MENTIONS_CELL_SLOT = r"""
    <q-td :props="props" auto-width>
        <q-btn flat round dense size="sm" icon="history"
            @click="$parent.$emit('show_mentions', props.row.file_path)">
            <q-tooltip>Every response that named this file</q-tooltip>
        </q-btn>
    </q-td>
"""


def _files_card(data_dir: Path, files: pl.DataFrame | None) -> None:
    with ui.card().classes("w-full"):
        ui.label("Files").classes("text-lg font-bold")
        ui.label(
            "Every file named in a telecommand response (step 10: directory "
            "listings, downlink headers, file maps and MPI analyses), and "
            "every file reassembled from bulk downlinks (step 11). A chunk is "
            "tied to its file by the downlink header before it or the "
            'beacons\' path hash. "Listing" compares a file with the latest complete '
            "listing of its directory that covers its name (a listing may be "
            'just one page). "File Timestamp" is when a camera image was '
            "captured, an MPI recording started or an ADCS file was last "
            "modified, by the satellite's clocks."
        ).classes("text-caption text-grey")
        if files is None:
            _loading()
            return
        if files.is_empty():
            _not_run_yet("step_11")
            return

        rows = [
            {
                "file_key": row["file_key"],
                "file_path": row["file_path"],
                "_open": file_page_url(row["file_key"])
                if row["downlink_count"]
                else None,
                "content_kind": row["content_kind"] or "",
                "file_timestamp": _when(row["file_timestamp"]),
                "status": downlinked_files.status_text(row),
                "received_fraction": row["received_fraction"],
                "size": row["file_size"] or row["byte_length"],
                "listing": _LISTING_STATUS_TEXT.get(
                    row["listing_status"], row["listing_status"] or ""
                ),
                "downlink_count": row["downlink_count"],
                "rejected_downlink_count": row["rejected_downlink_count"],
                "sources": _json_list_text(row["sources"], _SOURCE_TEXT),
                "mention_count": row["mention_count"],
                "last_seen": _when(row["last_seen_at"]),
                "last_listed": _when(row["last_listed_at"]),
                "last_downlinked": _when(row["last_downlinked_at"]),
                "sha256": (row["reported_sha256"] or row["sha256"] or "")[:12],
            }
            for row in files.to_dicts()
        ]

        search = ui.row()
        table = ui.table(
            columns=[
                {"name": "_mentions", "label": "", "field": "_mentions"},
                *({**c, "sortable": c["name"] != "_open"} for c in _FILES_COLUMNS),
            ],
            rows=rows,
            row_key="file_key",
            pagination={"rowsPerPage": TABLE_ROWS_PER_PAGE},
        ).classes("w-full")
        table.props("dense flat")
        table.add_slot("body-cell-_mentions", _MENTIONS_CELL_SLOT)
        table.add_slot(
            "body-cell-_open",
            _link_cell_slot("_open", "visibility", "Open the downlinked file"),
        )
        table.add_slot("body-cell-status", _status_cell_slot())
        table.on("show_mentions", lambda e: _show_mentions(data_dir, e.args))
        with search:
            _search_input(table, "Search files")

        downlinked = sum(1 for r in rows if r["downlink_count"])
        verified = sum(1 for r in rows if r["status"] == "Verified")
        ui.label(
            f"{len(rows):,} file(s): {downlinked:,} downlinked ({verified:,} "
            f"verified by SHA-256), {len(rows) - downlinked:,} not."
        ).classes("text-caption text-grey")


async def _show_mentions(data_dir: Path, file_path: str) -> None:
    mentions = await run.io_bound(
        downlinked_files.load_file_mentions, data_dir, file_path
    )
    if mentions is None:
        return
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-5xl"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(file_path).classes("text-lg font-bold break-all")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")
        columns = [
            ("seen_at", "Received (UTC)"),
            ("source", "Source"),
            ("action", "Action"),
            ("file_size", "Size"),
            ("sha256", "SHA-256"),
            ("raw_path", "As Named"),
            ("response_complete", "Complete"),
        ]
        ui.table(
            columns=[
                {"name": f, "label": label, "field": f, "align": "left"}
                for f, label in columns
            ],
            rows=[
                {
                    "_i": i,
                    "seen_at": _when(row["seen_at"]),
                    "source": _SOURCE_TEXT.get(row["source"], row["source"]),
                    "action": row["action"] or "",
                    "file_size": _size(row["file_size"]),
                    "sha256": (row["sha256"] or "")[:16],
                    "raw_path": row["raw_path"],
                    "response_complete": "yes" if row["response_complete"] else "no",
                }
                for i, row in enumerate(mentions.to_dicts())
            ],
            row_key="_i",
            pagination={"rowsPerPage": 20},
        ).classes("w-full").props("dense flat")
    dialog.open()


# -- One file's viewer ---------------------------------------------------------


def _hex_dump(data: bytes) -> str:
    lines = []
    for offset in range(0, len(data), HEX_BYTES_PER_LINE):
        chunk = data[offset : offset + HEX_BYTES_PER_LINE]
        hex_part = " ".join(f"{b:02x}" for b in chunk).ljust(HEX_BYTES_PER_LINE * 3)
        text = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)  # noqa: PLR2004
        lines.append(f"{offset:08x}  {hex_part} {text}")
    return "\n".join(lines)


def _summary_row(row: dict[str, Any]) -> None:
    status = downlinked_files.status_text(row)
    expected = orjson.loads(row["expected_sha256s"] or "[]")
    facts = [
        ("Size", _size(row["byte_length"])),
        ("File timestamp (UTC)", _when(row.get("file_timestamp")) or "unknown"),
        (
            "Received",
            f"{row['received_fraction']:.1%} "
            f"({row['good_bytes']:,} good, {row['missing_bytes']:,} missing, "
            f"{row['conflict_bytes']:,} conflicting)",
        ),
        (
            "Downlinks",
            f"{row['downlink_count']:,}"
            + (
                f" (+{row['rejected_downlink_count']:,} rejected)"
                if row["rejected_downlink_count"]
                else ""
            )
            + f", {row['chunk_count']:,} chunks"
            + (
                f" ({row['rescued_chunk_count']:,} filling gaps from nearby)"
                if row.get("rescued_chunk_count")
                else ""
            ),
        ),
        ("First downlinked", _when(row["first_downlinked_at"])),
        ("Last downlinked", _when(row["last_downlinked_at"])),
        ("SHA-256", row["sha256"]),
        ("Satellite's SHA-256", ", ".join(expected) or "never reported"),
    ]
    with ui.row().classes("items-center gap-2"):
        ui.badge(status, color=_STATUS_COLORS.get(status, "grey")).classes("text-sm")
    with ui.grid(columns="auto 1fr").classes("gap-x-4 gap-y-1 text-sm"):
        for label, value in facts:
            ui.label(label).classes("text-grey")
            ui.label(value).classes("break-all")


def _preview_section(file: downlinked_files.DownlinkedFile) -> None:
    row = file.row
    if row["preview_media_type"]:
        with ui.card().classes("w-full"):
            ui.label(row["preview_label"]).classes("text-lg font-bold")
            url = downlinked_files.file_preview_url(row["file_key"])
            ui.image(url).props("fit=contain").classes("max-w-full").style(
                "max-height: 80vh"
            )
            ui.button("Open image", icon="open_in_new").props(
                f'type=a href="{url}" target=_blank flat dense'
            )
    if row["content_kind"] == "text":
        text = (
            file.data[:TEXT_PREVIEW_BYTES]
            .decode("utf-8", errors="replace")
            .replace("\x00", "\N{SYMBOL FOR NULL}")
        )
        with ui.card().classes("w-full"):
            ui.label("Text").classes("text-lg font-bold")
            if len(file.data) > TEXT_PREVIEW_BYTES:
                ui.label(
                    f"The first {TEXT_PREVIEW_BYTES // 1024} KiB. Missing bytes "
                    "show as ␀."
                ).classes("text-caption text-grey")
            ui.code(text, language="text").classes("w-full max-h-[60vh] overflow-auto")
    elif row["content_kind"] == "binary":
        with ui.card().classes("w-full"):
            ui.label("Hex").classes("text-lg font-bold")
            ui.label(
                f"The first {min(len(file.data), HEX_PREVIEW_BYTES):,} bytes "
                "(missing bytes read as 00 -- see the coverage map)."
            ).classes("text-caption text-grey")
            ui.code(_hex_dump(file.data[:HEX_PREVIEW_BYTES]), language="text").classes(
                "w-full max-h-[60vh] overflow-auto"
            )


_DOWNLINK_COLUMNS = (
    ("started_at", "Started (UTC)"),
    ("ended_at", "Ended (UTC)"),
    ("attributions", "Tied By"),
    ("chunk_count", "Chunks"),
    ("offsets", "Offsets"),
    ("bytes_received", "Bytes"),
    ("header", "Header"),
    ("rejected_reason", "Rejected Because"),
)


def _downlinks_card(downlinks: pl.DataFrame) -> None:
    with ui.card().classes("w-full"):
        ui.label("Downlinks").classes("text-lg font-bold")
        ui.label(
            "Each run of this file's chunks. A rejected downlink disagreed "
            "with the rest of the file (before firmware issue #653's fix, a "
            "new downlink could carry the previous file's bytes), and is left "
            'out of it. One tied "nearby" was unattributed chunks received '
            "within minutes of this file's downlinks, taken only to fill its "
            "gaps without disagreeing with any byte (and, from across a "
            "header, only if that completed the file with the satellite's "
            "SHA-256)."
        ).classes("text-caption text-grey")
        rows = [
            {
                "_i": i,
                "started_at": _when(row["started_at"]),
                "ended_at": _when(row["ended_at"]),
                "attributions": _json_list_text(row["attributions"], _ATTRIBUTION_TEXT),
                "chunk_count": f"{row['chunk_count']:,}",
                "offsets": f"{row['offset_min']:,} to {row['offset_end']:,}",
                "bytes_received": f"{row['bytes_received']:,}",
                "header": (
                    f"{row['header_action']} @ {_when(row['header_received_at'])}"
                    if row["header_action"]
                    else ""
                ),
                "rejected_reason": row["rejected_reason"] or "",
            }
            for i, row in enumerate(downlinks.to_dicts())
        ]
        ui.table(
            columns=[
                {"name": f, "label": label, "field": f, "align": "left"}
                for f, label in _DOWNLINK_COLUMNS
            ],
            rows=rows,
            row_key="_i",
            pagination={"rowsPerPage": 20},
        ).classes("w-full").props("dense flat")


def _file_viewer(
    file: downlinked_files.DownlinkedFile | None,
    downlinks: pl.DataFrame,
    file_key: str,
) -> None:
    if file is None:
        ui.label(f"No downlinked file {file_key!r} -- it may be from an older run.")
        return
    row = file.row
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(row["file_path"]).classes("text-xl font-bold break-all")
            ui.button(
                f"Download {downlinked_files.download_filename(row)}", icon="download"
            ).props(
                f'type=a href="{downlinked_files.file_download_url(row["file_key"])}"'
            )
        _summary_row(row)
    _preview_section(file)
    with ui.card().classes("w-full"):
        ui.label("Byte Coverage").classes("text-lg font-bold")
        if file.result.span_bytes > MAX_COVERAGE_MAP_BYTES:
            ui.label("Too big to map.").classes("text-caption text-grey")
        else:
            coverage_map(file.result)
    _downlinks_card(downlinks)


# -- The page ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ListData:
    files: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _ViewerData:
    file: downlinked_files.DownlinkedFile | None
    downlinks: pl.DataFrame


def _load_list_data(data_dir: Path) -> _ListData:
    """Blocking -- run it via `run.io_bound`."""
    return _ListData(
        files=downlinked_files.load_downlinked_files(data_dir),
    )


def _load_viewer_data(data_dir: Path, file_key: str) -> _ViewerData:
    """Blocking -- run it via `run.io_bound`."""
    return _ViewerData(
        file=downlinked_files.load_downlinked_file(data_dir, file_key),
        downlinks=downlinked_files.load_file_downlink_time_periods(data_dir, file_key),
    )


def build_satellite_files_page(data_dir: Path, *, file_key: str | None = None) -> None:
    """The page -- the list of files, or with `file_key` that file's viewer."""

    @ui.refreshable
    def content(data: _ListData | _ViewerData | None) -> None:
        if file_key is not None:
            if data is None:
                _loading()
            else:
                assert isinstance(data, _ViewerData)
                _file_viewer(data.file, data.downlinks, file_key)
            return
        list_data = data if isinstance(data, _ListData) else None
        _files_card(data_dir, list_data and list_data.files)

    # Loads in a thread, then rebuilds synchronously -- see
    # `pipeline_status_page.build_pipeline_status_page._reload`. A None
    # result is only when the server's shutting down.
    async def _reload() -> None:
        data = await (
            run.io_bound(_load_viewer_data, data_dir, file_key)
            if file_key is not None
            else run.io_bound(_load_list_data, data_dir)
        )
        if data is not None:
            content.refresh(data)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                if file_key is not None:
                    ui.button(icon="arrow_back").props(
                        f'flat round type=a href="{SATELLITE_FILES_PATH}"'
                    ).tooltip("All files")
                ui.label("Satellite Files").classes("text-2xl font-bold")
            ui.button("Refresh", icon="refresh", on_click=_reload)
        content(None)

    ui.timer(0, _reload, once=True)


def build_image_gallery_page(data_dir: Path) -> None:
    """Every image among the reassembled files, linking to each one's viewer."""

    @ui.refreshable
    def content(images: pl.DataFrame | None) -> None:
        _gallery_card(images, sort_select.value, descending=direction_select.value)

    async def _reload() -> None:
        images = await run.io_bound(downlinked_files.load_gallery_images, data_dir)
        if images is not None:
            content.refresh(images)

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Image Gallery").classes("text-2xl font-bold")
            with ui.row().classes("items-center gap-2"):
                sort_select = ui.select(
                    list(downlinked_files.GALLERY_SORTS),
                    value=next(iter(downlinked_files.GALLERY_SORTS)),
                    label="Sort by",
                    on_change=lambda _: content.refresh(),
                ).classes("w-48")
                direction_select = ui.select(
                    {True: "Descending", False: "Ascending"},
                    value=True,
                    label="Direction",
                    on_change=lambda _: content.refresh(),
                ).classes("w-36")
                ui.button("View on Globe", icon="open_in_new").props(
                    f'type=a href="{GLOBE_VIEWER_URL}" target=_blank flat'
                )
                ui.button("Refresh", icon="refresh", on_click=_reload)
        content(None)

    ui.timer(0, _reload, once=True)
