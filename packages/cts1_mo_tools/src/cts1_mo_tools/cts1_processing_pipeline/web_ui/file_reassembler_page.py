"""The "File Reassembler" page: reassemble a downlinked bulk file from
`BULK_FILE_DOWNLINK` packet chunks over one or more selected UTC time
ranges -- see `file_reassembly` for the actual reassembly/decoding logic.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_file_reassembler_page"]

import base64
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from . import data as beacon_data
from .file_reassembly import (
    COVERAGE_ROW_WIDTH_BYTES,
    BulkHeaderCandidate,
    ReassemblyResult,
    detect_picam_image,
    find_header_candidates,
    reassemble_bulk_chunks,
    render_coverage_png,
)
from .layout import page_shell

if TYPE_CHECKING:
    from collections.abc import Callable

    from nicegui import events


@dataclass
class _TimeRangeRow:
    """One (mutable) start/end pair the user is editing in the range list."""

    start: str | None = None
    end: str | None = None


RANGE_INPUT_MASK = "####-##-## ##:##:##"
RANGE_INPUT_PLACEHOLDER = "YYYY-MM-DD HH:MM:SS"
RANGE_INPUT_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_range_input(value: datetime) -> str:
    return value.strftime(RANGE_INPUT_FORMAT)


def _default_time_range_row() -> _TimeRangeRow:
    """The last 24h (UTC, up to now) -- a reasonable starting guess so the
    page shows something on first load instead of an empty selection.
    """
    now = datetime.now(UTC)
    return _TimeRangeRow(
        start=_format_range_input(now - timedelta(hours=24)),
        end=_format_range_input(now),
    )


def _parse_range_input(value: str | None) -> datetime | None:
    """Parse a `RANGE_INPUT_MASK`-shaped value ("YYYY-MM-DD HH:MM:SS"),
    treated as UTC (like every other timestamp in this dashboard).

    A plain masked text input rather than `<input type=datetime-local>` is
    deliberate: that native picker's on-screen *display* follows the
    browser/OS locale (day-first vs. month-first, 12h vs. 24h), even though
    its underlying value is always ISO 8601 -- ambiguous exactly where a
    ground-station operator can least afford it. The mask forces
    YYYY-MM-DD/24h ordering no matter whose browser this runs in.

    Returns None for anything that doesn't (yet) parse -- an in-progress or
    invalid edit is just treated the same as an empty row rather than
    raising, since this runs on every keystroke.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _validate_datetime_field(value: str | None) -> str | None:
    """A `ui.input` `validation` callable for a lone Start/End field: an
    empty value is fine (that row just isn't contributing to the selection
    yet), but a non-empty one that doesn't parse gets flagged rather than
    silently doing nothing, per `RANGE_INPUT_MASK`.
    """
    if value and _parse_range_input(value) is None:
        return f"Expected {RANGE_INPUT_PLACEHOLDER}"
    return None


def _make_end_validator(row: _TimeRangeRow) -> Callable[[str | None], str | None]:
    """Like `_validate_datetime_field`, plus flagging an End that's before
    its row's Start. A factory (rather than a closure written inline in the
    row loop) so each row's validator is bound to *that* row's `row`
    object, not whatever the loop variable last pointed at.
    """

    def _validate(value: str | None) -> str | None:
        error = _validate_datetime_field(value)
        if error is not None:
            return error
        start = _parse_range_input(row.start)
        end = _parse_range_input(value)
        if start is not None and end is not None and end < start:
            return "End must be after Start"
        return None

    return _validate


def _valid_ranges(rows: list[_TimeRangeRow]) -> list[tuple[datetime, datetime]]:
    """Every row with both a start and an end, start <= end. Rows the user
    hasn't finished filling in (or got backwards) are silently skipped
    rather than erroring -- they just don't contribute to the selection yet.
    """
    ranges = []
    for row in rows:
        start = _parse_range_input(row.start)
        end = _parse_range_input(row.end)
        if start is not None and end is not None and start <= end:
            ranges.append((start, end))
    return ranges


def _byte_range_str(start: int, end: int) -> str:
    return f"{start:,}-{end:,} ({end - start:,} bytes)"


def _duplicates_status(result: ReassemblyResult) -> None:
    """The pass/fail line above the chunks table -- the table itself always
    renders (see `_build_chunks_table`) regardless of which of these
    applies, so it's still there to inspect even mid-download, before every gap is
    filled in.
    """
    if not result.duplicates:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("Every offset appears exactly once.").classes("text-positive")
    elif result.has_conflicts:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label(
                f"{len(result.conflicts)} offset(s) have CONFLICTING "
                "content -- narrow the time range(s) until these agree "
                "before trusting the reassembled bytes below (the other "
                f"{len(result.duplicates) - len(result.conflicts)} "
                "duplicated offset(s) agree and aren't a problem)."
            ).classes("text-negative")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.icon("info", color="grey")
            ui.label(
                f"{len(result.duplicates)} offset(s) were received more "
                "than once (e.g. a retransmission), but every copy agrees "
                "-- not a problem."
            ).classes("text-caption text-grey")


CHUNKS_TABLE_PAGE_SIZE = 100


def _build_chunks_table(result: ReassemblyResult) -> Callable[[], None]:
    """Build a paginated, independently-refreshable chunks table.

    A file's chunks (`result.offsets`) are already fully computed
    server-side by this point -- reassembling the bytes needs every one of
    them regardless -- but a 20+ MB file is 100,000+ 195-byte chunks, and
    shipping that many rows to the browser at once (even paginated
    client-side by Quasar, which still means the whole row set travels over
    the websocket first) would be its own problem. Paging *here*, before
    anything is handed to `ui.table`, keeps each page turn down to
    `CHUNKS_TABLE_PAGE_SIZE` rows -- flipping pages only re-renders this one
    component (via the returned callable's `.refresh()`), not the whole
    results section, so it never re-fetches or re-reassembles anything.
    """
    page_state = {"index": 0}

    @ui.refreshable
    def chunks_table() -> None:
        total = len(result.offsets)
        if total == 0:
            return
        total_pages = -(-total // CHUNKS_TABLE_PAGE_SIZE)  # ceil div
        page_state["index"] = max(0, min(page_state["index"], total_pages - 1))
        start = page_state["index"] * CHUNKS_TABLE_PAGE_SIZE
        page_offsets = result.offsets[start : start + CHUNKS_TABLE_PAGE_SIZE]

        columns = [
            {"name": "offset", "label": "Offset", "field": "offset"},
            {"name": "length", "label": "Length", "field": "length"},
            {"name": "count", "label": "Copies", "field": "count"},
            {
                "name": "distinct_contents_count",
                "label": "Distinct Contents Count",
                "field": "distinct_contents_count",
            },
            {"name": "consistent", "label": "Chunk Status", "field": "consistent"},
        ]
        rows = [
            {
                "offset": o.offset,
                "length": ", ".join(str(n) for n in o.lengths),
                "count": o.count,
                "distinct_contents_count": o.distinct_contents_count,
                "consistent": "Good" if o.consistent else "Conflicting",
            }
            for o in page_offsets
        ]
        table = ui.table(columns=columns, rows=rows, row_key="offset").classes("w-full")
        # Color "Chunk Status" the same green/yellow as the coverage map above
        # -- ties this table's rows back to the graphic's blocks at a glance.
        table.add_slot(
            "body-cell-consistent",
            r"""
                <q-td :props="props"
                    :class="props.value === 'Good' ? 'text-positive' : 'text-warning'">
                    {{ props.value }}
                </q-td>
            """,
        )

        if total_pages == 1:
            ui.label(f"{total:,} distinct offset(s).").classes("text-caption text-grey")
            return

        def _turn_page(delta: int) -> None:
            page_state["index"] += delta
            chunks_table.refresh()

        with ui.row().classes("w-full items-center justify-between mt-2"):
            ui.label(
                f"Showing {start + 1:,}-{min(start + CHUNKS_TABLE_PAGE_SIZE, total):,} "
                f"of {total:,} distinct offset(s)."
            ).classes("text-caption text-grey")
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="chevron_left", on_click=lambda: _turn_page(-1)).props(
                    "flat dense round"
                ).set_enabled(page_state["index"] > 0)
                ui.label(f"Page {page_state['index'] + 1} / {total_pages}")
                ui.button(icon="chevron_right", on_click=lambda: _turn_page(1)).props(
                    "flat dense round"
                ).set_enabled(page_state["index"] < total_pages - 1)

    return chunks_table


COVERAGE_BLOCK_WIDTH_PX = 3  # on-screen width of one byte's block; may change later
COVERAGE_BLOCK_HEIGHT_PX = 2  # shorter than wide -- rows are the scarce vertical space


def _coverage_map(result: ReassemblyResult) -> None:
    """A byte-coverage map: one block per byte, green if covered, red if
    still a gap, yellow if covered but conflicting (multiple disagreeing
    values), `COVERAGE_ROW_WIDTH_BYTES` blocks per row.

    `render_coverage_png` encodes one *pixel* per byte -- small and cheap to
    ship even for a 20+ MB file -- and the on-screen block size is applied
    here purely with CSS (`image-rendering: pixelated` keeps the block
    edges crisp instead of blurring the upscale). The block is shorter than
    it is wide on purpose: width already reads fine at
    `COVERAGE_BLOCK_WIDTH_PX`, and a file's row count (hence the image's
    total height) grows with its size, so trimming just the height keeps
    large files from needing as much scrolling without shrinking each row.
    """
    if result.span_bytes == 0:
        return
    data_uri = "data:image/png;base64," + base64.b64encode(
        render_coverage_png(result)
    ).decode("ascii")
    width_px = COVERAGE_ROW_WIDTH_BYTES * COVERAGE_BLOCK_WIDTH_PX
    row_count = -(-result.span_bytes // COVERAGE_ROW_WIDTH_BYTES)
    height_px = row_count * COVERAGE_BLOCK_HEIGHT_PX
    with ui.row().classes("w-full overflow-x-auto"):
        ui.image(data_uri).style(
            f"width: {width_px}px; height: {height_px}px; image-rendering: pixelated;"
        )
    with ui.row().classes("items-center gap-4"):
        for color_class, label in (
            ("text-positive", "covered"),
            ("text-negative", "missing (needs re-downlink)"),
            ("text-warning", "conflicting (multiple values)"),
        ):
            with ui.row().classes("items-center gap-1"):
                ui.icon("square", color=color_class.removeprefix("text-")).classes(
                    "text-xs"
                )
                ui.label(label).classes(f"text-caption {color_class}")


def _gaps_section(result: ReassemblyResult) -> None:
    """Byte ranges never received at all -- distinct from `_conflict_ranges_section`
    (ranges that *were* received, just with disagreeing values); the two are
    rendered as clearly separate blocks since they call for different next
    steps -- re-downlink one, narrow the time range for the other.
    """
    ui.label("Missing byte ranges").classes("text-base font-medium")
    if result.is_gapless:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("No gaps -- every byte in range is covered.").classes(
                "text-positive"
            )
        return

    with ui.row().classes("items-center gap-2"):
        ui.icon("warning", color="warning")
        ui.label(
            f"{len(result.gaps)} range(s) never received -- these need to be "
            "re-downlinked:"
        ).classes("text-warning")
    for start, end in result.gaps[:50]:
        ui.label(_byte_range_str(start, end)).classes("font-mono text-sm")
    if len(result.gaps) > 50:  # noqa: PLR2004
        ui.label(f"...and {len(result.gaps) - 50} more.").classes(
            "text-caption text-grey"
        )


def _conflict_ranges_section(result: ReassemblyResult) -> None:
    """Byte ranges that *were* received but with more than one disagreeing
    value -- see `_gaps_section` for the (deliberately separate) list of
    ranges never received at all.
    """
    ui.label("Conflicting byte ranges").classes("text-base font-medium")
    if not result.has_conflicts:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("No conflicts -- every received byte agrees.").classes(
                "text-positive"
            )
        return

    with ui.row().classes("items-center gap-2"):
        ui.icon("error", color="negative")
        ui.label(
            f"{len(result.conflict_ranges)} range(s) have multiple, "
            "disagreeing values -- narrow the time range(s) until these "
            "agree:"
        ).classes("text-negative")
    for start, end in result.conflict_ranges[:50]:
        ui.label(_byte_range_str(start, end)).classes("font-mono text-sm")
    if len(result.conflict_ranges) > 50:  # noqa: PLR2004
        ui.label(f"...and {len(result.conflict_ranges) - 50} more.").classes(
            "text-caption text-grey"
        )


def _sha256_comparison(result: ReassemblyResult, expected: str | None) -> None:
    ui.label(f"SHA-256 of assembled bytes: {result.sha256}").classes(
        "font-mono text-sm"
    )
    if not expected:
        ui.label("No SHA-256 to compare against -- pick a header below.").classes(
            "text-caption text-grey"
        )
    elif len(expected) < 64:  # noqa: PLR2004
        ui.label(
            f"Header's SHA-256 was truncated ({len(expected)}/64 hex chars) -- "
            "can't verify."
        ).classes("text-caption text-grey")
    elif expected == result.sha256:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("Matches the header's SHA-256.").classes("text-positive")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label("Does NOT match the header's SHA-256.").classes("text-negative")


def _best_named_candidate(
    candidates: list[BulkHeaderCandidate],
) -> BulkHeaderCandidate | None:
    """Whichever candidate looks most trustworthy: prefer one with a full
    (untruncated, 64-hex-char) SHA-256, then one with a known file_size,
    tie-broken by most recently received.
    """
    named = [c for c in candidates if c.file]
    if not named:
        return None

    def _score(c: BulkHeaderCandidate) -> tuple[int, int, datetime]:
        has_full_sha256 = c.sha256 is not None and len(c.sha256) == 64  # noqa: PLR2004
        return (int(has_full_sha256), int(c.file_size is not None), c.received_at)

    return max(named, key=_score)


def _is_truncated(candidate: BulkHeaderCandidate) -> bool:
    """Whether this candidate's SHA-256 looks cut off -- the only field
    truncation is currently detectable in (see the module docstring: a
    long file path pushes TCMD_RESPONSE's 186-byte cap into the sha256
    value before it's fully written).
    """
    return candidate.sha256 is not None and len(candidate.sha256) < 64  # noqa: PLR2004


@dataclass(frozen=True, slots=True)
class _HeaderGroup:
    """One or more consecutive (by `received_at`) candidates that all agree
    on action/file/size/crc16/sha256 -- distinct receipts of what's
    presumably the same underlying header, kept together as one row with
    the individual timestamps available on expand.
    """

    action: str | None
    file: str | None
    file_size: int | None
    crc16: str | None
    sha256: str | None
    truncated: bool
    members: tuple[BulkHeaderCandidate, ...]


def _group_key(
    c: BulkHeaderCandidate,
) -> tuple[str | None, str | None, int | None, str | None, str | None]:
    return (c.action, c.file, c.file_size, c.crc16, c.sha256)


def _group_consecutive_candidates(
    candidates_by_time: list[BulkHeaderCandidate],
) -> list[_HeaderGroup]:
    """Group *consecutive* (in `candidates_by_time`'s order) candidates that
    share action/file/file_size/crc16/sha256. Candidates with the same
    values but separated by a different one in between stay in separate
    groups -- e.g. two genuinely distinct downloads of the same file don't
    get merged just because they match.
    """
    groups: list[_HeaderGroup] = []
    for c in candidates_by_time:
        if groups and _group_key(groups[-1].members[-1]) == _group_key(c):
            groups[-1] = replace(groups[-1], members=(*groups[-1].members, c))
        else:
            groups.append(
                _HeaderGroup(
                    action=c.action,
                    file=c.file,
                    file_size=c.file_size,
                    crc16=c.crc16,
                    sha256=c.sha256,
                    truncated=_is_truncated(c),
                    members=(c,),
                )
            )
    return groups


# A `+`/`-` expand toggle per group row, native to NiceGUI/Quasar's own
# expandable-row pattern (custom header/body slots) rather than a separate
# `ui.expansion` per group -- keeps every group in the one table, and a
# single-member group (nothing more to reveal) just gets a blank cell
# instead of a button that would expand to repeat itself.
_HEADER_TABLE_HEADER_SLOT = r"""
    <q-tr :props="props">
        <q-th auto-width />
        <q-th v-for="col in props.cols" :key="col.name" :props="props">
            {{ col.label }}
        </q-th>
    </q-tr>
"""
_HEADER_TABLE_BODY_SLOT = r"""
    <q-tr :props="props">
        <q-td auto-width>
            <q-btn v-if="props.row.member_count > 1" size="sm" color="primary"
                round dense flat
                @click="props.expand = !props.expand"
                :icon="props.expand ? 'remove' : 'add'" />
        </q-td>
        <q-td v-for="col in props.cols" :key="col.name" :props="props"
            :class="col.name === 'sha256' ? 'ellipsis' : ''"
            :style="col.name === 'sha256' ? 'max-width: 220px' : ''">
            {{ col.value }}
        </q-td>
    </q-tr>
    <q-tr v-show="props.expand" :props="props">
        <q-td colspan="100%">
            <div v-for="ts in props.row.member_timestamps" :key="ts"
                class="text-left text-caption q-pl-lg">
                {{ ts }}
            </div>
        </q-td>
    </q-tr>
"""


def _header_candidates_table(candidates: list[BulkHeaderCandidate]) -> str | None:
    """Render the found header candidates -- sorted by Received, consecutive
    look-alike entries collapsed into one row expandable (via a `+`/`-`
    button) to its individual timestamps -- and return the SHA-256
    (possibly None/truncated) of whichever one looks most trustworthy, per
    `_best_named_candidate`.
    """
    if not candidates:
        ui.label(
            "No TCMD_RESPONSE header found in the selected range(s) -- that's "
            "fine, headers are sometimes missing entirely; just cross-check "
            "the file name/size/hash some other way."
        ).classes("text-caption text-grey")
        return None

    candidates_by_time = sorted(candidates, key=lambda c: c.received_at)
    groups = _group_consecutive_candidates(candidates_by_time)

    columns = [
        {"name": "received_at", "label": "Received", "field": "received_at"},
        {"name": "action", "label": "Action", "field": "action"},
        {"name": "file", "label": "File", "field": "file"},
        {"name": "file_size", "label": "Size (bytes)", "field": "file_size"},
        {"name": "crc16", "label": "CRC-16", "field": "crc16"},
        {"name": "sha256", "label": "SHA-256", "field": "sha256"},
        {"name": "truncated", "label": "Truncated?", "field": "truncated"},
    ]
    rows = [
        {
            "id": i,
            "received_at": (
                f"{group.members[0].received_at:%Y-%m-%d %H:%M:%S}"
                if len(group.members) == 1
                else f"{len(group.members)}x, "
                f"{group.members[0].received_at:%Y-%m-%d %H:%M:%S} - "
                f"{group.members[-1].received_at:%Y-%m-%d %H:%M:%S}"
            ),
            "action": group.action or "",
            "file": group.file or "",
            "file_size": f"{group.file_size:,}" if group.file_size is not None else "",
            "crc16": group.crc16 or "",
            "sha256": group.sha256 or "",
            "truncated": "yes" if group.truncated else "",
            "member_count": len(group.members),
            "member_timestamps": [
                f"{c.received_at:%Y-%m-%d %H:%M:%S}" for c in group.members
            ],
        }
        for i, group in enumerate(groups)
    ]
    table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
    table.add_slot("header", _HEADER_TABLE_HEADER_SLOT)
    table.add_slot("body", _HEADER_TABLE_BODY_SLOT)

    best = _best_named_candidate(candidates)
    return best.sha256 if best is not None else None


def _picam_image_section(jpg_bytes: bytes, filename_hint: str | None) -> None:
    """A detected PiCAM image: rendered inline, plus a JPG download button --
    see `file_reassembly.detect_picam_image` for the detection heuristic.
    """
    jpg_filename = (
        Path(filename_hint).with_suffix(".jpg").name
        if filename_hint
        else "picam_image.jpg"
    )
    with ui.card().classes("w-full"):
        ui.label("Detected PiCAM image").classes("text-lg font-bold")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode(
            "ascii"
        )
        ui.image(data_uri).classes("max-w-full")
        ui.button(
            "Download as JPG",
            icon="photo_camera",
            on_click=lambda: ui.download.content(jpg_bytes, filename=jpg_filename),
        )


def _reassembler_results(
    path: Path, ranges: list[tuple[datetime, datetime]], *, headers_only: bool = False
) -> None:
    if not ranges:
        ui.label(
            "No time range selected yet -- fill in a start and end above "
            "(an empty selection intentionally shows nothing, rather than "
            "every BULK_FILE_DOWNLINK packet ever decoded)."
        ).classes("text-caption text-grey")
        return

    tcmd_responses = beacon_data.load_tcmd_response_packets(path, ranges=ranges)
    candidates = find_header_candidates(tcmd_responses)

    with ui.card().classes("w-full"):
        ui.label("Header candidates").classes("text-lg font-bold")
        expected_sha256 = _header_candidates_table(candidates)

    if headers_only:
        return

    chunks = beacon_data.load_bulk_file_downlink_packets(path, ranges=ranges)

    with ui.card().classes("w-full"):
        ui.label("BULK_FILE_DOWNLINK chunks").classes("text-lg font-bold")
        if chunks.is_empty():
            ui.label("No BULK_FILE_DOWNLINK packets in the selected range(s).").classes(
                "text-caption text-grey"
            )
            return

        result = reassemble_bulk_chunks(chunks)
        ui.label(
            f"{result.total_chunks:,} packet(s), {result.unique_offsets:,} "
            f"distinct offset(s), spanning {result.span_bytes:,} bytes."
        ).classes("text-caption text-grey")

        _duplicates_status(result)
        _coverage_map(result)
        _build_chunks_table(result)()
        _gaps_section(result)
        ui.separator()
        _conflict_ranges_section(result)
        _sha256_comparison(result, expected_sha256)

        best = _best_named_candidate(candidates)
        # The full path (e.g. "ADCS/log_b51a.TLM"), not just the basename --
        # slashes become underscores since the browser would otherwise treat
        # them as directory separators in the downloaded filename.
        filename = (
            best.file.replace("/", "_") if best is not None and best.file else None
        )
        ui.button(
            "Download reassembled bytes",
            icon="download",
            on_click=lambda: ui.download.content(
                result.data, filename=filename or "reassembled_file.bin"
            ),
        )
        if result.has_conflicts or not result.is_gapless:
            ui.label(
                "Conflicting offsets and/or gaps are present -- the download "
                "will still work (missing bytes are filled with 0x00), but "
                "isn't a verified, complete copy of the file yet."
            ).classes("text-caption text-grey")

        picam_jpg = detect_picam_image(result.data)
        if picam_jpg is not None:
            _picam_image_section(picam_jpg, filename)


def build_file_reassembler_page(parquet_path: Path) -> None:
    rows: list[_TimeRangeRow] = [_default_time_range_row()]

    @ui.refreshable
    def range_editor() -> None:
        for i, row in enumerate(rows):
            with ui.row().classes("w-full items-center gap-2"):
                # A forward-reference box for the End input, filled in right
                # after it's created below -- `_set_start` only *calls*
                # into it once the user actually types (well after that
                # happens), but it's defined beforehand since Start renders
                # first. A plain loop-variable capture wouldn't work here:
                # every row's `_set_start` needs *its own* End input, not
                # whichever row's End happened to be created last.
                end_input_ref: dict[str, ui.input] = {}

                def _set_start(
                    e: events.ValueChangeEventArguments,
                    row: _TimeRangeRow = row,
                    end_input_ref: dict[str, ui.input] = end_input_ref,
                ) -> None:
                    row.start = e.value
                    # End's validity (the "must be after Start" half of it)
                    # depends on Start too, but NiceGUI only re-runs an
                    # input's own `validation` automatically on *its own*
                    # value changing -- ask it to re-check explicitly so a
                    # later Start edit doesn't leave a stale/missing error
                    # on End.
                    end_input_ref["input"].validate(return_result=False)

                def _set_end(e: events.ValueChangeEventArguments, row=row) -> None:  # noqa: ANN001
                    row.end = e.value

                ui.input(
                    "Start (UTC)",
                    value=row.start or "",
                    placeholder=RANGE_INPUT_PLACEHOLDER,
                    on_change=_set_start,
                    validation=_validate_datetime_field,
                ).props(f'mask="{RANGE_INPUT_MASK}"').classes("w-56")
                end_input_ref["input"] = (
                    ui.input(
                        "End (UTC)",
                        value=row.end or "",
                        placeholder=RANGE_INPUT_PLACEHOLDER,
                        on_change=_set_end,
                        validation=_make_end_validator(row),
                    )
                    .props(f'mask="{RANGE_INPUT_MASK}"')
                    .classes("w-56")
                )
                if len(rows) > 1:

                    def _remove(i: int = i) -> None:
                        rows.pop(i)
                        range_editor.refresh()

                    ui.button(icon="close", on_click=_remove).props("flat round dense")
        ui.button(
            "Add time range",
            icon="add",
            on_click=lambda: (rows.append(_TimeRangeRow()), range_editor.refresh()),
        ).props("flat")

    search_state = {"headers_only": False, "has_searched": False}

    @ui.refreshable
    def results() -> None:
        if not search_state["has_searched"]:
            # The range fields are pre-filled (see `_default_time_range_row`)
            # purely as a starting guess -- not run automatically, since
            # that'd mean every page load kicks off a query (potentially a
            # large one) before the user asked for anything.
            ui.label("Fields are pre-filled with the last 24h.").classes(
                "text-caption text-grey"
            )
            return
        _reassembler_results(
            parquet_path,
            _valid_ranges(rows),
            headers_only=search_state["headers_only"],
        )

    def _search(*, headers_only: bool) -> None:
        search_state["headers_only"] = headers_only
        search_state["has_searched"] = True
        results.refresh()

    with page_shell():
        ui.label("File Reassembler").classes("text-2xl font-bold")
        ui.label(
            "Pick UTC time ranges covering bulk file download(s). "
            "Then, filter to just a single file to reassemble it. "
            "Pro tip: Copy-and-paste time ranges from the Headers list."
        ).classes("text-caption text-grey")
        with ui.card().classes("w-full"):
            range_editor()
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Search (Headers only)",
                icon="search",
                on_click=lambda: _search(headers_only=True),
            ).props("outline")
            ui.button(
                "Search", icon="search", on_click=lambda: _search(headers_only=False)
            )
        results()
