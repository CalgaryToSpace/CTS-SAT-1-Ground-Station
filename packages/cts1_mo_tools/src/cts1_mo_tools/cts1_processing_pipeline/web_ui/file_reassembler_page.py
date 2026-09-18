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
from typing import TYPE_CHECKING, Any

from nicegui import ui

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

from . import data as beacon_data
from .file_reassembly import (
    COVERAGE_PACKETS_PER_ROW,
    COVERAGE_ROW_WIDTH_BYTES,
    DEFAULT_CONFLICT_POLICY,
    SHA256_HEX_LEN,
    BulkHeaderCandidate,
    ByteSegment,
    ByteStatus,
    ConflictPolicy,
    ReassemblyResult,
    coverage_png_size,
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


_STATUS_TEXT_CLASS = {
    ByteStatus.GOOD: "text-positive",
    ByteStatus.MISSING: "text-negative",
    ByteStatus.CONFLICTING: "text-warning",
}
_STATUS_ICON = {
    ByteStatus.GOOD: "check_circle",
    ByteStatus.MISSING: "warning",
    ByteStatus.CONFLICTING: "error",
}


def _assessment_summary(result: ReassemblyResult) -> None:
    """The headline per-byte verdict: how many of the file's bytes are good,
    missing, and conflicting, above the segment-by-segment breakdown.

    Deliberately phrased in *bytes*, not packets -- a single retransmitted
    packet that agrees isn't a problem, and a pair that disagree may only
    disagree about a handful of the bytes they overlap on, which counting
    packets would round up into a much scarier number than it is.
    """
    span = result.span_bytes
    counts = (
        (ByteStatus.GOOD, result.good_bytes),
        (ByteStatus.MISSING, result.missing_bytes),
        (ByteStatus.CONFLICTING, result.conflict_bytes),
    )
    with ui.row().classes("items-center gap-4"):
        for status, count in counts:
            percent = (100.0 * count / span) if span else 0.0
            with ui.row().classes("items-center gap-1"):
                ui.icon(
                    _STATUS_ICON[status],
                    color=_STATUS_TEXT_CLASS[status].removeprefix("text-"),
                )
                ui.label(f"{count:,} {status.lower()} ({percent:.1f}%)").classes(
                    _STATUS_TEXT_CLASS[status]
                )

    if result.is_complete:
        ui.label(
            "Every byte from 0 to the end of the download arrived, and every "
            "byte agrees."
        ).classes("text-caption text-positive")
        return

    next_steps = []
    if not result.is_gapless:
        next_steps.append(f"{len(result.gaps):,} missing range(s) need re-downlinking.")
    if result.has_conflicts:
        next_steps.append(
            f"{len(result.conflict_ranges):,} conflicting range(s) came down with "
            "more than one value -- either narrow the time range(s) until they "
            "agree, or pick a conflict resolution above and accept its answer."
        )
    ui.label(" ".join(next_steps)).classes("text-caption text-grey")


def _duplicates_note(result: ReassemblyResult) -> None:
    """A footnote about repeated offsets -- packet-level bookkeeping, kept
    well apart from the per-byte verdict above, since a duplicate offset is
    routine (a retransmission, or two ground stations catching the same
    overpass) and only matters at all if its copies actually disagree, which
    the byte assessment has already accounted for.
    """
    if not result.duplicates:
        return
    disagreeing = sum(1 for o in result.duplicates if not o.consistent)
    note = (
        f"{len(result.duplicates):,} offset(s) were received more than once"
        f" ({disagreeing:,} of them with differing payloads)."
        if disagreeing
        else f"{len(result.duplicates):,} offset(s) were received more than once, "
        "every copy agreeing."
    )
    ui.label(note).classes("text-caption text-grey")


def _copies_label(segment: ByteSegment) -> str:
    """How many packets carried each byte of `segment`: one number when
    every byte in the run arrived the same number of times, otherwise the
    range across it ("1-3").

    Worth showing next to the status because the two answer different
    questions: a `Good` run received once is right but has no corroboration,
    while one received three times is three packets that agree -- and a
    `Conflicting` run's count is how many disagreeing copies the resolution
    policy had to choose between.
    """
    if segment.min_copies == segment.max_copies:
        return f"{segment.min_copies:,}"
    return f"{segment.min_copies:,}-{segment.max_copies:,}"


SEGMENTS_TABLE_PAGE_SIZE = 100
_SEGMENT_FILTER_ALL = "All"

# Color the Status cell the same green/red/yellow as the coverage map above,
# tying each row back to the graphic's blocks at a glance.
_SEGMENTS_STATUS_CELL_SLOT = r"""
    <q-td :props="props" :class="props.row.status_class">{{ props.value }}</q-td>
"""


def _build_segments_table(result: ReassemblyResult) -> Callable[[], None]:
    """Build a paginated, filterable, independently-refreshable table of the
    file's byte segments -- consecutive runs of bytes sharing one status.

    Segments (not packets) are the unit here on purpose: a clean download of
    a 20 MB file is *one* row ("0-20,000,000: Good") instead of the 100,000+
    195-byte chunks it arrived as, and a messy one puts each missing or
    conflicting run in front of the operator as the single range it actually
    is. Pagination still happens *here*, before anything reaches `ui.table`,
    so a pathologically fragmented download can't ship more than
    `SEGMENTS_TABLE_PAGE_SIZE` rows over the websocket at once -- and
    flipping pages or changing the filter re-renders only this component
    (via the returned callable's `.refresh()`), never re-reassembling
    anything.
    """
    state: dict[str, Any] = {"index": 0, "status": _SEGMENT_FILTER_ALL}

    @ui.refreshable
    def segments_table() -> None:
        selected = state["status"]
        segments = [
            s for s in result.segments if selected in (_SEGMENT_FILTER_ALL, s.status)
        ]

        def _set_filter(value: str) -> None:
            state["status"] = value
            state["index"] = 0
            segments_table.refresh()

        with ui.row().classes("w-full items-center gap-2"):
            ui.select(
                [
                    _SEGMENT_FILTER_ALL,
                    *(
                        f"{status}"
                        for status in (
                            ByteStatus.MISSING,
                            ByteStatus.CONFLICTING,
                            ByteStatus.GOOD,
                        )
                    ),
                ],
                value=selected,
                label="Show",
                on_change=lambda e: _set_filter(e.value),
            ).classes("w-48")

        if not segments:
            ui.label(f"No {selected.lower()} byte ranges.").classes(
                "text-caption text-grey"
            )
            return

        total = len(segments)
        total_pages = -(-total // SEGMENTS_TABLE_PAGE_SIZE)  # ceil div
        state["index"] = max(0, min(state["index"], total_pages - 1))
        start = state["index"] * SEGMENTS_TABLE_PAGE_SIZE
        page = segments[start : start + SEGMENTS_TABLE_PAGE_SIZE]

        columns = [
            {"name": "start", "label": "First Byte", "field": "start"},
            {"name": "end", "label": "Last Byte", "field": "end"},
            {"name": "length", "label": "Length (bytes)", "field": "length"},
            {"name": "status", "label": "Status", "field": "status"},
            {
                "name": "copies",
                "label": "Copies Received",
                "field": "copies",
            },
        ]
        rows = [
            {
                "id": segment.start,
                "start": f"{segment.start:,}",
                # Inclusive, unlike `ByteSegment.end` -- "0-194" reads as the
                # range an operator would quote when asking for a re-downlink,
                # where a half-open "0-195" invites an off-by-one.
                "end": f"{segment.end - 1:,}",
                "length": f"{segment.length:,}",
                "status": f"{segment.status}",
                "status_class": _STATUS_TEXT_CLASS[segment.status],
                "copies": _copies_label(segment),
            }
            for segment in page
        ]
        table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
        table.add_slot("body-cell-status", _SEGMENTS_STATUS_CELL_SLOT)

        def _turn_page(delta: int) -> None:
            state["index"] += delta
            segments_table.refresh()

        with ui.row().classes("w-full items-center justify-between mt-2"):
            shown_to = min(start + SEGMENTS_TABLE_PAGE_SIZE, total)
            ui.label(
                f"Showing {start + 1:,}-{shown_to:,} of {total:,} byte range(s)."
            ).classes("text-caption text-grey")
            if total_pages > 1:
                with ui.row().classes("items-center gap-2"):
                    ui.button(
                        icon="chevron_left", on_click=lambda: _turn_page(-1)
                    ).props("flat dense round").set_enabled(state["index"] > 0)
                    ui.label(f"Page {state['index'] + 1} / {total_pages}")
                    ui.button(
                        icon="chevron_right", on_click=lambda: _turn_page(1)
                    ).props("flat dense round").set_enabled(
                        state["index"] < total_pages - 1
                    )

    return segments_table


# One byte is one screen pixel: the map's whole point is fitting a
# multi-MB download's shape on screen at once, which any zoom factor
# immediately spends. `image-rendering: pixelated` stays on so a browser
# zoom (or a HiDPI display's own scaling) shows hard block edges instead of
# blurring bytes into each other.
def _coverage_map(result: ReassemblyResult) -> None:
    """A byte-coverage map: one pixel per byte, green if good, red if
    missing, yellow if conflicting (received with multiple disagreeing
    values), `COVERAGE_ROW_WIDTH_BYTES` bytes per row.

    Both rulers -- packet boundaries across the top, byte offsets down the
    left -- are drawn into the PNG by `render_coverage_png` rather than
    overlaid as HTML here, so there's no way for a tick to end up a pixel
    off from the byte it points at. That leaves this with nothing to lay out
    but the image itself, at exactly the size it was encoded at.
    """
    if result.span_bytes == 0:
        return
    data_uri = "data:image/png;base64," + base64.b64encode(
        render_coverage_png(result)
    ).decode("ascii")
    width_px, height_px = coverage_png_size(result)
    with ui.row().classes("w-full overflow-x-auto"):
        ui.image(data_uri).style(
            f"width: {width_px}px; height: {height_px}px; image-rendering: pixelated;"
        )
    ui.label(
        f"One pixel per byte, {COVERAGE_ROW_WIDTH_BYTES:,} bytes "
        f"({COVERAGE_PACKETS_PER_ROW} packets) per row. Top ruler: byte offset "
        "within a row, ticked once per packet. Left: byte offset of the row."
    ).classes("text-caption text-grey")
    with ui.row().classes("items-center gap-4"):
        for status, label in (
            (ByteStatus.GOOD, "good"),
            (ByteStatus.MISSING, "missing (needs re-downlink)"),
            (ByteStatus.CONFLICTING, "conflicting (multiple values)"),
        ):
            color_class = _STATUS_TEXT_CLASS[status]
            with ui.row().classes("items-center gap-1"):
                ui.icon("square", color=color_class.removeprefix("text-")).classes(
                    "text-xs"
                )
                ui.label(label).classes(f"text-caption {color_class}")


PARTIAL_FILENAME_SUFFIX = "_partial"


def _full_sha256s(candidates: list[BulkHeaderCandidate]) -> set[str]:
    """Every distinct untruncated SHA-256 among the header candidates."""
    return {
        c.sha256
        for c in candidates
        if c.sha256 is not None and len(c.sha256) == SHA256_HEX_LEN
    }


def _partial_reason(
    result: ReassemblyResult,
    header: BulkHeaderCandidate | None,
    candidates: list[BulkHeaderCandidate],
) -> str | None:
    """Why what's been assembled isn't a verified, complete copy of the file
    -- or None if it is.

    Ordered worst-first, so the reason shown is the one worth acting on:
    bytes that never arrived, then bytes that arrived disagreeing, then the
    two checks that only a header can provide (the assembled span being the
    wrong size, and the hash matching none of the candidates' full SHA-256s).
    Anything but None gets `PARTIAL_FILENAME_SUFFIX` appended to the
    exported file's stem, so a
    half-assembled file can't be mistaken for the real thing later just
    because it was downloaded and filed away under the satellite's own name
    for it.
    """
    if result.span_bytes == 0:
        return "nothing was assembled"
    if not result.is_gapless:
        return f"{result.missing_bytes:,} byte(s) never arrived"
    if result.has_conflicts:
        return f"{result.conflict_bytes:,} byte(s) arrived with conflicting values"
    if (
        header is not None
        and header.file_size is not None
        and header.file_size != result.span_bytes
    ):
        return (
            f"the header's file_size ({header.file_size:,} bytes) doesn't match "
            f"the {result.span_bytes:,} byte(s) assembled"
        )
    full_hashes = _full_sha256s(candidates)
    if full_hashes and result.sha256 not in full_hashes:
        return "the SHA-256 doesn't match any header's"
    return None


def _mark_partial(filename: str, *, partial: bool) -> str:
    """`filename` with `PARTIAL_FILENAME_SUFFIX` appended to its stem (not
    its extension) when `partial` -- "log_b51a.TLM" -> "log_b51a_partial.TLM"
    -- so the file still opens in whatever tool its extension implies.
    """
    if not partial:
        return filename
    path = Path(filename)
    return f"{path.stem}{PARTIAL_FILENAME_SUFFIX}{path.suffix}"


def _verification_section(
    result: ReassemblyResult,
    header: BulkHeaderCandidate | None,
    candidates: list[BulkHeaderCandidate],
    partial_reason: str | None,
) -> None:
    """The cross-check against the headers: assembled size vs. the chosen
    header's `file_size`, assembled hash vs. every header candidate's
    `sha256` (a truncated one compared as a prefix), and a plain statement
    of whether what's here is a complete, verified file.
    """
    expected_size = header.file_size if header is not None else None

    if expected_size is not None:
        if expected_size == result.span_bytes:
            with ui.row().classes("items-center gap-2"):
                ui.icon("check_circle", color="positive")
                ui.label(
                    f"Assembled span matches the header's file_size "
                    f"({expected_size:,} bytes)."
                ).classes("text-positive")
        else:
            with ui.row().classes("items-center gap-2"):
                ui.icon("error", color="negative")
                ui.label(
                    f"Assembled {result.span_bytes:,} bytes, but the header's "
                    f"file_size is {expected_size:,} -- the download is cut "
                    "short (or these are two different files)."
                ).classes("text-negative")

    ui.label(f"SHA-256 of assembled bytes: {result.sha256}").classes(
        "font-mono text-sm"
    )
    hashes = {c.sha256 for c in candidates if c.sha256}
    prefix_matches = {
        h for h in hashes if len(h) < SHA256_HEX_LEN and result.sha256.startswith(h)
    }
    if result.sha256 in hashes:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("Matches a header candidate's SHA-256.").classes("text-positive")
    elif prefix_matches:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label(
                "Matches a header candidate's truncated SHA-256 (longest: "
                f"{max(map(len, prefix_matches))}/{SHA256_HEX_LEN} hex chars)."
            ).classes("text-positive")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label(
                "Does NOT match the SHA-256 of any of the "
                f"{len(hashes)} distinct header SHA-256(s) in the selected "
                "range(s)."
            ).classes("text-negative")

    if partial_reason is None:
        with ui.row().classes("items-center gap-2"):
            ui.icon("verified", color="positive")
            ui.label("Complete file.").classes("text-positive")


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


def _is_header_truncated(candidate: BulkHeaderCandidate) -> bool:
    """Whether this candidate's *header* was cut off -- nothing to do with
    whether the file itself came down whole (that's `_partial_reason`).

    Detected via the SHA-256, the only field truncation is currently
    visible in (see the module docstring: a long file path pushes
    TCMD_RESPONSE's 186-byte cap into the sha256 value before it's fully
    written).
    """
    return candidate.sha256 is not None and len(candidate.sha256) < SHA256_HEX_LEN


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
    is_header_truncated: bool
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
                    is_header_truncated=_is_header_truncated(c),
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


def _header_candidates_table(candidates: list[BulkHeaderCandidate]) -> None:
    """Render the found header candidates -- sorted by Received, consecutive
    look-alike entries collapsed into one row expandable (via a `+`/`-`
    button) to its individual timestamps.

    Which one the file's name/size/hash are then cross-checked against is
    `_best_named_candidate`'s call, made by the caller so the same choice
    drives the export's filename too.
    """
    if not candidates:
        ui.label(
            "No TCMD_RESPONSE header found in the selected range(s) -- that's "
            "fine, headers are sometimes missing entirely; just cross-check "
            "the file name/size/hash some other way."
        ).classes("text-caption text-grey")
        return

    candidates_by_time = sorted(candidates, key=lambda c: c.received_at)
    groups = _group_consecutive_candidates(candidates_by_time)

    columns = [
        {
            "name": "received_count",
            "label": "Times Received",
            "field": "received_count",
        },
        {
            "name": "first_received_at",
            "label": "First Received (UTC)",
            "field": "first_received_at",
        },
        {
            "name": "last_received_at",
            "label": "Last Received (UTC)",
            "field": "last_received_at",
        },
        {"name": "action", "label": "Action", "field": "action"},
        {"name": "file", "label": "File", "field": "file"},
        {"name": "file_size", "label": "Size (bytes)", "field": "file_size"},
        {"name": "crc16", "label": "CRC-16", "field": "crc16"},
        {"name": "sha256", "label": "SHA-256", "field": "sha256"},
        {
            "name": "is_header_truncated",
            "label": "Header Truncated?",
            "field": "is_header_truncated",
        },
    ]
    rows = [
        {
            "id": i,
            # `members` is in `received_at` order (see
            # `_group_consecutive_candidates`), so first/last are its ends --
            # equal for a group that was only received once.
            "received_count": len(group.members),
            "first_received_at": (f"{group.members[0].received_at:%Y-%m-%d %H:%M:%S}"),
            "last_received_at": f"{group.members[-1].received_at:%Y-%m-%d %H:%M:%S}",
            "action": group.action or "",
            "file": group.file or "",
            "file_size": f"{group.file_size:,}" if group.file_size is not None else "",
            "crc16": group.crc16 or "",
            "sha256": group.sha256 or "",
            "is_header_truncated": "yes" if group.is_header_truncated else "",
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


def _picam_image_section(
    jpg_bytes: bytes, filename_hint: str | None, *, partial: bool
) -> None:
    """A detected PiCAM image: rendered inline, plus a JPG download button --
    see `file_reassembly.detect_picam_image` for the detection heuristic.

    The JPG is marked partial on the same terms as the raw bytes it was
    decoded from: a PiCAM image whose middle never arrived still renders
    (that's the point of showing it mid-download), and its filename is the
    only thing that will still say so once it's saved to disk.
    """
    jpg_filename = _mark_partial(
        Path(filename_hint).with_suffix(".jpg").name
        if filename_hint
        else "picam_image.jpg",
        partial=partial,
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
    path: Path,
    ranges: list[tuple[datetime, datetime]],
    *,
    policy: ConflictPolicy,
    headers_only: bool = False,
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
        _header_candidates_table(candidates)

    if headers_only:
        return

    best = _best_named_candidate(candidates)
    chunks = beacon_data.load_bulk_file_downlink_packets(path, ranges=ranges)

    with ui.card().classes("w-full"):
        ui.label("BULK_FILE_DOWNLINK chunks").classes("text-lg font-bold")
        if chunks.is_empty():
            ui.label("No BULK_FILE_DOWNLINK packets in the selected range(s).").classes(
                "text-caption text-grey"
            )
            return

        result = reassemble_bulk_chunks(chunks, policy=policy)
        ui.label(
            f"{result.total_chunks:,} packet(s), {result.unique_offsets:,} "
            f"distinct offset(s), spanning {result.span_bytes:,} bytes."
        ).classes("text-caption text-grey")

        _assessment_summary(result)
        _coverage_map(result)
        _build_segments_table(result)()
        _duplicates_note(result)
        ui.separator()

        partial_reason = _partial_reason(result, best, candidates)
        _verification_section(result, best, candidates, partial_reason)

        # The full path (e.g. "ADCS/log_b51a.TLM"), not just the basename --
        # slashes become underscores since the browser would otherwise treat
        # them as directory separators in the downloaded filename.
        base_filename = (
            best.file.replace("/", "_")
            if best is not None and best.file
            else "reassembled_file.bin"
        )
        filename = _mark_partial(base_filename, partial=partial_reason is not None)
        ui.button(
            "Download reassembled bytes",
            icon="download",
            on_click=lambda: ui.download.content(result.data, filename=filename),
        )
        ui.label(f"Exports as: {filename}").classes("font-mono text-caption text-grey")
        if partial_reason is not None:
            ui.label(
                f'Marked "{PARTIAL_FILENAME_SUFFIX}" because {partial_reason}. The '
                "download still works (missing bytes are filled with 0x00, and "
                "conflicting ones resolved per the selected policy), but it "
                "isn't a verified, complete copy of the file."
            ).classes("text-caption text-grey")

        picam_jpg = detect_picam_image(result.data)
        if picam_jpg is not None:
            _picam_image_section(
                picam_jpg, base_filename, partial=partial_reason is not None
            )


def _conflict_policy_select(on_change: Callable[[str], None]) -> ui.select:
    """The conflict-resolution picker: which copy of a byte wins when the
    same byte arrived more than once with different values.
    """
    return ui.select(
        {policy.value: policy.label for policy in ConflictPolicy},
        value=DEFAULT_CONFLICT_POLICY.value,
        label="Conflict resolution",
        on_change=lambda e: on_change(e.value),
    ).classes("w-80")


def _make_policy_setter(
    search_state: dict[str, Any], results: Any
) -> Callable[[str], None]:
    """Switch conflict resolution, re-running the reassembly in place.

    Cheap enough to re-run on every change (the packets are already loaded
    and the assessment is linear in the file's size), and being able to flip
    between policies and watch the coverage map and SHA-256 settle -- or not
    -- is most of how an operator decides which one to trust for a given
    download.
    """

    def _set_policy(value: str) -> None:
        search_state["policy"] = ConflictPolicy(value)
        if search_state["has_searched"]:
            results.refresh()

    return _set_policy


def build_file_reassembler_page(data_dir: Path) -> None:
    parquet_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
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

    search_state: dict[str, Any] = {
        "headers_only": False,
        "has_searched": False,
        "policy": DEFAULT_CONFLICT_POLICY,
    }

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
            policy=search_state["policy"],
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
            _conflict_policy_select(
                on_change=_make_policy_setter(search_state, results)
            ).tooltip(
                "Which copy of a byte wins when the same byte came down more "
                "than once with different values. Conflicting bytes stay "
                "flagged either way -- this only decides what gets written "
                "into the exported file."
            )
        results()
