"""Reassembling a downlinked bulk file from BULK_FILE_DOWNLINK packet chunks.

A bulk file downlink is a run of `BULK_FILE_DOWNLINK` packets, each carrying
a byte offset into the file (`bulk_file_offset`) and up to 195 bytes of that
file's content (`bulk_data_hex`) -- see
`cts1_decode_satnogs_packets.decode_bulk_file_downlink_packet`. There's no
"file id" in the packet itself, so nothing in this module decides *which*
file a chunk belongs to -- that's the web UI's job, by letting the user pick
one or more `received_at` time ranges that isolate a single download. What
this module does is:

  - `reassemble_bulk_chunks()`: given whatever chunks the caller selected,
    assemble them into one byte string ordered by offset. The assessment of
    what came out is done **per byte**, not per packet: every byte in the
    file is `MISSING` (no packet ever carried it), `GOOD` (every packet
    that carried it agrees on its value), or `CONFLICTING` (two or more
    packets carried it with *different* values). Packets that overlap at
    unaligned offsets are therefore judged on the bytes they actually
    disagree about, rather than a whole repeated offset being written off
    as suspect. Those per-byte verdicts are then run-length encoded into
    `.segments` -- maximal consecutive runs of one status, always starting
    at byte 0 (a file whose first chunk never arrived opens with a
    `MISSING` segment rather than silently starting at the first byte that
    did) -- which is both what the UI renders and where `.gaps` /
    `.conflict_ranges` come from.

    A `CONFLICTING` byte still has to be *given* a value in the output, and
    which copy wins is the caller's call via `ConflictPolicy` (earliest /
    latest / most-common-then-earliest / most-common-then-latest): a
    conflict usually means two different transmissions' chunks were
    selected together, and which of those to trust is a judgement the
    operator makes from the download's history, not something this module
    can infer. Resolution never hides the conflict -- a resolved byte stays
    `CONFLICTING` in `.segments`, so `.is_complete` stays False and the
    export is still marked partial.

  - `find_header_candidates()`: a bulk downlink is nominally preceded by a
    `TCMD_RESPONSE` whose text is a JSON file descriptor (name, size,
    sha256, ...), but that header can just as easily arrive late, be
    interleaved with the data packets, or be missing outright. A descriptor
    is usually longer than one `TCMD_RESPONSE` frame's 186 bytes, so it's
    read from step 5's reassembled responses (every frame of one response
    already joined back together) rather than from raw packets. A response
    with a frame that never arrived has that frame filled with `?`s by step
    5, so parsing stays tolerant of broken JSON: a fully-parseable response
    gives up every property it has, and a broken one gives up whatever
    recognizable `"key": value` pairs survived around the gap. Either way,
    the result is surfaced to the user as candidates to cross-check
    against, not as ground truth.

  - `render_coverage_png()`: a byte-per-pixel bitmap of which bytes are
    good vs. missing vs. conflicting, one pixel per byte and displayed 1:1
    (so one byte is one screen pixel), which is what lets a whole multi-MB
    download's shape fit on screen at once. PNG's DEFLATE compression makes
    the (typically long) runs of one status cheap to ship over the
    websocket.
"""

from __future__ import annotations

__all__ = [
    "ADCS_SD_FILE_BLOB_ACTION",
    "CONFLICT_ISLAND_MAX_BYTES",
    "COVERAGE_PACKETS_PER_ROW",
    "COVERAGE_ROW_WIDTH_BYTES",
    "COVERAGE_RULER_FONT_SIZE_PX",
    "MAX_REASSEMBLY_SPAN_BYTES",
    "BulkHeaderCandidate",
    "ByteSegment",
    "ByteStatus",
    "ConflictPolicy",
    "DetectedImage",
    "OffsetSummary",
    "ReassemblyResult",
    "coverage_gutter_width_px",
    "coverage_label_interval_rows",
    "coverage_png_size",
    "coverage_ruler_height_px",
    "detect_image",
    "detect_picam_image",
    "find_header_candidates",
    "reassemble_bulk_chunks",
    "render_coverage_png",
]

import hashlib
import heapq
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from itertools import pairwise
from math import ceil
from typing import TYPE_CHECKING, Any, cast

from PIL import Image, ImageDraw, ImageFont

from cts1_mo_tools.cts1_decode_satnogs_packets import BULK_DOWNLINK_MAX_DATA
from cts1_mo_tools.cts1_picam_to_jpg import parse_picam_ascii_to_jpg_bytes

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl


# A corrupted/garbage `bulk_file_offset` (a raw uint32 straight off the
# wire) could claim to be near 4 GiB; refuse to allocate a buffer anywhere
# near that instead of hanging or exhausting memory on bad input.
MAX_REASSEMBLY_SPAN_BYTES = 256 * 1024 * 1024

SHA256_HEX_LEN = 64

# The longest `GOOD` run that `_absorb_conflict_islands` will re-label when
# it's stranded between two `CONFLICTING` runs. Sized to be well past what
# two different transmissions match on by coincidence (a shared zero byte, a
# repeated field value) but well short of a stretch two transmissions
# genuinely share.
CONFLICT_ISLAND_MAX_BYTES = 16


class ByteStatus(StrEnum):
    """The verdict on one byte (or, in a `ByteSegment`, one run of bytes)."""

    MISSING = "Missing"
    GOOD = "Good"
    CONFLICTING = "Conflicting"


class ConflictPolicy(StrEnum):
    """How to pick the winning value for a byte that arrived with two or
    more disagreeing values.

    Every policy is a total order over the copies of *one byte*, so a run
    of conflicting bytes is resolved byte by byte rather than by picking a
    single "winning packet" -- two packets can each be right about part of
    an overlap.

    `received_at` ties (two copies of the same byte received in the same
    timestamp tick) are broken by the chunk's position in the caller's
    DataFrame, so a given selection always resolves the same way.
    """

    EARLIEST = "earliest"
    LATEST = "latest"
    MOST_COMMON_THEN_EARLIEST = "most_common_then_earliest"
    MOST_COMMON_THEN_LATEST = "most_common_then_latest"

    @property
    def label(self) -> str:
        """A human-readable name for the UI's policy selector."""
        return _CONFLICT_POLICY_LABELS[self]


_CONFLICT_POLICY_LABELS = {
    ConflictPolicy.EARLIEST: "Earliest packet wins",
    ConflictPolicy.LATEST: "Latest packet wins",
    ConflictPolicy.MOST_COMMON_THEN_EARLIEST: ("Most common value wins, then earliest"),
    ConflictPolicy.MOST_COMMON_THEN_LATEST: "Most common value wins, then latest",
}

DEFAULT_CONFLICT_POLICY = ConflictPolicy.MOST_COMMON_THEN_LATEST


@dataclass(slots=True, frozen=True)
class ByteSegment:
    """One maximal run of consecutive bytes sharing a `ByteStatus`.

    `start` is inclusive, `end` exclusive. Segments tile `[0, span_bytes)`
    with no holes and no two neighbours sharing a status.

    `min_copies`/`max_copies` bound how many packets carried each byte in
    the run: `(1, 1)` for a byte that came down exactly once, `(0, 0)` for a
    `MISSING` run, and a wider range wherever the run spans bytes that were
    retransmitted different numbers of times. Since a run is merged on
    status alone, that spread is the honest summary -- "these bytes each
    arrived 1 to 3 times" -- rather than a single count that would be wrong
    for part of the range.
    """

    start: int
    end: int
    status: ByteStatus
    min_copies: int = 0
    max_copies: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True, frozen=True)
class OffsetSummary:
    """A summary of every copy seen at one `bulk_file_offset` in the
    selection -- `count` is 1 for an offset that showed up exactly once.

    This is packet bookkeeping (how often each offset was retransmitted),
    deliberately *not* the correctness verdict: whether the file is right
    is decided per byte, in `ReassemblyResult.segments`.
    """

    offset: int
    count: int
    distinct_contents_count: int  # how many distinct byte-content variants
    lengths: tuple[int, ...]  # distinct `bulk_data_len` values seen, ascending

    @property
    def consistent(self) -> bool:
        """Whether every copy at this offset carries identical bytes."""
        return self.distinct_contents_count == 1


@dataclass(slots=True, frozen=True)
class ReassemblyResult:
    """The result of assembling a set of `BULK_FILE_DOWNLINK` chunks.

    `data` always has length `span_bytes` (the highest `offset + length`
    seen across the selection); a `MISSING` byte in it is `\\x00`, and a
    `CONFLICTING` one holds whichever copy `policy` picked.

    `.segments` is the per-byte assessment, run-length encoded and covering
    `[0, span_bytes)` end to end -- so `.gaps` (the `MISSING` runs) and
    `.conflict_ranges` (the `CONFLICTING` ones) are just views onto it, and
    both are already merged/sorted by construction.

    `.offsets` summarizes every distinct offset seen -- present even when
    there's nothing wrong to report, so the caller always has something to
    show. Seeing the *same* offset more than once is routine (e.g. a
    retransmission, or two ground stations catching the same overpass) and
    isn't itself a problem; `.duplicates` is that subset. Only
    `.has_conflicts` -- bytes that were received with disagreeing values --
    means two different transmissions' chunks likely got selected together,
    and the caller should either narrow the time-range filter or accept a
    `ConflictPolicy`'s answer knowingly.
    """

    total_chunks: int
    unique_offsets: int
    offsets: tuple[OffsetSummary, ...]
    segments: tuple[ByteSegment, ...]
    data: bytes
    span_bytes: int
    policy: ConflictPolicy

    def _bytes_with_status(self, status: ByteStatus) -> int:
        return sum(s.length for s in self.segments if s.status is status)

    def ranges_with_status(self, status: ByteStatus) -> tuple[tuple[int, int], ...]:
        """The `(start, end)` byte ranges of every segment with `status`."""
        return tuple((s.start, s.end) for s in self.segments if s.status is status)

    @property
    def gaps(self) -> tuple[tuple[int, int], ...]:
        """Byte ranges never received at all."""
        return self.ranges_with_status(ByteStatus.MISSING)

    @property
    def conflict_ranges(self) -> tuple[tuple[int, int], ...]:
        """Byte ranges received with two or more disagreeing values."""
        return self.ranges_with_status(ByteStatus.CONFLICTING)

    @property
    def missing_bytes(self) -> int:
        return self._bytes_with_status(ByteStatus.MISSING)

    @property
    def good_bytes(self) -> int:
        return self._bytes_with_status(ByteStatus.GOOD)

    @property
    def conflict_bytes(self) -> int:
        return self._bytes_with_status(ByteStatus.CONFLICTING)

    @property
    def covered_bytes(self) -> int:
        """Bytes that arrived at all -- good or conflicting."""
        return self.good_bytes + self.conflict_bytes

    @property
    def duplicates(self) -> tuple[OffsetSummary, ...]:
        """Offsets that showed up more than once in the selection."""
        return tuple(o for o in self.offsets if o.count > 1)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflict_bytes)

    @property
    def is_gapless(self) -> bool:
        return not self.missing_bytes

    @property
    def is_complete(self) -> bool:
        """Whether every byte in `[0, span_bytes)` arrived and agreed.

        Note this can only speak for the span the *packets* implied: a
        download cut off before its last chunk looks complete here, which
        is why the UI also cross-checks `span_bytes` against the header's
        `file_size`.
        """
        return self.span_bytes > 0 and self.is_gapless and not self.has_conflicts

    @property
    def sha256(self) -> str:
        """SHA-256 of `data` as assembled -- only meaningful to compare
        against a known-good hash once `is_complete` (a gap zero-fills its
        bytes, and a conflict may have resolved to the "wrong" copy, either
        of which changes the hash).
        """
        return hashlib.sha256(self.data).hexdigest()


@dataclass(slots=True, frozen=True)
class _Chunk:
    """One decoded `BULK_FILE_DOWNLINK` packet's payload, ready to assess."""

    offset: int
    data: bytes
    received_at: datetime
    order: int  # position in the caller's DataFrame; the final tie-break

    @property
    def end(self) -> int:
        return self.offset + len(self.data)


# Sorted by "which copy of a byte wins", so `min` is the earliest copy and
# `max` the latest, with the DataFrame position breaking `received_at` ties.
def _recency_key(chunk: _Chunk) -> tuple[datetime, int]:
    return (chunk.received_at, chunk.order)


def _resolve_conflicting_byte(
    candidates: list[tuple[int, _Chunk]], policy: ConflictPolicy
) -> int:
    """Pick the winning value for one byte, from `(value, chunk)` pairs that
    don't all agree, per `policy`.
    """
    if policy is ConflictPolicy.EARLIEST:
        return min(candidates, key=lambda pair: _recency_key(pair[1]))[0]
    if policy is ConflictPolicy.LATEST:
        return max(candidates, key=lambda pair: _recency_key(pair[1]))[0]

    # "Most common": tally the copies of each distinct value, then break a
    # tied tally by the same earliest/latest rule -- comparing each value's
    # own earliest (resp. latest) copy, not the packets as a whole.
    keys_by_value: dict[int, list[tuple[datetime, int]]] = defaultdict(list)
    for value, chunk in candidates:
        keys_by_value[value].append(_recency_key(chunk))

    if policy is ConflictPolicy.MOST_COMMON_THEN_EARLIEST:
        return min(
            keys_by_value.items(), key=lambda item: (-len(item[1]), min(item[1]))
        )[0]
    return max(keys_by_value.items(), key=lambda item: (len(item[1]), max(item[1])))[0]


def _merge_segment(segments: list[ByteSegment], segment: ByteSegment) -> None:
    """Append `segment` to `segments`, extending the previous run instead if
    it has the same status and ends where this one starts.

    That merge is what keeps `.segments` a list of *maximal* runs however
    finely the assessment happened to decide them (a conflicting overlap is
    settled one byte at a time, and would otherwise arrive as hundreds of
    one-byte segments). Merging widens the run's `min_copies`/`max_copies`
    rather than overwriting them, so a run that spans a retransmission
    boundary reports the spread across it.
    """
    if segment.length <= 0:
        return
    previous = segments[-1] if segments else None
    if (
        previous is not None
        and previous.status is segment.status
        and previous.end == segment.start
    ):
        segments[-1] = ByteSegment(
            previous.start,
            segment.end,
            segment.status,
            min(previous.min_copies, segment.min_copies),
            max(previous.max_copies, segment.max_copies),
        )
    else:
        segments.append(segment)


def _append_segment(
    segments: list[ByteSegment],
    start: int,
    end: int,
    status: ByteStatus,
    copies: int,
) -> None:
    """`_merge_segment` for a run whose bytes were each carried by exactly
    `copies` packets.
    """
    _merge_segment(segments, ByteSegment(start, end, status, copies, copies))


def _absorb_conflict_islands(
    segments: list[ByteSegment],
) -> tuple[ByteSegment, ...]:
    """Re-label short `GOOD` runs stranded between two `CONFLICTING` runs as
    conflicting, and re-merge what that joins up.

    Where two transmissions of different data overlap, a few bytes will
    match by coincidence -- a shared 0x00, a run of spaces, the same field
    value -- and strictly per-byte labelling turns a solidly conflicted
    region into a stripe of one-byte "good" islands. Those islands aren't
    information: the bytes agree by chance, not because anything
    corroborates them, and the region as a whole still needs either a
    narrower time range or a conflict policy applied to it. Anything longer
    than `CONFLICT_ISLAND_MAX_BYTES` is left alone, since a genuinely
    agreeing stretch that long (e.g. the part of an overlap both
    transmissions really do share) is worth seeing as its own run.

    Only runs with conflict on *both* sides are absorbed -- a good run
    bordering missing bytes or the file's edge is at the boundary of the
    contested region, not stranded inside it. The bytes themselves are
    untouched: an absorbed run's copies agreed, so it already holds the
    value any `ConflictPolicy` would have picked.
    """
    absorbed: list[ByteSegment] = []
    for index, segment in enumerate(segments):
        is_island = (
            segment.status is ByteStatus.GOOD
            and segment.length <= CONFLICT_ISLAND_MAX_BYTES
            and 0 < index < len(segments) - 1
            and segments[index - 1].status is ByteStatus.CONFLICTING
            and segments[index + 1].status is ByteStatus.CONFLICTING
        )
        _merge_segment(
            absorbed,
            replace(segment, status=ByteStatus.CONFLICTING) if is_island else segment,
        )
    return tuple(absorbed)


def _assess_bytes(
    chunks: list[_Chunk], span: int, policy: ConflictPolicy
) -> tuple[bytes, tuple[ByteSegment, ...]]:
    """Assemble `chunks` into `span` bytes and assess every one of them.

    Walks the chunks' start/end boundaries (plus 0 and `span`, so the
    result always tiles the file from byte 0 even when nothing covers the
    start or the middle), which gives sub-ranges over which the set of
    covering chunks is constant. A sub-range covered by nothing is a
    `MISSING` run and one covered by a single chunk is a `GOOD` run, both
    settled in one step; only where chunks actually overlap does this drop
    to comparing individual bytes -- and even then, only after a whole-slice
    equality check has ruled out the common case of a plain retransmission
    that agrees.
    """
    if span == 0:
        return b"", ()

    buf = bytearray(span)
    segments: list[ByteSegment] = []

    by_start = sorted(chunks, key=lambda c: c.offset)
    boundaries = sorted(
        {0, span, *(c.offset for c in chunks), *(c.end for c in chunks)}
    )

    next_chunk = 0
    # Active chunks as a min-heap on `end`, so expiring them is O(log n)
    # rather than rescanning the active set at every boundary.
    active: list[tuple[int, int, _Chunk]] = []
    for start, end in pairwise(boundaries):
        while next_chunk < len(by_start) and by_start[next_chunk].offset <= start:
            chunk = by_start[next_chunk]
            heapq.heappush(active, (chunk.end, next_chunk, chunk))
            next_chunk += 1
        while active and active[0][0] <= start:
            heapq.heappop(active)

        covering = [chunk for _end, _i, chunk in active]
        if not covering:
            _append_segment(segments, start, end, ByteStatus.MISSING, 0)
            continue

        # Constant across the whole sub-range: every chunk in `covering`
        # spans it end to end, by construction of the boundary walk.
        copies = len(covering)

        slices = [c.data[start - c.offset : end - c.offset] for c in covering]
        if all(s == slices[0] for s in slices[1:]):
            # Nothing disagrees here (the overwhelmingly common case,
            # including a clean retransmission) -- write it in one go.
            buf[start:end] = slices[0]
            _append_segment(segments, start, end, ByteStatus.GOOD, copies)
            continue

        for pos in range(start, end):
            candidates = [(c.data[pos - c.offset], c) for c in covering]
            values = {value for value, _chunk in candidates}
            if len(values) == 1:
                buf[pos] = values.pop()
                status = ByteStatus.GOOD
            else:
                buf[pos] = _resolve_conflicting_byte(candidates, policy)
                status = ByteStatus.CONFLICTING
            _append_segment(segments, pos, pos + 1, status, copies)

    return bytes(buf), _absorb_conflict_islands(segments)


def _decode_chunks(df: pl.DataFrame) -> list[_Chunk]:
    """Decode `df`'s hex payloads into `_Chunk`s, in DataFrame order.

    The decoded bytes -- not the packet's own `bulk_data_len` -- are what
    coverage is judged on: a payload whose hex came up short can only
    account for the bytes it actually carries. `bulk_data_len` is still
    reported as-is in `OffsetSummary.lengths`, where a disagreement between
    the two is exactly the kind of thing an operator wants to see rather
    than have silently reconciled.
    """
    required = {"bulk_file_offset", "bulk_data_len", "bulk_data_hex", "received_at"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Chunk DataFrame is missing required column(s): {sorted(missing)}"
        raise ValueError(msg)

    return [
        _Chunk(
            offset=offset,
            data=bytes.fromhex(hex_str),
            received_at=received_at,
            order=order,
        )
        for order, (offset, hex_str, received_at) in enumerate(
            zip(
                df["bulk_file_offset"].to_list(),
                df["bulk_data_hex"].to_list(),
                df["received_at"].to_list(),
                strict=True,
            )
        )
    ]


def _summarize_offsets(df: pl.DataFrame) -> tuple[OffsetSummary, ...]:
    """Per-offset packet bookkeeping: how many copies of each offset arrived,
    how many distinct payloads they carried, and what lengths they claimed.
    """
    copies_by_offset: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for offset, length, hex_str in zip(
        df["bulk_file_offset"].to_list(),
        df["bulk_data_len"].to_list(),
        df["bulk_data_hex"].to_list(),
        strict=True,
    ):
        copies_by_offset[offset].append((length, hex_str))

    return tuple(
        OffsetSummary(
            offset,
            count=len(copies),
            distinct_contents_count=len({hex_str for _length, hex_str in copies}),
            lengths=tuple(sorted({length for length, _hex_str in copies})),
        )
        for offset, copies in sorted(copies_by_offset.items())
    )


def reassemble_bulk_chunks(
    df: pl.DataFrame, *, policy: ConflictPolicy = DEFAULT_CONFLICT_POLICY
) -> ReassemblyResult:
    """Assemble whatever `BULK_FILE_DOWNLINK` chunks are in `df` into one
    byte string, ordered by `bulk_file_offset`, and assess every byte of it.

    `df` must have `bulk_file_offset`, `bulk_data_len`, `bulk_data_hex`, and
    `received_at` columns -- already filtered by the caller (e.g. to one or
    more `received_at` time ranges isolating a single download) to whatever
    rows should be treated as chunks of the same file. This never raises on
    messy input beyond the size cap below: gaps and disagreements are
    reported on the result (see `ReassemblyResult.segments`) rather than
    treated as errors, so the caller's filter is what makes the result
    trustworthy, not this function.

    The result doesn't depend on `df`'s row order: which copy of a
    disagreed-upon byte wins is decided by `policy` from the chunks'
    `received_at`, with row order used only to break exact timestamp ties.

    Raises:
        ValueError: if `df` is missing a required column, or if the implied
            file is larger than `MAX_REASSEMBLY_SPAN_BYTES` -- almost
            certainly a corrupt `bulk_file_offset` rather than a real file
            this large.
    """
    if df.is_empty():
        return ReassemblyResult(0, 0, (), (), b"", 0, policy)

    chunks = _decode_chunks(df)
    span = max((c.end for c in chunks), default=0)
    if span > MAX_REASSEMBLY_SPAN_BYTES:
        msg = (
            f"Implied file size {span:,} bytes exceeds the "
            f"{MAX_REASSEMBLY_SPAN_BYTES:,}-byte safety cap -- check for a "
            f"corrupt bulk_file_offset in the selected range."
        )
        raise ValueError(msg)

    data, segments = _assess_bytes(chunks, span, policy)

    return ReassemblyResult(
        total_chunks=len(chunks),
        unique_offsets=len({c.offset for c in chunks}),
        offsets=_summarize_offsets(df),
        segments=segments,
        data=data,
        span_bytes=span,
        policy=policy,
    )


# -- Coverage map --------------------------------------------------------------

# One row is a whole number of full packets, so packet-aligned damage (a
# whole chunk lost) reads as a clean block rather than a diagonal smear
# across rows -- and the map's top ruler can tick once per packet boundary.
COVERAGE_PACKETS_PER_ROW = 4
COVERAGE_ROW_WIDTH_BYTES = BULK_DOWNLINK_MAX_DATA * COVERAGE_PACKETS_PER_ROW

# The map's two rulers, drawn into the image itself (see
# `render_coverage_png`): a top strip ticked once per packet, and a left
# margin ticked every `COVERAGE_OFFSET_LABEL_INTERVAL_ROWS` rows with that
# row's byte offset. Both are part of the PNG rather than HTML overlaid on
# it, so a tick can't drift from the pixel it labels -- at one byte per
# pixel there's no rounding to hide a half-pixel error behind.
COVERAGE_OFFSET_LABEL_INTERVAL_ROWS = 200
COVERAGE_MAX_OFFSET_LABELS = 500

# Twice Pillow's own default size: the map is drawn at one byte per pixel,
# so its labels are the one part with no reason to be that small -- but they
# still have to sit inside a margin that's pure overhead next to the data.
COVERAGE_RULER_FONT_SIZE_PX = 22

_COVERAGE_TICK_LEN_PX = 8
_COVERAGE_TICK_GAP_PX = 5  # between a tick and its label


@cache
def _ruler_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The rulers' font, cached -- every label in a map is drawn with it, and
    loading it per call would be the most expensive part of rendering one.

    Pillow's own default face at `COVERAGE_RULER_FONT_SIZE_PX`, so there's no
    font file to ship; it falls back to a bitmap font on a Pillow built
    without FreeType, which is why the return type admits either.
    """
    return ImageFont.load_default(size=COVERAGE_RULER_FONT_SIZE_PX)


def _label_size(label: str) -> tuple[int, int]:
    """`(width, height)` in pixels of `label` drawn in the rulers' font, as
    measured from the origin the `draw.text()` calls below use.
    """
    _left, _top, right, bottom = _ruler_font().getbbox(label)
    return (ceil(right), ceil(bottom))


def coverage_ruler_height_px() -> int:
    """Height of the map's top ruler: one label, the gap below it, and the
    tick that touches the data area.
    """
    _width, height = _label_size("0,")
    return height + _COVERAGE_TICK_GAP_PX + _COVERAGE_TICK_LEN_PX


def coverage_gutter_width_px(row_count: int, row_width_bytes: int) -> int:
    """Width of the map's left gutter, sized to the widest offset it will
    have to print for a map this tall.

    Measured rather than fixed: the labels run from "0" for a one-row file
    to eight digits and three separators for one near
    `MAX_REASSEMBLY_SPAN_BYTES`, and a constant wide enough for the latter
    would waste most of its width on every real file.
    """
    widest = f"{max(row_count - 1, 0) * row_width_bytes:,}"
    width, _height = _label_size(widest)
    return width + _COVERAGE_TICK_GAP_PX + _COVERAGE_TICK_LEN_PX


# Palette indices/colors for the coverage bitmap -- index 3 ("no data yet",
# past the end of the file but needed to fill out the last row's width, plus
# the rulers' background) is marked fully transparent so it doesn't paint a
# visible block and the page's own background shows through.
_COVERAGE_PAD_INDEX = 3
_COVERAGE_INK_INDEX = 4
_COVERAGE_STATUS_INDEX = {
    ByteStatus.MISSING: 0,
    ByteStatus.GOOD: 1,
    ByteStatus.CONFLICTING: 2,
}
_COVERAGE_PALETTE = (
    (0xC1, 0x00, 0x15),  # missing: Quasar "negative" red
    (0x21, 0xBA, 0x45),  # good: Quasar "positive" green
    (0xF2, 0xC0, 0x37),  # conflicting: Quasar "warning" yellow
    (0x00, 0x00, 0x00),  # padding: color irrelevant, made transparent below
    # Ruler ink: a dark grey, reading as labelling rather than data against
    # the page's own background, which shows through the transparent margins.
    (0x61, 0x61, 0x61),
)


def coverage_label_interval_rows(row_count: int) -> int:
    """How many rows apart the offset labels in the map's left gutter go.

    `COVERAGE_OFFSET_LABEL_INTERVAL_ROWS` normally, widened in multiples of
    it once a file is tall enough that a label every 200 rows would mean
    thousands of them -- past `COVERAGE_MAX_OFFSET_LABELS` they stop being a
    scale and start being a wall of text for no added precision.
    """
    max_rows = COVERAGE_OFFSET_LABEL_INTERVAL_ROWS * COVERAGE_MAX_OFFSET_LABELS
    if row_count <= max_rows:
        return COVERAGE_OFFSET_LABEL_INTERVAL_ROWS
    return COVERAGE_OFFSET_LABEL_INTERVAL_ROWS * -(-row_count // max_rows)


def coverage_png_size(
    result: ReassemblyResult, *, row_width_bytes: int = COVERAGE_ROW_WIDTH_BYTES
) -> tuple[int, int]:
    """The `(width, height)` in pixels `render_coverage_png` will produce --
    the rulers included -- so the caller can size the element that displays
    it without decoding the PNG or duplicating the arithmetic.
    """
    if result.span_bytes == 0:
        return (1, 1)
    row_count = -(-result.span_bytes // row_width_bytes)
    return (
        coverage_gutter_width_px(row_count, row_width_bytes) + row_width_bytes,
        coverage_ruler_height_px() + row_count,
    )


def _encode_coverage_png(image: Image.Image) -> bytes:
    """PNG-encode a palette image of `_COVERAGE_PALETTE` indices.

    Palette ("P") mode rather than RGB keeps it at one byte per byte-of-file
    before DEFLATE even runs, and `_COVERAGE_PAD_INDEX` is declared
    transparent so neither the last row's padding nor the rulers' background
    paints over the page.
    """
    image.putpalette(b"".join(bytes(color) for color in _COVERAGE_PALETTE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=_COVERAGE_PAD_INDEX)
    return buffer.getvalue()


def _draw_coverage_rulers(
    image: Image.Image, row_count: int, row_width_bytes: int
) -> None:
    """Draw the top (per-packet) and left (per-row byte offset) rulers into
    `image`, whose data area starts at
    `(coverage_gutter_width_px(...), coverage_ruler_height_px())`.

    Top ticks land on each packet boundary within a row and are labelled
    with the byte offset *within* the row; left ticks land on a row and are
    labelled with that row's absolute byte offset. Read together, a block's
    absolute offset is its row's label plus its column's.
    """
    draw = ImageDraw.Draw(image)
    font = _ruler_font()
    data_left = coverage_gutter_width_px(row_count, row_width_bytes)
    data_top = coverage_ruler_height_px()

    for offset in range(0, row_width_bytes, BULK_DOWNLINK_MAX_DATA):
        x = data_left + offset
        draw.line(
            [(x, data_top - _COVERAGE_TICK_LEN_PX), (x, data_top - 1)],
            fill=_COVERAGE_INK_INDEX,
        )
        label = f"{offset:,}"
        _width, height = _label_size(label)
        draw.text(
            (
                x + _COVERAGE_TICK_GAP_PX,
                data_top - _COVERAGE_TICK_LEN_PX - _COVERAGE_TICK_GAP_PX - height,
            ),
            label,
            fill=_COVERAGE_INK_INDEX,
            font=font,
        )

    for row in range(0, row_count, coverage_label_interval_rows(row_count)):
        y = data_top + row
        draw.line(
            [(data_left - _COVERAGE_TICK_LEN_PX, y), (data_left - 1, y)],
            fill=_COVERAGE_INK_INDEX,
        )
        label = f"{row * row_width_bytes:,}"
        width, height = _label_size(label)
        # Right-aligned against the tick, and centred on the row it marks,
        # except near the edges where it's nudged back inside the image --
        # the tick is the precise mark, so a label by the last row reads
        # fine slightly above it and would otherwise be cut in half.
        draw.text(
            (
                data_left - _COVERAGE_TICK_LEN_PX - _COVERAGE_TICK_GAP_PX - width,
                min(max(y - height // 2, 0), image.height - height),
            ),
            label,
            fill=_COVERAGE_INK_INDEX,
            font=font,
        )


def render_coverage_png(
    result: ReassemblyResult, *, row_width_bytes: int = COVERAGE_ROW_WIDTH_BYTES
) -> bytes:
    """A one-pixel-per-byte PNG of `result`'s per-byte assessment: green
    where the byte is good, red where it's missing, yellow where it arrived
    with multiple disagreeing values, `row_width_bytes` pixels per row,
    inset by the two rulers `_draw_coverage_rulers` draws around it. The
    caller (the web UI) displays it at its natural size, one byte per screen
    pixel, so the image's own dimensions are the on-screen ones.

    Returns a 1x1 (single good-color pixel) PNG if `result` is empty, rather
    than a 0-byte image some browsers may refuse to render.
    """
    span = result.span_bytes
    if span == 0:
        return _encode_coverage_png(
            Image.new("P", (1, 1), _COVERAGE_STATUS_INDEX[ByteStatus.GOOD])
        )

    pixels = bytearray(span)
    for segment in result.segments:
        index = _COVERAGE_STATUS_INDEX[segment.status]
        pixels[segment.start : segment.end] = bytes([index]) * segment.length

    remainder = span % row_width_bytes
    if remainder:
        pixels += bytes([_COVERAGE_PAD_INDEX]) * (row_width_bytes - remainder)
    row_count = len(pixels) // row_width_bytes

    width, height = coverage_png_size(result, row_width_bytes=row_width_bytes)
    image = Image.new("P", (width, height), _COVERAGE_PAD_INDEX)
    # Same mode both sides, so this copies palette indices straight across
    # rather than converting anything.
    image.paste(
        Image.frombytes("P", (row_width_bytes, row_count), bytes(pixels)),
        (
            coverage_gutter_width_px(row_count, row_width_bytes),
            coverage_ruler_height_px(),
        ),
    )
    _draw_coverage_rulers(image, row_count, row_width_bytes)

    return _encode_coverage_png(image)


# -- Header candidates --------------------------------------------------------

# Deliberately not *only* a JSON parser: a response with a frame that never
# arrived has that frame's bytes replaced by `?`s in step 5's output (see
# `step_5_reassemble_tcmd_responses`), which can land anywhere -- mid-value,
# mid-key, or over the closing brace. So each `"key": value` pair is pulled
# out independently, and a broken one just comes back missing rather than
# failing the whole parse. Only string/number/bool/null values are matched,
# which is all a file descriptor carries.
_JSON_PAIR_RE = re.compile(
    r'"([A-Za-z0-9_]+)"\s*:\s*'
    r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
)
# A `sha256` cut off before its closing quote (by a missing frame) is still
# worth a prefix match, so it gets its own unterminated-string pattern.
_SHA256_PREFIX_RE = re.compile(r'"sha256"\s*:\s*"([0-9a-fA-F]+)')

# The response to this action carries a `datetime` property: the ADCS's own
# last-modified timestamp for the file (UTC), which the older
# `adcs_get_latest_sd_file_blob` response didn't include.
ADCS_SD_FILE_BLOB_ACTION = "adcs_get_latest_sd_file_blob"


def _parse_properties(text: str) -> tuple[dict[str, Any], bool]:
    """Every property in one `TCMD_RESPONSE` body, and whether the body
    parsed as a whole JSON object (as opposed to being salvaged pair by pair
    from broken JSON).
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        pass
    else:
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed), True

    properties: dict[str, Any] = {}
    for m in _JSON_PAIR_RE.finditer(text):
        key, raw_value = m.groups()
        try:
            properties.setdefault(key, json.loads(raw_value))
        except ValueError:
            continue
    if "sha256" not in properties and (m := _SHA256_PREFIX_RE.search(text)):
        properties["sha256"] = m.group(1)
    return properties, False


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    # `bool` is an `int` subclass, but `"file_size": true` is not a size.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(slots=True, frozen=True)
class BulkHeaderCandidate:
    """One reassembled `TCMD_RESPONSE` that looks like a bulk-download file
    descriptor, with its well-known fields pulled out.

    Every well-known field may be `None` -- either the property wasn't
    present, or it was lost to a missing frame (`sha256` may also come back
    as a < 64-hex-char prefix of the real hash in that case). `properties`
    holds *every* property that could be parsed, well-known or not, and
    `raw_text` the response exactly as step 5 reassembled it, so nothing the
    satellite sent is hidden behind the fields this module happens to know
    about.
    """

    received_at: datetime  # when the response's first frame arrived
    tcmd_sent_at: datetime | None  # when the telecommand was sent, if known
    action: str | None
    file: str | None
    file_size: int | None
    sha256: str | None
    crc16: str | None
    offset: int | None
    length: int | None
    # The `datetime` property of an `SD_FILE_BLOB_V2_ACTION` response -- the
    # file's last-modified time according to the ADCS, in UTC, as sent.
    adcs_modified_timestamp_utc: str | None
    # Whether every frame of the response arrived and it parsed as JSON.
    is_complete: bool
    raw_text: str
    properties: tuple[tuple[str, Any], ...]


def _looks_like_file_descriptor(properties: dict[str, Any]) -> bool:
    """Whether a response is plausibly a bulk-download descriptor at all,
    rather than some other command's response.
    """
    return "file" in properties or "file_size" in properties


def find_header_candidates(
    tcmd_responses: pl.DataFrame,
) -> list[BulkHeaderCandidate]:
    """Every reassembled `TCMD_RESPONSE` in `tcmd_responses` that looks like
    a bulk-download file descriptor, oldest first.

    `tcmd_responses` is (a slice of) step 5's
    `reassembled_tcmd_responses.parquet`: it must have `first_received_at`,
    `tcmd_sent_at`, `is_complete`, and `tcmd_response_text` columns.
    """
    if tcmd_responses.is_empty():
        return []

    candidates: list[BulkHeaderCandidate] = []
    for text, received_at, tcmd_sent_at, response_complete in zip(
        tcmd_responses["tcmd_response_text"].to_list(),
        tcmd_responses["first_received_at"].to_list(),
        tcmd_responses["tcmd_sent_at"].to_list(),
        tcmd_responses["is_complete"].to_list(),
        strict=True,
    ):
        if not text:
            continue
        properties, parsed_whole = _parse_properties(text)
        if not _looks_like_file_descriptor(properties):
            continue

        action = _str_or_none(properties.get("action"))
        sha256 = _str_or_none(properties.get("sha256"))
        candidates.append(
            BulkHeaderCandidate(
                received_at=received_at,
                tcmd_sent_at=tcmd_sent_at,
                action=action,
                file=_str_or_none(properties.get("file")),
                file_size=_int_or_none(properties.get("file_size")),
                sha256=sha256.lower() if sha256 else None,
                crc16=_str_or_none(properties.get("crc16")),
                offset=_int_or_none(properties.get("offset")),
                length=_int_or_none(properties.get("length")),
                adcs_modified_timestamp_utc=(
                    _str_or_none(properties.get("datetime"))
                    if (action is not None)
                    and action.startswith(ADCS_SD_FILE_BLOB_ACTION)
                    else None
                ),
                is_complete=bool(response_complete) and parsed_whole,
                raw_text=text,
                properties=tuple(properties.items()),
            )
        )
    candidates.sort(key=lambda c: c.received_at)
    return candidates


# -- PiCAM image detection -----------------------------------------------------


def detect_picam_image(data: bytes) -> bytes | None:
    """Best-effort detection of a PiCAM ASCII-format image inside reassembled
    bulk-download `data`, decoded to JPG bytes if so.

    Heuristic: `data` decodes as ASCII text starting with the `START_CAM:`
    sentinel line, with at least one `@FACE...` end-of-telemetry line -- see
    `cts1_picam_to_jpg.parse_picam_ascii_to_jpg_bytes` for the actual
    per-line decode. Returns None if either condition fails (not a PiCAM
    image, or a download that's still missing its start/end -- e.g. gaps
    still need filling in).
    """
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return None

    if not text.startswith("START_CAM:"):
        return None
    if not any(line.startswith("@FACE") for line in text.splitlines()):
        return None

    return parse_picam_ascii_to_jpg_bytes(text, enable_logs=False)


# -- Previewable image detection -----------------------------------------------

# Sniffed from the reassembled bytes rather than trusted from the header's
# filename: the filename is a best-effort guess in the first place (see
# `find_header_candidates`), and a file named ".jpg" that isn't one would
# render as a broken image with no explanation. Magic bytes are also what
# make a preview possible for a download whose header never arrived at all.
_IMAGE_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    # SOI + the first marker of a JFIF/EXIF/raw-APPn JPEG.
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"BM", "image/bmp", ".bmp"),
)


@dataclass(frozen=True, slots=True)
class DetectedImage:
    """An image found in reassembled bytes, ready to render in the browser.

    `data` is what gets served -- the reassembled bytes themselves for a
    downlinked .jpg/.bmp, or the decoded JPG for a PiCAM ASCII image, which
    `is_converted` distinguishes (a converted image needs its own download
    button; raw bytes are already what the main download hands over).
    """

    data: bytes
    media_type: str
    suffix: str
    label: str
    is_converted: bool


def detect_image(data: bytes) -> DetectedImage | None:
    """Best-effort detection of something previewable in reassembled bulk
    download `data`: a PiCAM ASCII image (decoded to JPG), or a JPEG/BMP
    file downlinked as-is.

    Returns None for anything else -- including an image whose first bytes
    haven't arrived yet, since the magic number is what identifies it. A
    partial image *past* its header is still returned: browsers render what
    they can of a truncated JPEG/BMP, and watching one fill in mid-download
    is most of why this preview exists.
    """
    picam_jpg = detect_picam_image(data)
    if picam_jpg is not None:
        return DetectedImage(
            data=picam_jpg,
            media_type="image/jpeg",
            suffix=".jpg",
            label="Detected PiCAM image",
            is_converted=True,
        )

    for magic, media_type, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return DetectedImage(
                data=data,
                media_type=media_type,
                suffix=suffix,
                label=f"Detected {suffix.removeprefix('.').upper()} image",
                is_converted=False,
            )

    return None
