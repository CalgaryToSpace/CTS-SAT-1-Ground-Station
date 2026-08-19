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
    assemble them into one byte string ordered by offset (missing byte
    ranges are left as `\\x00` and reported separately as `.gaps`), and flag
    any offset that shows up more than once (`.duplicates`) -- the caller's
    time-range filter should be narrowed until there are none, since a
    duplicated offset means either a harmless retransmission or, worse, two
    different files' chunks got mixed into the same selection.

  - `find_header_candidates()`: a bulk downlink is nominally preceded by a
    `TCMD_RESPONSE` whose text is a JSON file descriptor (name, size,
    sha256), but that header can just as easily arrive late, be interleaved
    with the data packets, or be missing outright -- and even when present,
    `TCMD_RESPONSE`'s payload is a hard-capped 186 bytes, so a descriptor
    naming a long file path routinely runs out of room before its `sha256`
    is fully written. This is a best-effort scan for whatever recognizable
    fields (`file`, `file_size`, `sha256`, ...) show up in that text,
    tolerant of the JSON being incomplete/truncated -- surfaced to the user
    as candidates to cross-check against, not as ground truth.
"""

from __future__ import annotations

__all__ = [
    "BulkHeaderCandidate",
    "DuplicateOffset",
    "ReassemblyResult",
    "find_header_candidates",
    "reassemble_bulk_chunks",
]

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from datetime import datetime

# A corrupted/garbage `bulk_file_offset` (a raw uint32 straight off the
# wire) could claim to be near 4 GiB; refuse to allocate a buffer anywhere
# near that instead of hanging or exhausting memory on bad input.
MAX_REASSEMBLY_SPAN_BYTES = 256 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class DuplicateOffset:
    """A `bulk_file_offset` that showed up more than once in the selection."""

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
    seen across the selection) -- any never-covered byte in it is `\\x00`,
    and its exact position is also listed in `.gaps`, so a fully accurate
    file requires both `is_gapless` and no `.conflicts`.

    Seeing the *same* offset more than once is routine (e.g. a
    retransmission, or two ground stations catching the same overpass) and
    isn't itself a problem -- `.duplicates` lists every repeated offset, but
    only `.conflicts` (the subset where the repeats *disagree* on the
    bytes) means two different transmissions' chunks likely got selected
    together and the caller's time-range filter needs narrowing.
    """

    total_chunks: int
    unique_offsets: int
    duplicates: tuple[DuplicateOffset, ...]
    gaps: tuple[tuple[int, int], ...]
    data: bytes
    covered_bytes: int
    span_bytes: int

    @property
    def conflicts(self) -> tuple[DuplicateOffset, ...]:
        """Duplicated offsets whose repeated copies don't agree -- the only
        kind of duplicate that actually threatens correctness.
        """
        return tuple(d for d in self.duplicates if not d.consistent)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def is_gapless(self) -> bool:
        return not self.gaps

    @property
    def sha256(self) -> str:
        """SHA-256 of `data` as assembled -- only meaningful to compare
        against a known-good hash once `is_gapless` (a gap zero-fills its
        bytes, which changes the hash) and there are no `.conflicts`
        (a conflicting duplicate means the "wrong" copy may have won).
        """
        return hashlib.sha256(self.data).hexdigest()


def reassemble_bulk_chunks(df: pl.DataFrame) -> ReassemblyResult:
    """Assemble whatever `BULK_FILE_DOWNLINK` chunks are in `df` into one
    byte string, ordered by `bulk_file_offset`.

    `df` must have `bulk_file_offset`, `bulk_data_len`, and `bulk_data_hex`
    columns -- already filtered by the caller (e.g. to one or more
    `received_at` time ranges isolating a single download) to whatever rows
    should be treated as chunks of the same file. This never raises on messy
    input: duplicate offsets and gaps are reported on the result rather than
    resolved here (a duplicated offset's *last* row in `df` wins in `data`),
    so the caller's filter is what makes the result trustworthy, not this
    function.

    Raises:
        ValueError: if the implied file is larger than
            `MAX_REASSEMBLY_SPAN_BYTES` -- almost certainly a corrupt
            `bulk_file_offset` rather than a real file this large.
    """
    if df.is_empty():
        return ReassemblyResult(0, 0, (), (), b"", 0, 0)

    offsets = df["bulk_file_offset"].to_list()
    lengths = df["bulk_data_len"].to_list()
    hexes = df["bulk_data_hex"].to_list()

    # (length, hex) per copy, keyed by offset -- `length` is the decoder's
    # own `bulk_data_len`, trusted as-is rather than re-derived from the hex
    # string (which would silently mask a corrupt odd-length hex value).
    by_offset: dict[int, tuple[int, str]] = {}
    copies_by_offset: dict[int, list[tuple[int, str]]] = {}
    for offset, length, hex_str in zip(offsets, lengths, hexes, strict=True):
        by_offset[offset] = (length, hex_str)  # last one (in `df`'s order) wins
        copies_by_offset.setdefault(offset, []).append((length, hex_str))

    duplicates = tuple(
        DuplicateOffset(
            offset,
            count=len(copies),
            distinct_contents_count=len({hex_str for _length, hex_str in copies}),
            lengths=tuple(sorted({length for length, _hex_str in copies})),
        )
        for offset, copies in sorted(copies_by_offset.items())
        if len(copies) > 1
    )

    span = max(
        (offset + length for offset, (length, _hex) in by_offset.items()), default=0
    )
    if span > MAX_REASSEMBLY_SPAN_BYTES:
        msg = (
            f"Implied file size {span:,} bytes exceeds the "
            f"{MAX_REASSEMBLY_SPAN_BYTES:,}-byte safety cap -- check for a "
            f"corrupt bulk_file_offset in the selected range."
        )
        raise ValueError(msg)

    # Interval-merge for coverage/gaps (cheap: one interval per chunk, not
    # per byte) -- the byte buffer itself still needs an O(bytes) slice
    # assignment per chunk, which is unavoidable since it *is* the output.
    intervals = sorted(
        (offset, offset + length) for offset, (length, _hex) in by_offset.items()
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    buf = bytearray(span)
    for offset, (_length, hex_str) in by_offset.items():
        chunk = bytes.fromhex(hex_str)
        buf[offset : offset + len(chunk)] = chunk

    covered_bytes = sum(end - start for start, end in merged)
    gaps: list[tuple[int, int]] = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < span:
        gaps.append((prev_end, span))

    return ReassemblyResult(
        total_chunks=len(offsets),
        unique_offsets=len(by_offset),
        duplicates=duplicates,
        gaps=tuple(gaps),
        data=bytes(buf),
        covered_bytes=covered_bytes,
        span_bytes=span,
    )


# -- Header candidates --------------------------------------------------------

# Deliberately not a JSON parser: TCMD_RESPONSE's payload is a hard-capped
# 186 bytes (see cts1_decode_satnogs_packets.TCMD_RESPONSE_MAX_DATA), so a
# descriptor naming a long file path routinely gets cut off mid-value (most
# often mid-sha256) with no closing brace at all. Each field is pulled out
# independently so a truncated one just comes back missing/short rather than
# failing the whole parse.
_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]*)"')
_FILE_RE = re.compile(r'"file"\s*:\s*"([^"]*)"')
_FILE_SIZE_RE = re.compile(r'"file_size"\s*:\s*(\d+)')
_SHA256_RE = re.compile(r'"sha256"\s*:\s*"([0-9a-fA-F]*)')
_CRC16_RE = re.compile(r'"crc16"\s*:\s*"([^"]*)"')
_OFFSET_RE = re.compile(r'"offset"\s*:\s*(\d+)')
_LENGTH_RE = re.compile(r'"length"\s*:\s*(\d+)')


@dataclass(slots=True, frozen=True)
class BulkHeaderCandidate:
    """Whatever file-descriptor fields could be pulled out of one
    `TCMD_RESPONSE`'s text. Every field but `received_at` may be `None` --
    either the field wasn't present, or (for `sha256`) it was cut off with
    zero characters recovered.
    """

    received_at: datetime
    action: str | None
    file: str | None
    file_size: int | None
    sha256: str | None  # may be < 64 hex chars if the response was truncated
    crc16: str | None
    offset: int | None
    length: int | None


def _parse_header_text(text: str) -> dict[str, Any] | None:
    """Best-effort field extraction from one `TCMD_RESPONSE` body.

    Returns None if the text doesn't even mention "file"/"file_size" --
    i.e. it's plausibly some other command's response, not a bulk-download
    descriptor at all.
    """
    if '"file"' not in text and '"file_size"' not in text:
        return None

    result: dict[str, Any] = {}
    if (m := _ACTION_RE.search(text)) is not None:
        result["action"] = m.group(1)
    if (m := _FILE_RE.search(text)) is not None:
        result["file"] = m.group(1)
    if (m := _FILE_SIZE_RE.search(text)) is not None:
        result["file_size"] = int(m.group(1))
    if (m := _SHA256_RE.search(text)) is not None and m.group(1):
        result["sha256"] = m.group(1).lower()
    if (m := _CRC16_RE.search(text)) is not None:
        result["crc16"] = m.group(1)
    if (m := _OFFSET_RE.search(text)) is not None:
        result["offset"] = int(m.group(1))
    if (m := _LENGTH_RE.search(text)) is not None:
        result["length"] = int(m.group(1))
    return result


def find_header_candidates(tcmd_df: pl.DataFrame) -> list[BulkHeaderCandidate]:
    """Every `TCMD_RESPONSE` row in `tcmd_df` that looks like a bulk-download
    file descriptor, oldest first.

    `tcmd_df` must have `received_at`, `tcmd_ts_sent`, `tcmd_response_seq_num`,
    and `tcmd_response_text` columns. A response that spans multiple frames
    (`tcmd_response_seq_num`/`_max_seq_num` > 0, all sharing one
    `tcmd_ts_sent`) is concatenated in sequence-number order before parsing
    -- but since a fragment can itself go missing, each individual fragment
    is *also* tried on its own, so a still-recognizable field isn't lost
    just because a sibling fragment didn't make it down.
    """
    if tcmd_df.is_empty():
        return []

    candidates: list[BulkHeaderCandidate] = []
    grouped = tcmd_df.sort("tcmd_response_seq_num").group_by(
        "tcmd_ts_sent", maintain_order=True
    )
    for _, group in grouped:
        received_at = group.select(pl.col("received_at").min()).item()
        fragments = [t for t in group["tcmd_response_text"].to_list() if t]
        # The full reassembly first (it's the one worth trusting when
        # complete), then each fragment on its own -- deduped, but keeping
        # this order rather than a set's arbitrary one so a caller picking
        # "the last/best candidate" reliably lands on the fullest parse.
        candidate_texts = list(dict.fromkeys(["".join(fragments), *fragments]))
        for text in candidate_texts:
            if not text:
                continue
            parsed = _parse_header_text(text)
            if parsed is None:
                continue
            candidates.append(
                BulkHeaderCandidate(
                    received_at=received_at,
                    action=parsed.get("action"),
                    file=parsed.get("file"),
                    file_size=parsed.get("file_size"),
                    sha256=parsed.get("sha256"),
                    crc16=parsed.get("crc16"),
                    offset=parsed.get("offset"),
                    length=parsed.get("length"),
                )
            )
    candidates.sort(key=lambda c: c.received_at)
    return candidates
