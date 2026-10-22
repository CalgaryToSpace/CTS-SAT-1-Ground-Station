"""Step 6: extract MPI telemetry frames from bulk file downlinks.

How the firmware writes an MPI science data file (see the OBC firmware's
`rtos_mpi_tasks.c`/`mpi_command_handling.c`):

    {"mpi_start":1,"uptime_ms":...,"timestamp_ms":...}      <- file header
    [20480 bytes of raw MPI UART stream]
    {"uptime_ms":...,"timestamp":"..._E","timestamp_ms":...} <- timestamp
    [20480 bytes of raw MPI UART stream]
    {"uptime_ms":...,"timestamp":"..._E","timestamp_ms":...} <- timestamp
    ...
    {"data_lost_bytes": 0, "time_taken_ms": ..., ...}        <- file footer

(Files are opened in append mode, so one file can hold several such
recordings back to back.) Every timestamp marks the moment its preceding
20480-byte buffer was written, so each pair of consecutive timestamps
brackets exactly 20480 bytes of MPI data. The MPI streams fixed-size
frames (`0x0C 0xFF 0xFF 0x0C` sync, 2-byte frame counter, housekeeping,
pixels, CRC-16 -- see `_parse_frame`), and since the buffers aren't aligned
to frames, a frame regularly has a timestamp JSON wedged into the middle of
it, which has to be carved back out.

The file comes down as `BULK_FILE_DOWNLINK` packets carrying a byte offset
and up to 195 bytes each -- but nothing saying *which* file. The same
offsets get downlinked again and again (re-downlinks of one file over many
passes, and other files entirely), so the core problem is working out
which chunks belong together. This step does it from the content:

  1. Distinct chunks: packets with the same offset and bytes are the same
     chunk (e.g. one file region downlinked over several passes). Only
     chunks that look like MPI data (a sync word or one of the JSON
     markers) are considered.
  2. Linking chunks into components (union-find), where every link is
     evidence the two chunks are contiguous bytes of the same file:
       - Straddling frame: a chunk and a chunk starting where it ends are
         linked when the MPI frame spanning their boundary passes its
         CRC. The frame counter restarts at 0 on every recording, so a
         frame's leading header bytes come out identical across
         recordings -- the link only counts when the earlier chunk
         contributes at least `STRONG_LINK_MIN_LEADING_BYTES` (i.e. some
         pixel bytes) to that frame.
       - Received together: otherwise (a boundary that splits a frame in
         its header, or falls between frames), two adjacent chunks are
         linked when a copy of each was received in the same SatNOGS
         observation within a few seconds of each other -- i.e. they were
         consecutive packets of one downlink. The same rule bridges over up
         to `MAX_BRIDGED_CHUNKS` never-received chunks. Only `received_at`s
         from a decoder with per-frame timing count (some, e.g.
         gr_satellites_pdu, report the observation's end time for every
         frame).
       - Overlap: chunks at non-aligned offsets (a downlink started at an
         arbitrary offset) are linked when at least
         `OVERLAP_LINK_MIN_BYTES` of their overlap is identical.
     A boundary with more than one candidate is left unlinked. A byte that
     two of a component's chunks disagree on is treated as never received.
  3. Anchors: the JSON timestamps found in each component. Two
     consecutive timestamps form an anchor span when the data between
     them is exactly 20480 bytes (one block), or 2 x 20480 plus a
     plausible JSON length (two blocks whose middle timestamp was never
     received -- its position is still pinned exactly by the 20480-byte
     structure). Both timestamps must share a real time-sync source (not
     `N`, never synced), and their `timestamp_ms` and `uptime_ms` must
     advance by the same amount (no time resync between them). The block
     right after a `mpi_start` header is never a span: its header is
     written before the MPI's 5s power-on delay, and the firmware doesn't
     reset its buffer index between recordings, so that block can start
     with stale data from the previous recording.
  4. Frames: every frame whose last byte falls inside an anchor span, with
     all its bytes received, is parsed. A span is dropped as a whole if
     any of its frames fails its CRC, or if its frames' counters don't
     advance exactly in step with their positions (one frame per frame
     length -- otherwise the MPI data rate wasn't constant, e.g. data was
     lost onboard). Spans whose per-block duration is more than
     `BLOCK_DURATION_TOLERANCE` off from the component's median are
     dropped too.
  5. Timing: a frame's `collected_at` is linearly interpolated between its
     span's two timestamps, by where its last byte falls within the
     span's MPI data -- i.e. roughly when the frame finished arriving at
     the OBC. The timestamps themselves are taken right after each buffer
     is written to flash, so they (and every frame time) lag the true
     buffer-full time by that write's latency (on the order of 100ms).

The same frame recovered from more than one component is kept once,
preferring the narrowest anchor span. Like steps 2-5, this step is
parquet-in-parquet-out and reprocesses from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "BLOCK_DURATION_TOLERANCE",
    "DEFAULT_DATA_DIR",
    "MAX_BRIDGED_CHUNKS",
    "MPI_BUFFER_LEN",
    "MPI_SYNC",
    "OUTPUT_FILENAME",
    "OVERLAP_LINK_MIN_BYTES",
    "STRONG_LINK_MIN_LEADING_BYTES",
    "compute_mpi_frames",
    "mpi_crc16",
    "run",
]

import binascii
import json
import re
import statistics
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise, product
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    BULK_DOWNLINK_MAX_DATA,
    MAX_VALID_EPOCH_MS,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets.pipeline import (
    BASELINE_DECODERS,
    DEMOD_DECODER,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "mpi_frames.parquet"

MPI_SYNC = b"\x0c\xff\xff\x0c"
# Bytes of MPI data the firmware writes between two timestamp JSONs.
MPI_BUFFER_LEN = 20480
# Frame layout: 20 header bytes, 2 bytes per pixel, 2 CRC bytes.
_FRAME_HEADER_LEN = 20
_FRAME_CRC_LEN = 2
_FRAME_FIXED_LEN = _FRAME_HEADER_LEN + _FRAME_CRC_LEN
_MAX_PIXEL_INDEX = 132
_MAX_FRAME_LEN = _FRAME_FIXED_LEN + 2 * (_MAX_PIXEL_INDEX + 1)

# Linking thresholds -- see the module docstring.
STRONG_LINK_MIN_LEADING_BYTES = _FRAME_HEADER_LEN + 4
OVERLAP_LINK_MIN_BYTES = 100
MAX_BRIDGED_CHUNKS = 3
# Consecutive packets of one downlink go out ~250ms apart; this leaves
# plenty of room for retries and ground station clock jitter while still
# being far shorter than any downlink of a different file could be.
_RECEIVED_TOGETHER_MAX_SECONDS = 10.0
_RECEIVED_TOGETHER_SECONDS_PER_BRIDGED_CHUNK = 1.0
_RECEIVED_TOGETHER_MIN_SECONDS = -1.0  # Allows for decoders' 1s-granularity.

# Anchor span sanity checks -- see the module docstring.
_MAX_SPAN_BLOCKS = 2
_MISSING_TIMESTAMP_LEN_RANGE = (100, 200)
_UNSYNCED_TIME_SOURCE = "N"
_MAX_RESYNC_DRIFT_MS = 2
_BLOCK_DURATION_RANGE_MS = (1_000, 60_000)
BLOCK_DURATION_TOLERANCE = 0.10
_MIN_SPANS_FOR_MEDIAN = 3

# Inner dome voltage conversion, from the MPI's data packet definition.
_INNER_DOME_ADC_MASK = 0x3FF
_INNER_DOME_VOLTS_PER_COUNT = 0.105361
_INNER_DOME_VOLTS_OFFSET = -101.808
_BOARD_TEMPERATURE_COUNTS_PER_C = 128.0

# The firmware's JSON markers never contain braces. The `\Z` alternative
# also catches one cut off at the end of the buffer (only used when
# checking a chunk boundary, where the rest is in the next chunk).
_MARKER_RE = re.compile(rb'\{"(?:mpi_start|uptime_ms|data_lost_bytes)"[^{}]{0,250}\}')
_MARKER_OR_TRUNCATED_RE = re.compile(
    rb'\{"(?:mpi_start|uptime_ms|data_lost_bytes)"[^{}]{0,250}(?:\}|\Z)'
)
_TIMESTAMP_STR_RE = re.compile(r"(?P<sync_epoch>\d+)\+\d+_(?P<source>\w)")
_LOOKS_LIKE_MPI_RE = re.compile(
    re.escape(MPI_SYNC) + rb'|\{"(?:mpi_start|uptime_ms|data_lost_bytes)"'
)

_BIT_REVERSED_BYTES = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))

_OUTPUT_SCHEMA = {
    "collected_at": pl.Datetime("us", "UTC"),
    "obc_uptime_ms": pl.Int64,
    "ms_since_time_sync": pl.Int64,
    "time_sync_source": pl.String,
    "anchor_span_blocks": pl.Int64,
    "anchor_before_at": pl.Datetime("us", "UTC"),
    "anchor_after_at": pl.Datetime("us", "UTC"),
    "frame_counter": pl.Int64,
    "board_temperature_c": pl.Float64,
    "firmware_version": pl.Int64,
    "detector_status": pl.Int64,
    "inner_dome_voltage_setting": pl.Int64,
    "inner_dome_scan_index": pl.Int64,
    "inner_dome_voltage_adc": pl.Int64,
    "inner_dome_voltage_v": pl.Float64,
    "first_pixel_index": pl.Int64,
    "last_pixel_index": pl.Int64,
    "integration_period": pl.Int64,
    "pixels": pl.String,  # JSON array, e.g. "[2003,2002,...]"
    "file_offset": pl.Int64,
    "first_received_at": pl.Datetime("us", "UTC"),
    "last_received_at": pl.Datetime("us", "UTC"),
    "packet_ids": pl.String,  # JSON array, e.g. '["ab12","cd34"]'
}


def mpi_crc16(data: bytes) -> int:
    """The MPI frame CRC (CRC-16/MCRF4XX: poly 0x1021 reflected, init
    0xFFFF, no final XOR), over everything in the frame before the CRC.

    `binascii.crc_hqx` is the same polynomial unreflected, so it's run on
    bit-reversed bytes and its result bit-reversed back.
    """
    crc = binascii.crc_hqx(data.translate(_BIT_REVERSED_BYTES), 0xFFFF)
    return int(f"{crc:016b}"[::-1], 2)


def _frame_len_at(stream: bytes | bytearray, start: int) -> int | None:
    """The length of the frame starting at `start`, from its pixel index
    header bytes, or None if those bytes are out of range/nonsensical.
    """
    if start + _FRAME_HEADER_LEN > len(stream):
        return None
    first, last = stream[start + 16], stream[start + 17]
    if not first <= last <= _MAX_PIXEL_INDEX:
        return None
    return _FRAME_FIXED_LEN + 2 * (last - first + 1)


def _frame_crc_ok(frame: bytes) -> bool:
    return mpi_crc16(frame[:-_FRAME_CRC_LEN]) == int.from_bytes(frame[-2:], "big")


def _parse_frame(frame: bytes) -> dict[str, Any]:
    """Decode one CRC-checked MPI frame's fields (all words big-endian)."""

    def u16(i: int) -> int:
        return int.from_bytes(frame[i : i + 2], "big")

    inner_dome_adc = u16(14)
    pixel_count = (len(frame) - _FRAME_FIXED_LEN) // 2
    pixels = [u16(_FRAME_HEADER_LEN + 2 * i) for i in range(pixel_count)]
    return {
        "frame_counter": u16(4),
        "board_temperature_c": (
            int.from_bytes(frame[6:8], "big", signed=True)
            / _BOARD_TEMPERATURE_COUNTS_PER_C
        ),
        "firmware_version": frame[8],
        "detector_status": u16(9),
        "inner_dome_voltage_setting": u16(11),
        "inner_dome_scan_index": frame[13],
        "inner_dome_voltage_adc": inner_dome_adc,
        "inner_dome_voltage_v": (
            (inner_dome_adc & _INNER_DOME_ADC_MASK) * _INNER_DOME_VOLTS_PER_COUNT
            + _INNER_DOME_VOLTS_OFFSET
        ),
        "first_pixel_index": frame[16],
        "last_pixel_index": frame[17],
        "integration_period": u16(18),
        "pixels": json.dumps(pixels, separators=(",", ":")),
    }


@dataclass(slots=True)
class _Sighting:
    """One received packet carrying a chunk."""

    packet_id: str
    received_at: datetime | None
    observation_ids: frozenset[int]
    # Whether `received_at` is this packet's own time, rather than e.g. its
    # observation's end time -- see `_load_chunks`.
    has_own_time: bool


@dataclass(slots=True)
class _Chunk:
    """One distinct (offset, bytes) chunk, and every packet that carried it."""

    offset: int
    data: bytes
    sightings: list[_Sighting] = field(default_factory=list[_Sighting])

    @property
    def end(self) -> int:
        return self.offset + len(self.data)


def _load_chunks(decoded_packets: pl.DataFrame) -> list[_Chunk]:
    """Every distinct MPI-looking chunk among the CRC-valid bulk packets.

    A packet's `received_at` is its own time when it came from a baseline
    decoder, or else (step 2's "estimated", the earliest of its decodes)
    when satnogs_client_live_data was one of its decoders.
    """
    rows = decoded_packets.filter(
        (pl.col("packet_type") == "BULK_FILE_DOWNLINK")
        & pl.col("csp_crc_valid")
        & pl.col("bulk_file_offset").is_not_null()
        & pl.col("bulk_data_hex").is_not_null()
    ).select(
        "packet_id",
        "received_at",
        pl.col("received_at_source").cast(pl.String),
        "decoders",
        "observation_ids",
        "bulk_file_offset",
        "bulk_data_hex",
    )
    chunks: dict[tuple[int, bytes], _Chunk] = {}
    for (
        packet_id,
        received_at,
        source,
        decoders,
        obs_ids,
        offset,
        data_hex,
    ) in rows.iter_rows():
        data = bytes.fromhex(data_hex)
        if not data or not _LOOKS_LIKE_MPI_RE.search(data):
            continue
        chunk = chunks.setdefault((offset, data), _Chunk(offset, data))
        chunk.sightings.append(
            _Sighting(
                packet_id=packet_id,
                received_at=received_at,
                observation_ids=frozenset(json.loads(obs_ids or "[]")),
                has_own_time=(
                    source in BASELINE_DECODERS
                    or DEMOD_DECODER in json.loads(decoders or "[]")
                ),
            )
        )
    return list(chunks.values())


def _carve_markers(
    data: bytes, positions: list[int], pattern: re.Pattern[bytes]
) -> tuple[bytes, list[int]]:
    """Remove every JSON marker matching `pattern` from `data`, and map each
    of `positions` (indexes into `data`) to where it lands afterwards (a
    position inside a removed marker lands at that marker's start).
    """
    out = bytearray()
    removed: list[tuple[int, int]] = []
    last = 0
    for m in pattern.finditer(data):
        out += data[last : m.start()]
        removed.append((m.start(), m.end()))
        last = m.end()
    out += data[last:]
    mapped: list[int] = []
    for pos in positions:
        shift = sum(min(end, pos) - start for start, end in removed if start < pos)
        mapped.append(pos - shift)
    return bytes(out), mapped


_StraddleEvidence = Literal["strong", "weak", "fail"] | None


def _straddle_evidence(first: bytes, second: bytes) -> _StraddleEvidence:
    """What the MPI frame spanning the `first`|`second` boundary says about
    `second` directly following `first` in the same file.

    "strong"/"weak": it passes its CRC, with `first` contributing at least /
    fewer than `STRONG_LINK_MIN_LEADING_BYTES` of it. "fail": it doesn't
    pass. None: no frame spans the boundary in full.
    """
    stream, (boundary,) = _carve_markers(
        first + second, [len(first)], _MARKER_OR_TRUNCATED_RE
    )
    for m in re.finditer(re.escape(MPI_SYNC), stream):
        start = m.start()
        if start >= boundary:
            break
        frame_len = _frame_len_at(stream, start)
        if frame_len is None or start + frame_len <= boundary:
            continue
        if start + frame_len > len(stream):
            return None
        if not _frame_crc_ok(stream[start : start + frame_len]):
            return "fail"
        if boundary - start >= STRONG_LINK_MIN_LEADING_BYTES:
            return "strong"
        return "weak"
    return None


def _received_together(first: _Chunk, second: _Chunk, *, max_seconds: float) -> bool:
    """Whether a copy of `second` was received in the same observation as,
    and at most `max_seconds` after, a copy of `first`.
    """
    for a in first.sightings:
        if not a.has_own_time or a.received_at is None:
            continue
        for b in second.sightings:
            if (
                b.has_own_time
                and b.received_at is not None
                and a.observation_ids & b.observation_ids
                and _RECEIVED_TOGETHER_MIN_SECONDS
                <= (b.received_at - a.received_at).total_seconds()
                <= max_seconds
            ):
                return True
    return False


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _adjacent_link(
    chunk: _Chunk, chunks: list[_Chunk], by_offset: dict[int, list[int]]
) -> tuple[int | None, str]:
    """The chunk (index) that follows `chunk` in its file, and the kind of
    link found -- (None, "") when there's no link, (None, "ambiguous") when
    there's more than one candidate.
    """
    # A chunk starting exactly where this one ends.
    following = by_offset.get(chunk.end, [])
    evidence = {j: _straddle_evidence(chunk.data, chunks[j].data) for j in following}
    strong = [j for j, e in evidence.items() if e == "strong"]
    together = [
        j
        for j, e in evidence.items()
        if e != "fail"
        and _received_together(
            chunk, chunks[j], max_seconds=_RECEIVED_TOGETHER_MAX_SECONDS
        )
    ]
    for candidates, kind in (
        (strong, "straddling_frame"),
        (together, "received_together"),
    ):
        if len(candidates) > 1:
            return None, "ambiguous"
        if candidates:
            return candidates[0], kind

    # A chunk a few never-received chunks further on.
    if len(chunk.data) != BULK_DOWNLINK_MAX_DATA:
        return None, ""
    for skipped in range(1, MAX_BRIDGED_CHUNKS + 1):
        max_seconds = (
            _RECEIVED_TOGETHER_MAX_SECONDS
            + skipped * _RECEIVED_TOGETHER_SECONDS_PER_BRIDGED_CHUNK
        )
        candidates = by_offset.get(chunk.end + skipped * len(chunk.data), [])
        together = [
            j
            for j in candidates
            if _received_together(chunk, chunks[j], max_seconds=max_seconds)
        ]
        if len(together) > 1:
            return None, "ambiguous"
        if together:
            return together[0], "bridged"
        if candidates:
            break
    return None, ""


def _overlap_links(
    chunks: list[_Chunk], by_offset: dict[int, list[int]]
) -> Iterator[tuple[int, int]]:
    """Every pair of chunks (indexes) at different alignments whose overlap
    is identical and at least `OVERLAP_LINK_MIN_BYTES` long.
    """
    sorted_offsets = sorted(by_offset)
    for oi, offset in enumerate(sorted_offsets):
        for other in sorted_offsets[oi:]:
            if other >= offset + BULK_DOWNLINK_MAX_DATA:
                break
            for i, j in product(by_offset[offset], by_offset[other]):
                a, b = chunks[i], chunks[j]
                overlap_end = min(a.end, b.end)
                if (
                    i != j
                    and overlap_end - b.offset >= OVERLAP_LINK_MIN_BYTES
                    and a.data[b.offset - a.offset : overlap_end - a.offset]
                    == b.data[: overlap_end - b.offset]
                ):
                    yield i, j


def _link_chunks(chunks: list[_Chunk]) -> tuple[list[list[int]], Counter[str]]:
    """Group `chunks` into components that are each (as far as the content
    and reception times can tell) from one file -- see the module docstring.

    Returns each component's chunk indexes, and a tally of link kinds.
    """
    by_offset: dict[int, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_offset[chunk.offset].append(i)
    uf = _UnionFind(len(chunks))
    tally: Counter[str] = Counter()

    for i, chunk in enumerate(chunks):
        j, kind = _adjacent_link(chunk, chunks, by_offset)
        if kind:
            tally[kind] += 1
        if j is not None:
            uf.union(i, j)
    for i, j in _overlap_links(chunks, by_offset):
        uf.union(i, j)
        tally["overlap"] += 1

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(chunks)):
        components[uf.find(i)].append(i)
    return list(components.values()), tally


@dataclass(slots=True)
class _Component:
    """A component's chunks laid out into one sparse byte buffer."""

    base: int  # File offset of `data[0]`.
    data: bytearray
    have: bytearray  # 1 where `data` holds a received, uncontested byte.
    chunks: list[_Chunk]  # Sorted by offset.
    chunk_offsets: list[int]
    conflicting_bytes: int

    def has_all(self, start: int, end: int) -> bool:
        """Whether every byte in file offsets [start, end) is present."""
        lo, hi = start - self.base, end - self.base
        return lo >= 0 and hi <= len(self.have) and all(self.have[lo:hi])

    def get(self, start: int, end: int) -> bytes:
        return bytes(self.data[start - self.base : end - self.base])

    def chunks_covering(self, start: int, end: int) -> list[_Chunk]:
        first = bisect_left(self.chunk_offsets, start - BULK_DOWNLINK_MAX_DATA)
        return [c for c in self.chunks[first:] if c.offset < end and c.end > start]


def _assemble(chunks: list[_Chunk]) -> _Component:
    chunks = sorted(chunks, key=lambda c: c.offset)
    base = chunks[0].offset
    size = max(c.end for c in chunks) - base
    data = bytearray(size)
    have = bytearray(size)
    contested = bytearray(size)
    for chunk in chunks:
        lo = chunk.offset - base
        for k, value in enumerate(chunk.data):
            pos = lo + k
            if contested[pos]:
                continue
            if have[pos] and data[pos] != value:
                contested[pos] = 1
                have[pos] = 0
            else:
                data[pos] = value
                have[pos] = 1
    return _Component(
        base=base,
        data=data,
        have=have,
        chunks=chunks,
        chunk_offsets=[c.offset for c in chunks],
        conflicting_bytes=sum(contested),
    )


@dataclass(slots=True, frozen=True)
class _Marker:
    start: int  # File offsets, end exclusive.
    end: int
    fields: dict[str, Any]

    @property
    def is_timestamp(self) -> bool:
        return "uptime_ms" in self.fields and "mpi_start" not in self.fields


def _find_markers(component: _Component) -> list[_Marker]:
    """Every fully-received, parseable JSON marker in `component`."""
    markers: list[_Marker] = []
    for m in _MARKER_RE.finditer(component.data):
        start, end = m.start() + component.base, m.end() + component.base
        if not component.has_all(start, end):
            continue
        try:
            fields = json.loads(m.group())
        except ValueError:
            continue
        if isinstance(fields, dict):
            markers.append(_Marker(start, end, cast("dict[str, Any]", fields)))
    return markers


@dataclass(slots=True, frozen=True)
class _AnchorSpan:
    """Two timestamp markers bracketing `blocks` x 20480 bytes of MPI data,
    plus the file offsets of any never-received timestamp between them.
    """

    before: _Marker
    after: _Marker
    blocks: int
    gaps: tuple[tuple[int, int], ...]
    sync_source: str

    @property
    def duration_ms(self) -> int:
        return self.after.fields["uptime_ms"] - self.before.fields["uptime_ms"]


def _timestamp_fields(marker: _Marker) -> tuple[int, int, str, int] | None:
    """(uptime_ms, timestamp_ms, sync source letter, ms since that sync), if
    well-formed. The last two come from the `timestamp` string,
    "<epoch ms at last sync>+<ms since sync>_<source>" -- except the time
    since sync is worked out as `timestamp_ms` minus the sync epoch, as the
    firmware's own "<ms since sync>" is garbage: it's computed from the
    (truncated) epoch time where it expects the uptime.
    """
    uptime = marker.fields.get("uptime_ms")
    epoch_ms = marker.fields.get("timestamp_ms")
    timestamp = marker.fields.get("timestamp")
    if not (
        isinstance(uptime, int)
        and isinstance(epoch_ms, int)
        and isinstance(timestamp, str)
        and 0 < epoch_ms <= MAX_VALID_EPOCH_MS
    ):
        return None
    m = _TIMESTAMP_STR_RE.fullmatch(timestamp)
    if m is None:
        return None
    return uptime, epoch_ms, m["source"], epoch_ms - int(m["sync_epoch"])


def _span_sync_source(before: _Marker, after: _Marker) -> str | None:
    """The time-sync source both timestamps were taken under, if they're a
    usable pair: well-formed, synced, and with no time resync between them.
    """
    if not (before.is_timestamp and after.is_timestamp):
        return None
    before_ts, after_ts = _timestamp_fields(before), _timestamp_fields(after)
    if before_ts is None or after_ts is None:
        return None
    (up_a, epoch_a, source_a, _), (up_b, epoch_b, source_b, _) = before_ts, after_ts
    if source_a != source_b or source_a == _UNSYNCED_TIME_SOURCE:
        return None
    if abs((epoch_b - epoch_a) - (up_b - up_a)) > _MAX_RESYNC_DRIFT_MS:
        return None
    return source_a


def _span_layout(
    component: _Component, before: _Marker, after: _Marker
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """How many 20480-byte blocks lie between `before` and `after`, and the
    file offsets of the never-received timestamps between those blocks --
    or None if the gap between them doesn't fit that layout.
    """
    gap = after.start - before.end
    min_len, max_len = _MISSING_TIMESTAMP_LEN_RANGE
    for blocks in range(1, _MAX_SPAN_BLOCKS + 1):
        missing = blocks - 1
        if not min_len * missing <= gap - blocks * MPI_BUFFER_LEN <= max_len * missing:
            continue
        if missing == 0:
            return blocks, ()
        gap_start = before.end + MPI_BUFFER_LEN
        gap_end = after.start - MPI_BUFFER_LEN
        # Whatever bytes of the missing timestamp did arrive must look like
        # the start/end of one.
        for pos, expected in ((gap_start, b"{"), (gap_end - 1, b"}")):
            if (
                component.has_all(pos, pos + 1)
                and component.get(pos, pos + 1) != expected
            ):
                return None
        return blocks, ((gap_start, gap_end),)
    return None


def _anchor_span(
    component: _Component, before: _Marker, after: _Marker
) -> _AnchorSpan | None:
    """The span between consecutive markers `before` and `after`, or None
    if they don't make a valid one -- see the module docstring.
    """
    sync_source = _span_sync_source(before, after)
    layout = _span_layout(component, before, after)
    if sync_source is None or layout is None:
        return None
    blocks, gaps = layout
    span = _AnchorSpan(before, after, blocks, gaps, sync_source)
    low, high = _BLOCK_DURATION_RANGE_MS
    if not low <= span.duration_ms / blocks <= high:
        return None
    return span


def _data_runs(span: _AnchorSpan) -> list[tuple[int, int]]:
    """The file offset ranges [start, end) holding the span's MPI data, in
    order -- led by the tail of the previous block, where a frame ending in
    this span can begin.
    """
    runs = [(span.before.start - _MAX_FRAME_LEN, span.before.start)]
    cursor = span.before.end
    for gap_start, gap_end in span.gaps:
        runs.append((cursor, gap_start))
        cursor = gap_end
    runs.append((cursor, span.after.start))
    return runs


@dataclass(slots=True)
class _FrameCandidate:
    stream_start: int
    frame: bytes
    file_offsets: list[int]  # File offset of each byte of `frame`.


def _span_stream(
    component: _Component, span: _AnchorSpan
) -> tuple[bytearray, bytearray, list[int], int]:
    """The span's MPI data with the timestamps carved out: its bytes, which
    of them were received, each one's file offset, and how many bytes of
    the previous block lead the span itself.
    """
    runs = _data_runs(span)
    stream = bytearray()
    have = bytearray()
    file_offsets: list[int] = []
    for start, end in runs:
        for pos in range(max(start, component.base), end):
            idx = pos - component.base
            stream.append(component.data[idx])
            have.append(component.have[idx])
            file_offsets.append(pos)
    lead = runs[0][1] - max(runs[0][0], component.base)
    return stream, have, file_offsets, lead


def _frames_consistent(candidates: list[_FrameCandidate]) -> bool:
    """Whether every frame passes its CRC, and they're all the same length
    with counters advancing exactly one per frame length of stream.
    """
    first = candidates[0]
    frame_len = len(first.frame)
    first_counter = int.from_bytes(first.frame[4:6], "big")
    for c in candidates:
        if len(c.frame) != frame_len or not _frame_crc_ok(c.frame):
            return False
        frames_apart, misaligned = divmod(
            c.stream_start - first.stream_start, frame_len
        )
        counter = int.from_bytes(c.frame[4:6], "big")
        if misaligned or (counter - first_counter) % 0x10000 != frames_apart % 0x10000:
            return False
    return True


def _frames_in_span(
    component: _Component, span: _AnchorSpan
) -> list[_FrameCandidate] | None:
    """Every fully-received frame ending inside `span` (`stream_start`
    relative to the span's first data byte), or None if the span turns out
    inconsistent -- see `_frames_consistent`.
    """
    stream, have, file_offsets, lead = _span_stream(component, span)
    span_end = lead + span.blocks * MPI_BUFFER_LEN

    candidates: list[_FrameCandidate] = []
    for m in re.finditer(re.escape(MPI_SYNC), stream):
        start = m.start()
        if not all(have[start : start + _FRAME_HEADER_LEN]):
            continue
        frame_len = _frame_len_at(stream, start)
        if frame_len is None:
            continue
        end = start + frame_len
        if lead < end <= span_end and all(have[start:end]):
            candidates.append(
                _FrameCandidate(
                    stream_start=start,
                    frame=bytes(stream[start:end]),
                    file_offsets=file_offsets[start:end],
                )
            )

    if not candidates:
        return []
    if not _frames_consistent(candidates):
        return None
    for c in candidates:
        c.stream_start -= lead
    return candidates


def _frame_rows(
    component: _Component, span: _AnchorSpan, candidates: list[_FrameCandidate]
) -> list[dict[str, Any]]:
    """Output rows for `span`'s frames, with interpolated times."""
    up_a = span.before.fields["uptime_ms"]
    before_ts = _timestamp_fields(span.before)
    assert before_ts is not None  # Checked when the span was made.
    since_sync_a = before_ts[3]
    epoch_a = span.before.fields["timestamp_ms"]
    epoch_b = span.after.fields["timestamp_ms"]
    span_bytes = span.blocks * MPI_BUFFER_LEN
    rows: list[dict[str, Any]] = []
    for c in candidates:
        fraction = (c.stream_start + len(c.frame)) / span_bytes
        offset_ms = fraction * span.duration_ms
        sightings = [
            s
            for chunk in component.chunks_covering(
                c.file_offsets[0], c.file_offsets[-1] + 1
            )
            for s in chunk.sightings
        ]
        received = [s.received_at for s in sightings if s.received_at is not None]
        rows.append(
            {
                "collected_at": _epoch_ms_to_datetime(epoch_a + offset_ms),
                "obc_uptime_ms": round(up_a + offset_ms),
                "ms_since_time_sync": round(since_sync_a + offset_ms),
                "time_sync_source": span.sync_source,
                "anchor_span_blocks": span.blocks,
                "anchor_before_at": _epoch_ms_to_datetime(epoch_a),
                "anchor_after_at": _epoch_ms_to_datetime(epoch_b),
                **_parse_frame(c.frame),
                "file_offset": c.file_offsets[0],
                "first_received_at": min(received, default=None),
                "last_received_at": max(received, default=None),
                "packet_ids": json.dumps(sorted({s.packet_id for s in sightings})),
                "_frame": c.frame,
            }
        )
    return rows


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _epoch_ms_to_datetime(epoch_ms: float) -> datetime:
    return _EPOCH + timedelta(milliseconds=epoch_ms)


def _spans_for_component(
    component: _Component, tally: Counter[str]
) -> list[_AnchorSpan]:
    markers = _find_markers(component)
    spans: list[_AnchorSpan] = []
    for before, after in pairwise(markers):
        span = _anchor_span(component, before, after)
        if span is None:
            if before.is_timestamp and after.is_timestamp:
                tally["spans_rejected_anchors"] += 1
            continue
        spans.append(span)

    per_block = [s.duration_ms / s.blocks for s in spans]
    if len(per_block) >= _MIN_SPANS_FOR_MEDIAN:
        median = statistics.median(per_block)
        kept = [
            s
            for s, d in zip(spans, per_block, strict=True)
            if abs(d - median) <= BLOCK_DURATION_TOLERANCE * median
        ]
        tally["spans_rejected_duration"] += len(spans) - len(kept)
        spans = kept
    return spans


def compute_mpi_frames(decoded_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: `everything_decoded` -> one row per MPI frame
    recovered from bulk file downlinks, with its interpolated collection
    time.

    Args:
        decoded_packets: `everything_decoded.parquet`, as written by step 3.

    Returns:
        One row per frame, oldest first -- see the module docstring for how
        frames are recovered and timed, and `_OUTPUT_SCHEMA` for the
        columns.
    """
    if decoded_packets.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    rows, _tally = _compute_rows(decoded_packets)
    return _rows_to_df(rows)


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    return (
        pl.DataFrame(rows, schema_overrides=_OUTPUT_SCHEMA, infer_schema_length=None)
        .select(pl.col(name).cast(dtype) for name, dtype in _OUTPUT_SCHEMA.items())
        .sort("collected_at", "file_offset")
    )


def _compute_rows(
    decoded_packets: pl.DataFrame,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    chunks = _load_chunks(decoded_packets)
    if not chunks:
        return [], Counter()
    groups, tally = _link_chunks(chunks)
    tally["chunks"] = len(chunks)
    tally["components"] = len(groups)

    # Frame bytes -> best row so far (narrowest span, then earliest).
    best: dict[bytes, dict[str, Any]] = {}
    for group in groups:
        component = _assemble([chunks[i] for i in group])
        tally["conflicting_bytes"] += component.conflicting_bytes
        for span in _spans_for_component(component, tally):
            candidates = _frames_in_span(component, span)
            if candidates is None:
                tally["spans_rejected_frames"] += 1
                continue
            tally["spans"] += 1
            for row in _frame_rows(component, span, candidates):
                frame = row.pop("_frame")
                current = best.get(frame)
                key = (row["anchor_span_blocks"], row["collected_at"])
                if current is None or key < (
                    current["anchor_span_blocks"],
                    current["collected_at"],
                ):
                    best[frame] = row
    return list(best.values()), tally


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `mpi_frames.parquet` from step 3's output.

    Reads `everything_decoded.parquet` from `data_dir` (where step 3
    writes it) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    if not decoded_path.exists():
        msg = f"{decoded_path} not found -- run step_3 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {decoded_path}")
    bulk_packets = (
        pl.scan_parquet(decoded_path)
        .filter(pl.col("packet_type") == "BULK_FILE_DOWNLINK")
        .select(
            "packet_id",
            "packet_type",
            "csp_crc_valid",
            "received_at",
            "received_at_source",
            "decoders",
            "observation_ids",
            "bulk_file_offset",
            "bulk_data_hex",
        )
        .collect()
    )
    rows, tally = _compute_rows(bulk_packets)
    result_df = _rows_to_df(rows)

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)
    logger.info(f"MPI chunk linking/anchoring stats: {dict(sorted(tally.items()))}")
    logger.info(
        f"Done. {result_df.height:,} MPI frame(s) written to {out_path}, from "
        f"{tally['spans']:,} anchor span(s)."
    )
