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
     them is exactly 20480 bytes (one block), or up to `_MAX_SPAN_BLOCKS`
     x 20480 plus a plausible JSON length per never-received timestamp in
     between. Each missing timestamp starts exactly 20480 bytes after the
     last; with more than one missing, every split of their total length
     that fits whatever bytes of them were received is tried, and the
     span is only kept if exactly one split gives consistent frames (or
     all of those that do agree on every frame). Both timestamps must
     share a real time-sync source (not `N`, never synced), and their
     `timestamp_ms` and `uptime_ms` must advance by the same amount (no
     time resync between them).
  4. Frames: every frame whose last byte falls inside an anchor span, with
     all its bytes received, is parsed. A span is dropped as a whole if
     any of its frames fails its CRC, or if its frames' counters don't
     advance exactly in step with their positions (one frame per frame
     length -- otherwise the MPI data rate wasn't constant, e.g. data was
     lost onboard). Spans whose per-block duration is more than
     `BLOCK_DURATION_TOLERANCE` off from the component's median are
     dropped too.
  5. Timing (`timing_method` "interpolated"): a frame's `collected_at` is
     linearly interpolated between its span's two timestamps, by where its
     last byte falls within the span's MPI data -- i.e. roughly when the
     frame finished arriving at the OBC. The timestamps themselves are
     taken right after each buffer is written to flash, so they (and every
     frame time) lag the true buffer-full time by that write's latency (on
     the order of 100ms, with jitter of a few tens of ms).
  6. Extrapolation (`timing_method` "extrapolated"): a block next to a
     synced timestamp but with nothing received on its other side (most
     often where the received data runs out) is timed from that one
     timestamp, at the MPI data rate measured as the median over every
     anchor span with the same frame length and integration period. Its
     frames get the same all-or-nothing consistency check as a span's.
     This includes the first block of a recording, which can't make a
     span (its header is written before the MPI's 5s power-on delay): the
     firmware doesn't reset its buffer index between recordings, so that
     block can start with stale frames from the previous recording, which
     fail the counter check and drop the block. Not extrapolated: a block
     whose other timestamp was received but rejected (unsynced, resynced,
     odd duration, inconsistent frames), as the timestamps are suspect.

The same frame recovered more than once is kept once, preferring
interpolated over extrapolated, then the narrowest anchor span. Every
distinct CRC-valid frame received but not kept is tallied by why (see
`compute_mpi_frames_with_outcomes`).

Frames are then grouped into MPI collections (`assign_mpi_collections`):
runs of frames from one recording, at most `MPI_COLLECTION_MAX_DURATION`
long, each identified by its first frame (time and counter). A second
output, `mpi_collections.parquet`, summarizes each one: its start/end,
frames present vs expected from the counters, and its largest gap
(`summarize_mpi_collections`).

The satellite can also summarize an MPI data file itself, on board, with
the `analyze_mpi_data` telecommand: its response (reassembled by step 5)
gives the file's name, size, SHA-256, frame counts, and `earliest`/
`latest` timestamps. Every distinct one is written to
`mpi_analyses.parquet` (`extract_all_mpi_analyses`), and each file's
primary analysis -- the latest version, then the largest -- is what its
collection uses (`primary_mpi_analyses`), along with the strong signal
count at every threshold the file was analyzed with. Every frame collection
overlapping an analysis's `earliest`..`latest` is merged into it, and the
collection is then timed by the analysis -- the whole file on board, not
just the frames that happened to be downlinked -- with the file's metadata
alongside. An analysis with no frames downlinked yet is still a collection.

Like steps 2-5, this step is parquet-in-parquet-out and reprocesses from
scratch on every run. It reads step 5's output too, if it's there (the
collections just aren't merged with any analyses if not).
"""

from __future__ import annotations

__all__ = [
    "ANALYSES_OUTPUT_FILENAME",
    "BLOCK_DURATION_TOLERANCE",
    "COLLECTIONS_OUTPUT_FILENAME",
    "DEFAULT_DATA_DIR",
    "MAX_BRIDGED_CHUNKS",
    "MPI_ANALYSIS_ACTION_PREFIX",
    "MPI_BUFFER_LEN",
    "MPI_COLLECTION_MAX_DURATION",
    "MPI_SYNC",
    "OUTPUT_FILENAME",
    "OVERLAP_LINK_MIN_BYTES",
    "STRONG_LINK_MIN_LEADING_BYTES",
    "assign_mpi_collections",
    "compute_mpi_frames",
    "compute_mpi_frames_with_outcomes",
    "extract_all_mpi_analyses",
    "extract_mpi_analyses",
    "mpi_crc16",
    "primary_mpi_analyses",
    "run",
    "sort_mpi_analyses_by_preference",
    "summarize_mpi_collections",
]

import binascii
import json
import re
import statistics
from bisect import bisect_left, bisect_right
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
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEFAULT_DATA_DIR = step_3_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "mpi_frames.parquet"
COLLECTIONS_OUTPUT_FILENAME = "mpi_collections.parquet"
ANALYSES_OUTPUT_FILENAME = "mpi_analyses.parquet"

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
_MAX_SPAN_BLOCKS = 3
_MISSING_TIMESTAMP_LEN_RANGE = (100, 200)
_UNSYNCED_TIME_SOURCE = "N"
_MAX_RESYNC_DRIFT_MS = 2
_BLOCK_DURATION_RANGE_MS = (1_000, 60_000)
BLOCK_DURATION_TOLERANCE = 0.10
_MIN_SPANS_FOR_MEDIAN = 3

# How a kept frame was timed (`timing_method`).
_INTERPOLATED = "interpolated"
_EXTRAPOLATED = "extrapolated"
# What became of a received frame -- see `compute_mpi_frames_with_outcomes`.
_KEPT_INTERPOLATED = "kept: interpolated between two timestamps"
_KEPT_EXTRAPOLATED = "kept: extrapolated from one timestamp"
_LOST_NO_TIMESTAMP = "lost: no timestamp received in its piece of the file"
_LOST_TOO_FAR = "lost: no usable timestamp within one block"
_LOST_STALE_FIRST_BLOCK = "lost: stale first block of a recording (firmware bug)"
_LOST_LAYOUT = "lost: too many timestamps missing in a row to place them"
_LOST_UNSYNCED = "lost: clock unsynced, or resynced, at its timestamp(s)"
_LOST_DURATION = "lost: unusual block duration between its timestamps"
_LOST_INCONSISTENT = "lost: CRC/frame counter mismatch in its block(s)"
_LOST_NO_RATE = "lost: no data rate measured for its frame length/integration period"
# A pair of timestamps that made an anchor span.
_TIMED = "timed"
# Pairs of timestamps whose blocks can still be extrapolated into: ones
# that aren't both timestamps, or are too far apart to make a span.
_OPEN_PAIR_OUTCOMES = (None, _LOST_LAYOUT)

# MPI collections -- see `assign_mpi_collections`.
MPI_COLLECTION_MAX_DURATION = timedelta(minutes=20)
# A counter drop this small, between frames this close in time, isn't a
# new recording, but two frames timed from different timestamps landing
# slightly out of order (e.g. the two ends of a block extrapolated from
# either side). A recording too short to reach it couldn't be timed anyway:
# that takes at least a 20480-byte block.
_MAX_OUT_OF_ORDER_FRAMES = 64
_MAX_OUT_OF_ORDER_TIME = timedelta(seconds=2)

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

# What can sit between a JSON marker's braces: printable ASCII.
_MARKER_INNER_BYTES = frozenset(range(0x20, 0x7F)) - set(b"{}")

_BIT_REVERSED_BYTES = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))

_OUTPUT_SCHEMA = {
    "collected_at": pl.Datetime("us", "UTC"),
    "obc_uptime_ms": pl.Int64,
    "ms_since_time_sync": pl.Int64,
    "time_sync_source": pl.String,
    "timing_method": pl.String,  # "interpolated" or "extrapolated"
    "anchor_span_blocks": pl.Int64,  # Null when extrapolated.
    "anchor_before_at": pl.Datetime("us", "UTC"),
    "anchor_after_at": pl.Datetime("us", "UTC"),
    "frame_counter": pl.Int64,
    # The MPI collection the frame is in, by its first frame: see
    # `assign_mpi_collections`.
    "collection_started_at": pl.Datetime("us", "UTC"),
    "collection_first_frame_counter": pl.Int64,
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
    # How many times the whole frame was received: the fewest distinct
    # `everything_decoded` packets carrying any one of its chunks.
    "times_received": pl.Int64,
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
        return "uptime_ms" in self.fields and not self.is_header

    @property
    def is_header(self) -> bool:
        return "mpi_start" in self.fields


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
    plus the file offsets of any never-received timestamps between them.
    """

    before: _Marker
    after: _Marker
    blocks: int
    gaps: tuple[tuple[int, int], ...]
    sync_source: str

    @property
    def duration_ms(self) -> int:
        return self.after.fields["uptime_ms"] - self.before.fields["uptime_ms"]

    @property
    def ms_per_byte(self) -> float:
        return self.duration_ms / (self.blocks * MPI_BUFFER_LEN)


def _timestamp_fields(marker: _Marker) -> tuple[int, int, str, int] | None:
    """(uptime_ms, timestamp_ms, sync source letter, ms since that sync), if
    well-formed. The last two come from the `timestamp` string,
    "<epoch ms at last sync>+<ms since sync>_<source>" -- except the time
    since sync is worked out as `timestamp_ms` minus the sync epoch, as the
    firmware's own "<ms since sync>" is garbage: it's computed from the
    (truncated) epoch time where it expects the uptime. Workaround for
    https://github.com/CalgaryToSpace/CTS-SAT-1-OBC-Firmware/issues/676
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


def _synced_source(marker: _Marker) -> str | None:
    """The time-sync source of a well-formed, synced timestamp."""
    fields = _timestamp_fields(marker) if marker.is_timestamp else None
    if fields is None or fields[2] == _UNSYNCED_TIME_SOURCE:
        return None
    return fields[2]


def _span_sync_source(before: _Marker, after: _Marker) -> str | None:
    """The time-sync source both timestamps were taken under, if they're a
    usable pair: well-formed, synced, and with no time resync between them.
    """
    source = _synced_source(before)
    if source is None or _synced_source(after) != source:
        return None
    up_a, epoch_a = before.fields["uptime_ms"], before.fields["timestamp_ms"]
    up_b, epoch_b = after.fields["uptime_ms"], after.fields["timestamp_ms"]
    if abs((epoch_b - epoch_a) - (up_b - up_a)) > _MAX_RESYNC_DRIFT_MS:
        return None
    return source


def _could_be_missing_marker(component: _Component, start: int, end: int) -> bool:
    """Whether whatever bytes of file offsets [start, end) were received
    could be a JSON timestamp's: `{` first, `}` last, and printable ASCII
    without braces in between.
    """
    for pos in range(start, end):
        idx = pos - component.base
        if not (0 <= idx < len(component.have) and component.have[idx]):
            continue
        byte = component.data[idx]
        if pos == start:
            ok = byte == ord("{")
        elif pos == end - 1:
            ok = byte == ord("}")
        else:
            ok = byte in _MARKER_INNER_BYTES
        if not ok:
            return False
    return True


def _gap_layouts(
    component: _Component, pos: int, missing: int, total: int
) -> Iterator[tuple[tuple[int, int], ...]]:
    """Every way to fit `missing` never-received timestamps, `total` bytes
    long in all, the first at file offset `pos` and each next one
    `MPI_BUFFER_LEN` bytes after the last -- as their [start, end) offsets.
    """
    if missing == 0:
        if total == 0:
            yield ()
        return
    min_len, max_len = _MISSING_TIMESTAMP_LEN_RANGE
    for length in range(min_len, max_len + 1):
        rest = total - length
        if not min_len * (missing - 1) <= rest <= max_len * (missing - 1):
            continue
        if not _could_be_missing_marker(component, pos, pos + length):
            continue
        for tail in _gap_layouts(
            component, pos + length + MPI_BUFFER_LEN, missing - 1, rest
        ):
            yield ((pos, pos + length), *tail)


def _anchor_spans(
    component: _Component, before: _Marker, after: _Marker
) -> list[_AnchorSpan] | str:
    """Every way consecutive timestamps `before` and `after` could bracket
    whole blocks of MPI data (more than one when the missing timestamps
    between them could sit in more than one place) -- or, when they can't,
    why not (a `_LOST_*` outcome). See the module docstring.
    """
    sync_source = _span_sync_source(before, after)
    if sync_source is None:
        return _LOST_UNSYNCED
    gap = after.start - before.end
    min_len, max_len = _MISSING_TIMESTAMP_LEN_RANGE
    for blocks in range(1, _MAX_SPAN_BLOCKS + 1):
        missing, total = blocks - 1, gap - blocks * MPI_BUFFER_LEN
        if min_len * missing <= total <= max_len * missing:
            break
    else:
        return _LOST_LAYOUT
    low, high = _BLOCK_DURATION_RANGE_MS
    duration_ms = after.fields["uptime_ms"] - before.fields["uptime_ms"]
    if not low <= duration_ms / blocks <= high:
        return _LOST_DURATION
    spans = [
        _AnchorSpan(before, after, blocks, gaps, sync_source)
        for gaps in _gap_layouts(component, before.end + MPI_BUFFER_LEN, missing, total)
    ]
    return spans or _LOST_LAYOUT


@dataclass(slots=True)
class _Stream:
    """MPI data pieced together from ranges of file offsets (i.e. with the
    JSON markers between them left out). Bytes outside the component are
    zeros that weren't received.
    """

    data: bytearray = field(default_factory=bytearray)
    have: bytearray = field(default_factory=bytearray)
    run_starts: list[int] = field(default_factory=list[int])  # Stream indexes.
    run_offsets: list[int] = field(default_factory=list[int])  # File offsets.

    def file_offset(self, index: int) -> int:
        run = bisect_right(self.run_starts, index) - 1
        return self.run_offsets[run] + index - self.run_starts[run]


def _stream(component: _Component, runs: list[tuple[int, int]]) -> _Stream:
    stream = _Stream()
    lo, hi = component.base, component.base + len(component.data)
    for start, end in runs:
        stream.run_starts.append(len(stream.data))
        stream.run_offsets.append(start)
        inner_start = min(max(start, lo), end)
        inner_end = max(min(end, hi), inner_start)
        padding = bytes(inner_start - start)
        stream.data += padding + component.data[inner_start - lo : inner_end - lo]
        stream.have += padding + component.have[inner_start - lo : inner_end - lo]
        stream.data += bytes(end - inner_end)
        stream.have += bytes(end - inner_end)
    return stream


@dataclass(slots=True)
class _FrameCandidate:
    stream_start: int
    frame: bytes
    # Bytes of MPI data from the anchor timestamp to the frame's end
    # (negative when the frame comes before it).
    end_from_anchor: int
    first_offset: int  # File offsets of the frame's first and last bytes.
    last_offset: int


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


def _frames_in_region(
    component: _Component,
    runs: list[tuple[int, int]],
    *,
    region: tuple[int, int],
    anchor_pos: int,
) -> list[_FrameCandidate] | None:
    """Every fully-received frame in the stream of `runs` whose end falls
    in `region` (stream positions, (start, end]), measured from the anchor
    timestamp's position `anchor_pos` -- or None if they turn out
    inconsistent (see `_frames_consistent`).
    """
    stream = _stream(component, runs)
    candidates: list[_FrameCandidate] = []
    for m in re.finditer(re.escape(MPI_SYNC), stream.data):
        start = m.start()
        if not all(stream.have[start : start + _FRAME_HEADER_LEN]):
            continue
        frame_len = _frame_len_at(stream.data, start)
        if frame_len is None:
            continue
        end = start + frame_len
        if region[0] < end <= region[1] and all(stream.have[start:end]):
            candidates.append(
                _FrameCandidate(
                    stream_start=start,
                    frame=bytes(stream.data[start:end]),
                    end_from_anchor=end - anchor_pos,
                    first_offset=stream.file_offset(start),
                    last_offset=stream.file_offset(end - 1),
                )
            )
    if candidates and not _frames_consistent(candidates):
        return None
    return candidates


def _frames_in_span(
    component: _Component, span: _AnchorSpan
) -> list[_FrameCandidate] | None:
    """Every fully-received frame ending inside `span` -- or None if they
    turn out inconsistent. The stream is led by the tail of the previous
    block, where a frame ending in this span can begin.
    """
    runs = [(span.before.start - _MAX_FRAME_LEN, span.before.start)]
    cursor = span.before.end
    for gap_start, gap_end in span.gaps:
        runs.append((cursor, gap_start))
        cursor = gap_end
    runs.append((cursor, span.after.start))
    lead = _MAX_FRAME_LEN
    return _frames_in_region(
        component,
        runs,
        region=(lead, lead + span.blocks * MPI_BUFFER_LEN),
        anchor_pos=lead,
    )


def _frames_in_any_layout(
    component: _Component, spans: list[_AnchorSpan]
) -> tuple[_AnchorSpan, list[_FrameCandidate]] | str:
    """The frames of the one way of placing the missing timestamps between
    two timestamps (`spans`, one per way) whose frames are consistent --
    or, if several are, of the first, so long as they all agree on every
    frame and its position. Otherwise why not (a `_LOST_*` outcome).
    """
    consistent = [
        (span, candidates)
        for span in spans
        if (candidates := _frames_in_span(component, span)) is not None
    ]
    if not consistent:
        return _LOST_INCONSISTENT
    placements = {
        tuple((c.frame, c.end_from_anchor) for c in candidates)
        for _, candidates in consistent
    }
    if len(placements) > 1:
        return _LOST_LAYOUT
    return consistent[0]


def _timed_spans(
    component: _Component, markers: list[_Marker]
) -> tuple[list[tuple[_AnchorSpan, list[_FrameCandidate]]], dict[int, str]]:
    """Every anchor span in `component` that holds up, with its frames; and
    for every pair of consecutive timestamps (keyed by the first's file
    offset), `_TIMED` or why its frames can't be timed by it (a `_LOST_*`
    outcome).
    """
    outcomes: dict[int, str] = {}
    pairs: list[list[_AnchorSpan]] = []
    for before, after in pairwise(markers):
        if not (before.is_timestamp and after.is_timestamp):
            continue
        spans = _anchor_spans(component, before, after)
        if isinstance(spans, str):
            outcomes[before.start] = spans
        else:
            pairs.append(spans)

    per_block = [spans[0].duration_ms / spans[0].blocks for spans in pairs]
    if len(per_block) >= _MIN_SPANS_FOR_MEDIAN:
        median = statistics.median(per_block)
        for spans, duration in zip(list(pairs), per_block, strict=True):
            if abs(duration - median) > BLOCK_DURATION_TOLERANCE * median:
                outcomes[spans[0].before.start] = _LOST_DURATION
                pairs.remove(spans)

    timed: list[tuple[_AnchorSpan, list[_FrameCandidate]]] = []
    for spans in pairs:
        result = _frames_in_any_layout(component, spans)
        if isinstance(result, str):
            outcomes[spans[0].before.start] = result
        else:
            outcomes[spans[0].before.start] = _TIMED
            timed.append(result)
    return timed, outcomes


def _extrapolation_anchors(
    markers: list[_Marker], pair_outcomes: dict[int, str]
) -> Iterator[tuple[_Marker, int, bool]]:
    """(timestamp, direction, whether it's the recording's first block) for
    every block next to a timestamp that's bracketed by nothing else
    received: direction 1 for the block after the timestamp, -1 for the
    one before it. A block whose other timestamp was received but didn't
    make a span with this one (other than for too many timestamps missing
    in between) is left alone, as the timestamps themselves are suspect.
    """
    for i, marker in enumerate(markers):
        if not marker.is_timestamp:
            continue
        prev = markers[i - 1] if i > 0 else None
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        if nxt is None or (
            nxt.start > marker.end + MPI_BUFFER_LEN
            and pair_outcomes.get(marker.start) in _OPEN_PAIR_OUTCOMES
        ):
            yield marker, 1, False
        if prev is None or (
            prev.end < marker.start - MPI_BUFFER_LEN
            and pair_outcomes.get(prev.start) in _OPEN_PAIR_OUTCOMES
        ):
            yield marker, -1, False
        elif prev.is_header and prev.end == marker.start - MPI_BUFFER_LEN:
            yield marker, -1, True


def _extrapolation_region(anchor: _Marker, direction: int) -> tuple[int, int]:
    """The file offsets [start, end) of the block next to `anchor`."""
    if direction > 0:
        return anchor.end, anchor.end + MPI_BUFFER_LEN
    return anchor.start - MPI_BUFFER_LEN, anchor.start


def _frame_config(frame: bytes) -> tuple[int, int]:
    """(frame length, integration period): what sets the MPI's data rate."""
    return len(frame), int.from_bytes(frame[18:20], "big")


@dataclass(slots=True, frozen=True)
class _Timing:
    """How a run of frames' collection times are worked out: from where
    each one ends relative to an anchor timestamp, at a steady data rate.
    """

    method: str
    anchor: _Marker
    ms_per_byte: float
    sync_source: str
    span_blocks: int | None
    before_at: datetime | None
    after_at: datetime | None


def _span_timing(span: _AnchorSpan) -> _Timing:
    return _Timing(
        method=_INTERPOLATED,
        anchor=span.before,
        ms_per_byte=span.ms_per_byte,
        sync_source=span.sync_source,
        span_blocks=span.blocks,
        before_at=_epoch_ms_to_datetime(span.before.fields["timestamp_ms"]),
        after_at=_epoch_ms_to_datetime(span.after.fields["timestamp_ms"]),
    )


def _extrapolated_rows(
    component: _Component,
    anchor: _Marker,
    direction: int,
    ms_per_byte_by_config: dict[tuple[int, int], float],
) -> list[dict[str, Any]] | str:
    """Output rows for the frames in the block next to `anchor` (see
    `_extrapolation_anchors`), timed from it at the data rate measured
    across all anchor spans -- or why they can't be (a `_LOST_*` outcome).
    """
    sync_source = _synced_source(anchor)
    if sync_source is None:
        return _LOST_UNSYNCED
    if direction > 0:
        # Led by the previous block's tail, as in `_frames_in_span`.
        lead = _MAX_FRAME_LEN
        runs = [
            (anchor.start - lead, anchor.start),
            (anchor.end, anchor.end + MPI_BUFFER_LEN),
        ]
        region, anchor_pos = (lead, lead + MPI_BUFFER_LEN), lead
    else:
        # A frame straddling the block's start would have the previous
        # (never-received) timestamp wedged in it, so it isn't looked for.
        runs = [(anchor.start - MPI_BUFFER_LEN, anchor.start)]
        region, anchor_pos = (0, MPI_BUFFER_LEN), MPI_BUFFER_LEN
    candidates = _frames_in_region(
        component, runs, region=region, anchor_pos=anchor_pos
    )
    if candidates is None:
        return _LOST_INCONSISTENT
    if not candidates:
        return []
    ms_per_byte = ms_per_byte_by_config.get(_frame_config(candidates[0].frame))
    if ms_per_byte is None:
        return _LOST_NO_RATE
    anchor_at = _epoch_ms_to_datetime(anchor.fields["timestamp_ms"])
    timing = _Timing(
        method=_EXTRAPOLATED,
        anchor=anchor,
        ms_per_byte=ms_per_byte,
        sync_source=sync_source,
        span_blocks=None,
        before_at=anchor_at if direction > 0 else None,
        after_at=None if direction > 0 else anchor_at,
    )
    return _frame_rows(component, timing, candidates)


def _frame_rows(
    component: _Component, timing: _Timing, candidates: list[_FrameCandidate]
) -> list[dict[str, Any]]:
    """Output rows for `candidates`, timed by `timing`."""
    anchor_ts = _timestamp_fields(timing.anchor)
    assert anchor_ts is not None  # Checked when the timing was made.
    uptime, epoch_ms, _, since_sync = anchor_ts
    rows: list[dict[str, Any]] = []
    for c in candidates:
        offset_ms = c.end_from_anchor * timing.ms_per_byte
        covering = component.chunks_covering(c.first_offset, c.last_offset + 1)
        sightings = [s for chunk in covering for s in chunk.sightings]
        received = [s.received_at for s in sightings if s.received_at is not None]
        rows.append(
            {
                "collected_at": _epoch_ms_to_datetime(epoch_ms + offset_ms),
                "obc_uptime_ms": round(uptime + offset_ms),
                "ms_since_time_sync": round(since_sync + offset_ms),
                "time_sync_source": timing.sync_source,
                "timing_method": timing.method,
                "anchor_span_blocks": timing.span_blocks,
                "anchor_before_at": timing.before_at,
                "anchor_after_at": timing.after_at,
                **_parse_frame(c.frame),
                "file_offset": c.first_offset,
                "first_received_at": min(received, default=None),
                "last_received_at": max(received, default=None),
                "times_received": min(
                    len({s.packet_id for s in chunk.sightings}) for chunk in covering
                ),
                "packet_ids": json.dumps(sorted({s.packet_id for s in sightings})),
                "_frame": c.frame,
            }
        )
    return rows


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _epoch_ms_to_datetime(epoch_ms: float) -> datetime:
    return _EPOCH + timedelta(milliseconds=epoch_ms)


def compute_mpi_frames(decoded_packets: pl.DataFrame) -> pl.DataFrame:
    """Pure computation: `everything_decoded` -> one row per MPI frame
    recovered from bulk file downlinks, with its collection time.

    Args:
        decoded_packets: `everything_decoded.parquet`, as written by step 3.

    Returns:
        One row per frame, oldest first -- see the module docstring for how
        frames are recovered and timed, and `_OUTPUT_SCHEMA` for the
        columns.
    """
    return compute_mpi_frames_with_outcomes(decoded_packets)[0]


def compute_mpi_frames_with_outcomes(
    decoded_packets: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`compute_mpi_frames`, plus a table of what became of every distinct
    CRC-valid frame received: how many were kept (and how they were
    timed), and how many were dropped, for each reason.
    """
    if decoded_packets.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA), _outcome_table(Counter())
    rows, _tally, outcomes = _compute_rows(decoded_packets)
    return _rows_to_df(rows), _outcome_table(outcomes)


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    frames = pl.DataFrame(
        rows, schema_overrides=_OUTPUT_SCHEMA, infer_schema_length=None
    ).sort("collected_at", "file_offset")
    return assign_mpi_collections(frames).select(
        pl.col(name).cast(dtype) for name, dtype in _OUTPUT_SCHEMA.items()
    )


def assign_mpi_collections(frames: pl.DataFrame) -> pl.DataFrame:
    """`frames` (any table with `collected_at` and `frame_counter`), sorted
    by `collected_at`, with the MPI collection each frame is in, identified
    by the collection's first frame: `collection_started_at` and
    `collection_first_frame_counter`.

    A new collection starts at the first frame, and then at the first frame
    `MPI_COLLECTION_MAX_DURATION` or more after the current collection's
    first, or with a lower counter than the frame before it: a new
    recording (the counter restarts at 0 on each one), or the counter
    wrapping around. A drop of up to `_MAX_OUT_OF_ORDER_FRAMES` within
    `_MAX_OUT_OF_ORDER_TIME` doesn't count -- that's timing jitter, not a
    reset.
    """
    frames = frames.sort("collected_at", maintain_order=True)
    times = frames["collected_at"].to_list()
    counters = frames["frame_counter"].to_list()
    starts: list[int] = []
    start = 0
    for i in range(len(times)):
        drop = counters[i - 1] - counters[i] if i else 0
        if i and (
            times[i] - times[start] >= MPI_COLLECTION_MAX_DURATION
            or drop > _MAX_OUT_OF_ORDER_FRAMES
            or (drop > 0 and times[i] - times[i - 1] > _MAX_OUT_OF_ORDER_TIME)
        ):
            start = i
        starts.append(start)
    first = pl.Series(starts, dtype=pl.UInt32)
    return frames.with_columns(
        collection_started_at=frames["collected_at"].gather(first),
        collection_first_frame_counter=frames["frame_counter"].gather(first),
    )


# The per-frame-collection summary, before any merging with MPI analyses.
_FRAME_COLLECTIONS_SCHEMA = {
    "collection_started_at": pl.Datetime("us", "UTC"),
    "collection_ended_at": pl.Datetime("us", "UTC"),
    "frames_present": pl.Int64,
    # From the counters: every frame from the first present to the last.
    "frames_expected": pl.Int64,
    "frames_missing": pl.Int64,
    "first_frame_counter": pl.Int64,
    "last_frame_counter": pl.Int64,
    # The longest run of missing frames (by counter), when there is one: how
    # many frames, the time of the frame before it, and the time between
    # the frames either side of it.
    "largest_gap_frames": pl.Int64,
    "largest_gap_started_at": pl.Datetime("us", "UTC"),
    "largest_gap_duration_ms": pl.Int64,
}

# One `analyze_mpi_data` telecommand response, as `extract_all_mpi_analyses`
# pulls it out of step 5's output: the satellite's own summary of one MPI
# data file, from the full file on board.
_ANALYSES_SCHEMA = {
    "analysis_started_at": pl.Datetime("us", "UTC"),  # `earliest`
    "analysis_ended_at": pl.Datetime("us", "UTC"),  # `latest`
    "analysis_file": pl.String,
    "analysis_file_size": pl.Int64,
    "analysis_sha256": pl.String,
    "analysis_frame_count": pl.Int64,
    "analysis_valid_frame_count": pl.Int64,
    "analysis_bad_frame_count": pl.Int64,
    "analysis_background_frame_count": pl.Int64,
    "analysis_warmup_frame_count": pl.Int64,
    "analysis_first_frame_byte": pl.Int64,  # `frame_byte_range`
    "analysis_last_frame_byte": pl.Int64,
    "analysis_time_sync_count": pl.Int64,
    "analysis_mpi_start_count": pl.Int64,
    "analysis_malformed_time_sync_count": pl.Int64,
    "analysis_strong_threshold_dn": pl.Int64,
    "analysis_strong_frame_count": pl.Int64,
    "analysis_strong_signal_count": pl.Int64,
    "analysis_strong_signals": pl.String,  # JSON array, as sent
    "analysis_action": pl.String,
    # The `v<N>` at the end of `analysis_action`, if any.
    "analysis_version": pl.Int64,
    "analysis_tcmd_sent_at": pl.Datetime("us", "UTC"),
    "analysis_received_at": pl.Datetime("us", "UTC"),
}

# `mpi_analyses.parquet`: every distinct analysis, with the response itself
# (so nothing a future version adds is lost).
_ALL_ANALYSES_SCHEMA = {
    **_ANALYSES_SCHEMA,
    "analysis_response_text": pl.String,
}

# One file's primary analysis (see `primary_mpi_analyses`), plus what the
# file's other analyses add to it.
_FILE_ANALYSES_SCHEMA = {
    **_ANALYSES_SCHEMA,
    # How many distinct analyses of the file were received.
    "analysis_count": pl.Int64,
    # The strong signal count at every `strong_threshold_dn` the file was
    # analyzed with, as a JSON object of threshold -> count, e.g.
    # '{"100":5,"200":0}' -- each from the primary analysis at that
    # threshold (picked the same way as the file's primary analysis).
    "analysis_strong_signal_counts": pl.String,
}

_COLLECTIONS_SCHEMA = {
    **_FRAME_COLLECTIONS_SCHEMA,
    # "analysis" when the start/end are the MPI analysis's `earliest`/
    # `latest` (the timestamps in the file on board), "frames" when they're
    # just the first/last frame downlinked.
    "collection_time_source": pl.String,
    # The recording's duration: from `collection_started_at` to
    # `collection_ended_at`.
    "collection_duration_ms": pl.Int64,
    # How many frame collections (see `assign_mpi_collections`) were merged
    # into this one -- 0 for an analysis with no frames downlinked yet, more
    # than 1 when one file holds several recordings.
    "frame_collection_count": pl.Int64,
    **_FILE_ANALYSES_SCHEMA,
}

# The `action` of an `analyze_mpi_data` telecommand's response -- any
# version, e.g. "analyze_mpi_data_v1".
MPI_ANALYSIS_ACTION_PREFIX = "analyze_mpi_data"
_ANALYSIS_VERSION_RE = re.compile(r"_v(\d+)$")
# How far outside an MPI analysis's `earliest`..`latest` a frame collection
# can reach and still be that analysis's: frames are timed from the same
# timestamps, but may be extrapolated up to a block past them.
_ANALYSIS_MATCH_TOLERANCE = timedelta(seconds=_BLOCK_DURATION_RANGE_MS[1] // 1000)


def _int_field(value: object) -> int | None:
    # `bool` is an `int` subclass, but `"size": true` is not a size.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _analysis_time(value: object) -> datetime | None:
    """An analysis's `earliest`/`latest`: `{"timestamp_ms": ..., ...}`."""
    if not isinstance(value, dict):
        return None
    epoch_ms = _int_field(cast("dict[str, Any]", value).get("timestamp_ms"))
    if epoch_ms is None or not 0 < epoch_ms <= MAX_VALID_EPOCH_MS:
        return None
    return _epoch_ms_to_datetime(epoch_ms)


def _analysis_row(
    properties: dict[str, Any], tcmd_sent_at: datetime | None, received_at: datetime
) -> dict[str, Any] | None:
    started_at = _analysis_time(properties.get("earliest"))
    ended_at = _analysis_time(properties.get("latest"))
    if started_at is None or ended_at is None or ended_at < started_at:
        return None
    byte_range = properties.get("frame_byte_range")
    first_byte, last_byte = (
        (_int_field(byte_range[0]), _int_field(byte_range[1]))
        if isinstance(byte_range, list)
        and len(byte_range := cast("list[object]", byte_range)) == 2  # noqa: PLR2004
        else (None, None)
    )
    file = properties.get("file")
    sha256 = properties.get("sha256")
    strong_signals = properties.get("strong_signals")
    return {
        "analysis_started_at": started_at,
        "analysis_ended_at": ended_at,
        "analysis_file": file if isinstance(file, str) else None,
        "analysis_file_size": _int_field(properties.get("size")),
        "analysis_sha256": sha256.lower() if isinstance(sha256, str) else None,
        **{
            f"analysis_{key}": _int_field(properties.get(key))
            for key in (
                "frame_count",
                "valid_frame_count",
                "bad_frame_count",
                "background_frame_count",
                "warmup_frame_count",
                "time_sync_count",
                "mpi_start_count",
                "malformed_time_sync_count",
                "strong_threshold_dn",
                "strong_frame_count",
                "strong_signal_count",
            )
        },
        "analysis_first_frame_byte": first_byte,
        "analysis_last_frame_byte": last_byte,
        "analysis_strong_signals": (
            json.dumps(strong_signals) if isinstance(strong_signals, list) else None
        ),
        "analysis_action": properties["action"],
        "analysis_version": (
            int(m.group(1))
            if (m := _ANALYSIS_VERSION_RE.search(properties["action"]))
            else None
        ),
        "analysis_tcmd_sent_at": tcmd_sent_at,
        "analysis_received_at": received_at,
    }


def extract_all_mpi_analyses(tcmd_responses: pl.DataFrame) -> pl.DataFrame:
    """Every distinct successful `analyze_mpi_data` response in
    `tcmd_responses` (step 5's `reassembled_tcmd_responses.parquet`), oldest
    first -- see `_ALL_ANALYSES_SCHEMA`.

    Only complete responses (every part received), with response code 0,
    that parse as JSON and have a valid `earliest`/`latest` count. The same
    response received more than once (an identical re-downlink) is kept
    once, as first received.
    """
    if tcmd_responses.is_empty():
        return pl.DataFrame(schema=_ALL_ANALYSES_SCHEMA)
    rows: list[dict[str, Any]] = []
    for text, tcmd_sent_at, received_at in (
        tcmd_responses.filter(
            pl.col("is_complete")
            & (pl.col("tcmd_response_code") == 0)
            & pl.col("tcmd_response_text").str.contains(
                MPI_ANALYSIS_ACTION_PREFIX, literal=True
            )
        )
        .select("tcmd_response_text", "tcmd_sent_at", "first_received_at")
        .iter_rows()
    ):
        try:
            properties = json.loads(text)
        except ValueError:
            continue
        if not isinstance(properties, dict):
            continue
        properties = cast("dict[str, Any]", properties)
        action = properties.get("action")
        if not (
            isinstance(action, str) and action.startswith(MPI_ANALYSIS_ACTION_PREFIX)
        ):
            continue
        if row := _analysis_row(properties, tcmd_sent_at, received_at):
            rows.append({**row, "analysis_response_text": text})
    if not rows:
        return pl.DataFrame(schema=_ALL_ANALYSES_SCHEMA)
    return (
        pl.DataFrame(rows, schema=_ALL_ANALYSES_SCHEMA)
        .sort("analysis_received_at", nulls_last=True)
        .unique("analysis_response_text", keep="first", maintain_order=True)
        .sort("analysis_started_at", "analysis_received_at")
    )


def sort_mpi_analyses_by_preference(analyses: pl.DataFrame) -> pl.DataFrame:
    """`analyses`, the most-preferred first: the latest version (the `v<N>`
    in its `action`), then the largest (the file may have grown in
    between), then the earliest received.
    """
    return analyses.sort(
        "analysis_version",
        "analysis_file_size",
        "analysis_received_at",
        descending=[True, True, False],
        nulls_last=True,
    )


def primary_mpi_analyses(all_analyses: pl.DataFrame) -> pl.DataFrame:
    """One row per MPI data file in `all_analyses` (from
    `extract_all_mpi_analyses`), oldest first: its most-preferred analysis
    (see `sort_mpi_analyses_by_preference`), plus how many there were and
    the strong signal count at each threshold -- see
    `_FILE_ANALYSES_SCHEMA`. An analysis with no file name is its own file.
    """
    if all_analyses.is_empty():
        return pl.DataFrame(schema=_FILE_ANALYSES_SCHEMA)
    # A stand-in file name, so nameless analyses each stay on their own.
    file_key = pl.coalesce("analysis_file", pl.format("\\0{}", pl.int_range(pl.len())))
    analyses = sort_mpi_analyses_by_preference(
        all_analyses.with_columns(_file_key=file_key)
    )
    strong_counts = (
        analyses.filter(
            pl.col("analysis_strong_threshold_dn").is_not_null()
            & pl.col("analysis_strong_signal_count").is_not_null()
        )
        .unique(["_file_key", "analysis_strong_threshold_dn"], keep="first")
        .sort("analysis_strong_threshold_dn")
        .group_by("_file_key")
        .agg(
            analysis_strong_signal_counts=pl.concat_str(
                pl.lit("{"),
                pl.format(
                    '"{}":{}',
                    "analysis_strong_threshold_dn",
                    "analysis_strong_signal_count",
                ).str.join(","),
                pl.lit("}"),
            )
        )
    )
    return (
        analyses.with_columns(analysis_count=pl.len().over("_file_key"))
        .unique("_file_key", keep="first", maintain_order=True)
        .join(strong_counts, on="_file_key", how="left")
        .select(
            pl.col(name).cast(dtype) for name, dtype in _FILE_ANALYSES_SCHEMA.items()
        )
        .sort("analysis_started_at")
    )


def extract_mpi_analyses(tcmd_responses: pl.DataFrame) -> pl.DataFrame:
    """`primary_mpi_analyses` straight from step 5's output."""
    return primary_mpi_analyses(extract_all_mpi_analyses(tcmd_responses))


def _summarize_frame_collections(frames: pl.DataFrame) -> pl.DataFrame:
    """One row per frame collection in `frames` (as labelled by
    `assign_mpi_collections`), oldest first -- see
    `_FRAME_COLLECTIONS_SCHEMA`.
    """
    if frames.is_empty():
        return pl.DataFrame(schema=_FRAME_COLLECTIONS_SCHEMA)
    collection = "collection_started_at"
    gap_frames = pl.col("frame_counter").diff().over(collection) - 1
    has_gap = gap_frames > 0
    gaps = frames.sort(collection, "frame_counter").with_columns(
        gap_frames=pl.when(has_gap).then(gap_frames),
        gap_started_at=pl.when(has_gap).then(
            pl.col("collected_at").shift().over(collection)
        ),
        gap_ended_at=pl.when(has_gap).then(pl.col("collected_at")),
    )

    def at_largest_gap(column: str) -> pl.Expr:
        return (
            pl.col(column)
            .sort_by("gap_frames", descending=True, nulls_last=True)
            .first()
        )

    return (
        gaps.group_by(collection)
        .agg(
            collection_ended_at=pl.col("collected_at").max(),
            frames_present=pl.len(),
            first_frame_counter=pl.col("frame_counter").min(),
            last_frame_counter=pl.col("frame_counter").max(),
            largest_gap_frames=pl.col("gap_frames").max().fill_null(0),
            largest_gap_started_at=at_largest_gap("gap_started_at"),
            largest_gap_ended_at=at_largest_gap("gap_ended_at"),
        )
        .with_columns(
            frames_expected=pl.col("last_frame_counter")
            - pl.col("first_frame_counter")
            + 1,
            largest_gap_duration_ms=(
                pl.col("largest_gap_ended_at") - pl.col("largest_gap_started_at")
            ).dt.total_milliseconds(),
        )
        .with_columns(
            frames_missing=pl.col("frames_expected") - pl.col("frames_present")
        )
        .select(
            pl.col(name).cast(dtype)
            for name, dtype in _FRAME_COLLECTIONS_SCHEMA.items()
        )
        .sort(collection)
    )


def _matching_analysis(
    collection: dict[str, Any], analyses: list[dict[str, Any]]
) -> int | None:
    """The index of the analysis (in `analyses`) whose `earliest`..`latest`
    the frame collection overlaps, if any -- the most-overlapped one, should
    several.
    """
    best: tuple[timedelta, int] | None = None
    for i, analysis in enumerate(analyses):
        overlap = min(
            collection["collection_ended_at"],
            analysis["analysis_ended_at"] + _ANALYSIS_MATCH_TOLERANCE,
        ) - max(
            collection["collection_started_at"],
            analysis["analysis_started_at"] - _ANALYSIS_MATCH_TOLERANCE,
        )
        if overlap >= timedelta(0) and (best is None or overlap > best[0]):
            best = (overlap, i)
    return None if best is None else best[1]


def _merged_collection(
    analysis: dict[str, Any], collections: list[dict[str, Any]]
) -> dict[str, Any]:
    """One analysis and every frame collection in it (oldest first) as one
    collection, timed by the analysis.
    """
    row: dict[str, Any] = {
        "collection_started_at": analysis["analysis_started_at"],
        "collection_ended_at": analysis["analysis_ended_at"],
        "collection_time_source": "analysis",
        "frame_collection_count": len(collections),
        **analysis,
    }
    if not collections:
        return {**row, "frames_present": 0, "frames_missing": None}
    largest_gap = max(collections, key=lambda c: c["largest_gap_frames"])
    return {
        **row,
        **{
            key: sum(c[key] for c in collections)
            for key in ("frames_present", "frames_expected", "frames_missing")
        },
        "first_frame_counter": collections[0]["first_frame_counter"],
        "last_frame_counter": collections[-1]["last_frame_counter"],
        **{
            key: largest_gap[key]
            for key in (
                "largest_gap_frames",
                "largest_gap_started_at",
                "largest_gap_duration_ms",
            )
        },
    }


def summarize_mpi_collections(
    frames: pl.DataFrame, analyses: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per MPI collection, oldest first -- see `_COLLECTIONS_SCHEMA`.

    Each frame collection in `frames` (as labelled by
    `assign_mpi_collections`) that overlaps an MPI analysis in `analyses`
    (from `extract_mpi_analyses`) is merged into it: its start/end become
    the analysis's `earliest`/`latest` -- the timestamps in the whole file
    on board, rather than just whichever frames were downlinked -- and its
    file metadata comes along. Every analysis is a collection, even with no
    frames downlinked yet; every other frame collection stays as it was.
    """
    frame_collections = _summarize_frame_collections(frames)
    analysis_rows = [] if analyses is None else analyses.to_dicts()
    merged: list[list[dict[str, Any]]] = [[] for _ in analysis_rows]
    rows: list[dict[str, Any]] = []
    for collection in frame_collections.iter_rows(named=True):
        i = _matching_analysis(collection, analysis_rows)
        if i is None:
            rows.append(
                {
                    **collection,
                    "collection_time_source": "frames",
                    "frame_collection_count": 1,
                }
            )
        else:
            merged[i].append(collection)
    rows += [
        _merged_collection(analysis, collections)
        for analysis, collections in zip(analysis_rows, merged, strict=True)
    ]
    if not rows:
        return pl.DataFrame(schema=_COLLECTIONS_SCHEMA)
    return (
        pl.DataFrame(rows, schema=_COLLECTIONS_SCHEMA)
        .with_columns(
            collection_duration_ms=(
                pl.col("collection_ended_at") - pl.col("collection_started_at")
            ).dt.total_milliseconds()
        )
        .sort("collection_started_at")
    )


def _outcome_table(outcomes: Counter[str]) -> pl.DataFrame:
    """One row per outcome: kept ones first, then the biggest losses."""
    ordered = sorted(
        outcomes.items(),
        key=lambda item: (not item[0].startswith("kept"), -item[1], item[0]),
    )
    return pl.DataFrame(
        {
            "outcome": [outcome for outcome, _ in ordered],
            "frames": [frames for _, frames in ordered],
        },
        schema={"outcome": pl.String, "frames": pl.Int64},
    ).with_columns(percent=(pl.col("frames") * 100 / pl.col("frames").sum()).round(1))


def _received_frames(component: _Component, markers: list[_Marker]) -> dict[bytes, int]:
    """Every fully-received, CRC-valid frame anywhere in `component`, timed
    or not (with its fully-received JSON markers carved out), and the file
    offset of its last byte.
    """
    runs: list[tuple[int, int]] = []
    cursor = component.base
    for m in markers:
        runs.append((cursor, m.start))
        cursor = m.end
    runs.append((cursor, component.base + len(component.data)))
    stream = _stream(component, runs)
    frames: dict[bytes, int] = {}
    for m in re.finditer(re.escape(MPI_SYNC), stream.data):
        start = m.start()
        frame_len = _frame_len_at(stream.data, start)
        if frame_len is None or start + frame_len > len(stream.data):
            continue
        if not all(stream.have[start : start + frame_len]):
            continue
        frame = bytes(stream.data[start : start + frame_len])
        if _frame_crc_ok(frame):
            frames.setdefault(frame, stream.file_offset(start + frame_len - 1))
    return frames


def _keep_best(best: dict[bytes, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Add `rows` to `best` (frame bytes -> row), keeping one row per frame:
    interpolated over extrapolated, then the narrowest span, then earliest.
    """

    def preference(row: dict[str, Any]) -> tuple[bool, int, datetime]:
        return (
            row["timing_method"] != _INTERPOLATED,
            row["anchor_span_blocks"] or 0,
            row["collected_at"],
        )

    for row in rows:
        frame = row.pop("_frame")
        current = best.get(frame)
        if current is None or preference(row) < preference(current):
            best[frame] = row


def _measured_ms_per_byte(
    rates: dict[tuple[int, int], list[float]],
) -> dict[tuple[int, int], float]:
    """The median MPI data rate over all anchor spans, per frame config.
    (A single span's rate is off by up to a few % from timestamp jitter.)
    """
    measured: dict[tuple[int, int], float] = {}
    for (frame_len, integration_period), span_rates in sorted(rates.items()):
        measured[frame_len, integration_period] = statistics.median(span_rates)
        logger.info(
            f"Measured MPI frame period: "
            f"{measured[frame_len, integration_period] * frame_len:.2f} ms "
            f"(median of {len(span_rates):,} anchor spans), for {frame_len}-byte "
            f"frames at integration period {integration_period}."
        )
    return measured


def _compute_rows(
    decoded_packets: pl.DataFrame,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    """(output rows, link/anchor stats, frames per outcome)."""
    chunks = _load_chunks(decoded_packets)
    if not chunks:
        return [], Counter(), Counter()
    groups, tally = _link_chunks(chunks)
    tally["chunks"] = len(chunks)
    tally["components"] = len(groups)
    components = [_assemble([chunks[i] for i in group]) for group in groups]
    markers = [_find_markers(component) for component in components]

    # Interpolate between pairs of timestamps first: the data rate they
    # measure is what frames near a lone timestamp are timed by.
    best: dict[bytes, dict[str, Any]] = {}
    rates: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_outcomes: list[dict[int, str]] = []
    for component, component_markers in zip(components, markers, strict=True):
        tally["conflicting_bytes"] += component.conflicting_bytes
        timed, outcomes = _timed_spans(component, component_markers)
        pair_outcomes.append(outcomes)
        for span, candidates in timed:
            tally["spans"] += 1
            if candidates:
                rates[_frame_config(candidates[0].frame)].append(span.ms_per_byte)
            _keep_best(best, _frame_rows(component, _span_timing(span), candidates))
    ms_per_byte_by_config = _measured_ms_per_byte(rates)

    lost: dict[bytes, str] = {}
    for component, component_markers, outcomes in zip(
        components, markers, pair_outcomes, strict=True
    ):
        # File offset ranges [start, end) a lost frame's last byte can fall
        # in, with why it was lost -- the last one that matches wins.
        why_lost = [
            (before.end, after.start, outcomes[before.start])
            for before, after in pairwise(component_markers)
            if outcomes.get(before.start, _TIMED) != _TIMED
        ]
        for anchor, direction, first_block in _extrapolation_anchors(
            component_markers, outcomes
        ):
            result = _extrapolated_rows(
                component, anchor, direction, ms_per_byte_by_config
            )
            if isinstance(result, list):
                tally["extrapolated_blocks"] += 1
                _keep_best(best, result)
                continue
            if first_block and result == _LOST_INCONSISTENT:
                result = _LOST_STALE_FIRST_BLOCK
            why_lost.append((*_extrapolation_region(anchor, direction), result))

        default = (
            _LOST_TOO_FAR
            if any(m.is_timestamp for m in component_markers)
            else _LOST_NO_TIMESTAMP
        )
        for frame, last_offset in _received_frames(
            component, component_markers
        ).items():
            reasons = [r for lo, hi, r in why_lost if lo <= last_offset < hi]
            lost.setdefault(frame, reasons[-1] if reasons else default)

    outcomes_count: Counter[str] = Counter(
        _KEPT_INTERPOLATED
        if row["timing_method"] == _INTERPOLATED
        else _KEPT_EXTRAPOLATED
        for row in best.values()
    )
    outcomes_count.update(reason for frame, reason in lost.items() if frame not in best)
    # A frame straddling a never-received timestamp is only found once
    # that timestamp is carved out by position, so `best` can have frames
    # `lost` doesn't.
    tally["frames_received"] = outcomes_count.total()
    return list(best.values()), tally, outcomes_count


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `mpi_frames.parquet` and `mpi_collections.parquet` from
    step 3's (and step 5's) output.

    Reads `everything_decoded.parquet` and `reassembled_tcmd_responses.parquet`
    from `data_dir` (where steps 3 and 5 write them) and writes the result
    back into the same directory --
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
    rows, tally, outcomes = _compute_rows(bulk_packets)
    result_df = _rows_to_df(rows)

    tcmd_path = data_dir / step_5_pipeline.OUTPUT_FILENAME
    if tcmd_path.exists():
        all_analyses_df = extract_all_mpi_analyses(pl.read_parquet(tcmd_path))
    else:
        logger.warning(f"{tcmd_path} not found -- no MPI analyses (run step_5).")
        all_analyses_df = pl.DataFrame(schema=_ALL_ANALYSES_SCHEMA)
    analyses_df = primary_mpi_analyses(all_analyses_df)
    logger.info(
        f"{all_analyses_df.height:,} distinct MPI analysis response(s) found, "
        f"of {analyses_df.height:,} file(s)."
    )

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result_df, out_path)
    _write_parquet_atomic(all_analyses_df, data_dir / ANALYSES_OUTPUT_FILENAME)
    collections_df = summarize_mpi_collections(result_df, analyses_df)
    _write_parquet_atomic(collections_df, data_dir / COLLECTIONS_OUTPUT_FILENAME)
    logger.info(f"MPI chunk linking/anchoring stats: {dict(sorted(tally.items()))}")
    received = tally["frames_received"]
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        fmt_str_lengths=100,
        tbl_width_chars=200,
    ):
        logger.info(
            f"What became of the {received:,} distinct CRC-valid MPI frames "
            f"received:\n{_outcome_table(outcomes)}"
        )
    utilized = f"{result_df.height / received:.1%}" if received else "n/a"
    logger.info(
        f"Done. {result_df.height:,} MPI frame(s) written to {out_path}, from "
        f"{tally['spans']:,} anchor span(s) and {tally['extrapolated_blocks']:,} "
        f"block(s) extrapolated from one timestamp. Quality: "
        f"{result_df.height:,} / {received:,} = {utilized} of the distinct "
        f"CRC-valid MPI frames received were timed and kept."
    )
