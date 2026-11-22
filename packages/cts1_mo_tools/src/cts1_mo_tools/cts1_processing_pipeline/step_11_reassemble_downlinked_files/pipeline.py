"""Step 11: work out which file every bulk downlink chunk is of, and
reassemble each file.

A `BULK_FILE_DOWNLINK` packet carries a file offset and up to 195 bytes,
but nothing naming the file. Everything here is placed on one timeline,
when it was sent by the satellite's clock (`sent_at`): a packet's
`transmitted_at_sat_clock` from step 2 if it has one, else its
`received_at` moved by the offset between the two around it
(`CLOCK_OFFSET_WINDOW`) -- the recordings' own times are often tens of
seconds apart. Two things around a chunk name its file:

  - Headers: the telecommand that starts a downlink
    (`bulk_downlink_start*`, `adcs_get_latest_sd_file_blob*`) answers with
    the file's path, size, SHA-256 and the `offset`/`length` it's about to
    send. The satellite sends that range at about `DOWNLINK_BYTES_PER_SEC`
    from then, so a header only vouches for a chunk in its range sent when
    the downlink would have got to that offset (`DOWNLINK_LEAD`,
    `DOWNLINK_SLACK`, `DOWNLINK_RATE_SLACK`) -- by either clock: `sent_at`,
    or `received_at` against the header's. A new header ends the one
    before (starting a downlink closes the last), but its last few chunks
    can still go out just after it.
  - Beacons: every v5+ beacon carries the first 3 bytes of SHA-256 of the
    path being downlinked when it was stamped
    (`bulk_downlink_file_path_sha256_prefix`), or nothing when no downlink
    is running. Stored ones too, replayed from the satellite's store long
    after: each is placed at its own stamp (plus the time a beacon takes to
    go out), as are live ones. Step 10's `satellite_files.parquet` maps the
    prefix back to a path.

Each chunk is attributed, in order, by the header it fits (`header`); else
the latest beacon within `BEACON_CONTEXT` naming a file, if the chunk's
downlink could already have started by then (by its offset) and no header
came between (`beacon`); else the next one naming a file within
`BEACON_CONTEXT`, if no header comes first (`next_beacon`) -- a downlink
whose header was never heard. Else, if no beacon around names a file (as
before the v5 beacons, 2026-11-03), the latest header still vouches for
chunks in its range until its downlink would have finished at
`MIN_DOWNLINK_BYTES_PER_SEC`, plus `HEADER_MARGIN` (`header`): the same file
sent again by a time-tagged telecommand whose header nobody heard. A beacon
saying nothing is downlinking doesn't stop that: seen in flight while a
file's chunks were coming down. Otherwise the chunk is `unattributed`.

Consecutive chunks of one file, with no gap over `DOWNLINK_GAP`, no new
header in between, and starting within `DOWNLINK_RESTART` of each other
(going by their offsets), are one downlink
(`file_downlink_time_periods.parquet`).

A header doesn't guarantee what follows it is that file: before the
`bulk_downlink_start_blob` fix (firmware issue #653), starting a downlink
didn't always close the file already open, so chunks of the previous file
could follow a new header -- seen in flight, e.g. `ADCS/log_180b.TLM`'s
bytes downlinked under `ADCS/log_1228.TLM`'s header. So a file's downlinks
are grouped into clusters that agree on every byte they overlap (to within
`CONSISTENCY_TOLERANCE_BYTES`), and only one cluster is kept: the one whose
reassembly matches a SHA-256 the satellite reported, else one not carrying
another verified file's bytes, else the one covering the most bytes. The
other downlinks are rejected (`rejected_reason`, naming the file they carry
where known) and left out of the file.

Then the gaps left in each file are filled, where possible, from the
unattributed chunks received within `RESCUE_WINDOW` of one of its kept
downlinks (`rescue_nearby_chunks`; `time_window` attribution) -- e.g. the
last chunks of a downlink that ran past the header's expected end, or the
start of one whose first beacon came late. A chunk near more than one
file's downlinks is left alone. Candidates are taken per unattributed
downlink, nearest first, and only if every chunk fits the file's known
size, agrees byte for byte with whatever the file already has (or has
been given), and together they fill at least one missing byte. If that
completes a file with a SHA-256 on record but the hash doesn't match,
none of it is taken.

A candidate with a header sent between it and the downlink is from
a different downlink -- seen in flight right before nearly every header:
the tail of whatever was downlinking before -- as is one outside the
byte range the file's own header said it was sending. The header says
nothing about those, so they're only taken if they complete the file
with a SHA-256 the satellite reported (often they're an earlier pass of
the same file, whose own header was missed). On the data so far, the
rescue verifies ten more files, and completes none with the wrong hash.

All the kept chunks of a file -- over every downlink of it -- are
reassembled together
(`file_reassembly.reassemble_bulk_chunks`, judging every byte good, missing
or conflicting), padded with missing bytes out to its known size, and
checked against every SHA-256 the satellite reported for it
(`downlinked_files.parquet`, with the bytes themselves). Chunks of no
named file -- a prefix matching no known path, or unattributed and not
rescued -- are left out of both outputs. Every file step 10 knows of but that
was never downlinked gets a row too (no bytes), and every row carries
step 10's listing and mention columns -- so it's the one list of files.

Only CRC-valid packets are used. Reads steps 3, 5 (via 10) and 10's
outputs; parquet-in-parquet-out, recomputed from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "BEACON_CONTEXT",
    "CLOCK_OFFSET_WINDOW",
    "CONSISTENCY_TOLERANCE_BYTES",
    "DEFAULT_DATA_DIR",
    "DOWNLINKED_FILES_OUTPUT_FILENAME",
    "DOWNLINK_BYTES_PER_SEC",
    "DOWNLINK_GAP",
    "DOWNLINK_LEAD",
    "DOWNLINK_RATE_SLACK",
    "DOWNLINK_RESTART",
    "DOWNLINK_SLACK",
    "DOWNLINK_TIME_PERIODS_OUTPUT_FILENAME",
    "HEADER_MARGIN",
    "MIN_DOWNLINK_BYTES_PER_SEC",
    "RESCUE_WINDOW",
    "Attribution",
    "attribute_chunks",
    "decode_segments",
    "encode_segments",
    "reassemble_downlinked_files",
    "reject_inconsistent_downlinks",
    "rescue_nearby_chunks",
    "run",
    "summarize_downlinks",
]

import hashlib
import string
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import orjson
import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS
from cts1_mo_tools.cts1_picam_to_jpg import parse_picam_ascii_to_jpg_bytes
from cts1_mo_tools.cts1_processing_pipeline.file_reassembly import (
    MAX_REASSEMBLY_SPAN_BYTES,
    ByteSegment,
    ByteStatus,
    DetectedImage,
    detect_image,
    reassemble_bulk_chunks,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    sat_clock,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_10_list_satellite_files import (
    pipeline as step_10_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_10_pipeline.DEFAULT_DATA_DIR
DOWNLINK_TIME_PERIODS_OUTPUT_FILENAME = "file_downlink_time_periods.parquet"
DOWNLINKED_FILES_OUTPUT_FILENAME = "downlinked_files.parquet"

# The satellite sends 195 B of file every 208 ms (firmware
# `COMMS_bulk_downlink_delay_per_packet_ms`), which with the beacons and
# responses sharing the radio comes to about this (measured: 750-850 B/s).
DOWNLINK_BYTES_PER_SEC = 780
# A header's chunk at `offset` is sent `(offset - its offset) /
# DOWNLINK_BYTES_PER_SEC` after it, give or take: up to `DOWNLINK_LEAD`
# sooner (the first chunks can beat the header out), or later by up to
# `DOWNLINK_SLACK` plus `DOWNLINK_RATE_SLACK` of that (other packets slow it
# down). Measured against headers: 98% within -2 s to +4 s.
DOWNLINK_LEAD = timedelta(seconds=3)
DOWNLINK_SLACK = timedelta(seconds=5)
DOWNLINK_RATE_SLACK = 0.25
# With no beacon around to say otherwise (as before the v5 beacons), a
# header also vouches for chunks in its byte range until its downlink would
# have finished at `MIN_DOWNLINK_BYTES_PER_SEC`, plus `HEADER_MARGIN`: the
# same file sent again by a telecommand whose header was never heard.
MIN_DOWNLINK_BYTES_PER_SEC = 500
HEADER_MARGIN = timedelta(minutes=2)
# Live beacons come every ~10 s during a pass; this spans a few lost ones.
# Stored ones fill in between.
BEACON_CONTEXT = timedelta(minutes=2)
# Packets with no `transmitted_at_sat_clock` are moved onto the satellite's
# clock by the median offset between it and `received_at` around them.
CLOCK_OFFSET_WINDOW = timedelta(minutes=10)
_BEACON_STAMP_TO_TRANSMIT_MS = sat_clock.BEACON_STAMP_TO_TRANSMIT_MS[
    "BEACON_EXTENDED_V5PLUS"
]
# A longer gap between two chunks of a file starts a new downlink, as does
# their downlinks' start times (by their offsets) being further apart than
# `DOWNLINK_RESTART` -- the same file started again (every 30 s, say)
# whose header wasn't heard, or another file's.
DOWNLINK_GAP = timedelta(minutes=2)
DOWNLINK_RESTART = timedelta(seconds=15)
# How far either side of a file's downlinks `rescue_nearby_chunks` looks for
# unattributed chunks to fill its gaps. Measured on the data so far: 2 min
# verifies 6 more files, 5 min 10, 15 min 11 -- while the chunks near more
# than one file (so left alone) grow from 38 to 268 to 352.
RESCUE_WINDOW = timedelta(minutes=5)

# Two downlinks of a file disagreeing on more bytes than this (where both
# have them) are of different content -- see the module docstring. A few
# bytes' slack keeps a rare corrupted-but-CRC-valid chunk from splitting a
# file's downlinks apart.
CONSISTENCY_TOLERANCE_BYTES = 16
# A rejected downlink "carries" another file if at least this fraction of
# its chunks match that file's verified bytes at the same offsets.
CARRIED_FILE_MIN_FRACTION = 0.5

# How much of a file's received bytes `_content_kind` looks at.
_TEXT_SNIFF_BYTES = 64 * 1024
_TEXT_MIN_PRINTABLE_FRACTION = 0.95
_PRINTABLE_BYTES = frozenset(string.printable.encode())

UNIDENTIFIED_KEY_PREFIX = "unidentified:"
PREFIX_KEY_PREFIX = "sha256-prefix:"


class Attribution(StrEnum):
    """How a chunk was tied to a file -- see the module docstring."""

    HEADER = "header"
    BEACON = "beacon"
    NEXT_BEACON = "next_beacon"
    UNATTRIBUTED = "unattributed"
    # Unattributed, then taken into a file's gaps by `rescue_nearby_chunks`.
    TIME_WINDOW = "time_window"


_SEGMENT_STATUS_CODES = {
    ByteStatus.GOOD: "G",
    ByteStatus.MISSING: "M",
    ByteStatus.CONFLICTING: "C",
}
_SEGMENT_STATUSES = {code: status for status, code in _SEGMENT_STATUS_CODES.items()}

_DOWNLINKS_SCHEMA = {
    "file_key": pl.String,
    "file_path": pl.String,
    "path_sha256_prefix": pl.String,
    "attributions": pl.String,  # JSON array of `Attribution`s
    "started_at": pl.Datetime("us", "UTC"),
    "ended_at": pl.Datetime("us", "UTC"),
    "chunk_count": pl.Int64,
    "distinct_offset_count": pl.Int64,
    "offset_min": pl.Int64,
    "offset_end": pl.Int64,  # the highest offset + length received
    "bytes_received": pl.Int64,  # distinct bytes covered
    "header_received_at": pl.Datetime("us", "UTC"),
    "header_action": pl.String,
    "header_offset": pl.Int64,
    "header_length": pl.Int64,
    "header_file_size": pl.Int64,
    "header_sha256": pl.String,
    # Null if kept; else why it was left out of the file -- see the module
    # docstring.
    "rejected_reason": pl.String,
}

_FILES_SCHEMA = {
    "file_key": pl.String,
    "file_path": pl.String,
    "path_sha256_prefix": pl.String,
    "file_size": pl.Int64,  # as last reported by the satellite, if ever
    "byte_length": pl.Int64,  # of `data`
    "good_bytes": pl.Int64,
    "missing_bytes": pl.Int64,
    "conflict_bytes": pl.Int64,
    "received_fraction": pl.Float64,  # (good + conflicting) / byte_length
    "is_complete": pl.Boolean,
    "sha256": pl.String,  # of `data`, as reassembled
    "expected_sha256s": pl.String,  # JSON array, from the satellite
    # True: complete and matches one of `expected_sha256s`; False: complete
    # and matches none; null: incomplete, or no hash to check against.
    "sha256_verified": pl.Boolean,
    "first_downlinked_at": pl.Datetime("us", "UTC"),
    "last_downlinked_at": pl.Datetime("us", "UTC"),
    "downlink_count": pl.Int64,  # kept ones
    "rejected_downlink_count": pl.Int64,
    "chunk_count": pl.Int64,  # kept ones
    # Of `chunk_count`, how many `rescue_nearby_chunks` filled gaps with.
    "rescued_chunk_count": pl.Int64,
    # JSON array of [start, end, status, min_copies, max_copies] runs
    # tiling `data` -- status "G"ood, "M"issing or "C"onflicting.
    "segments": pl.String,
    "content_kind": pl.String,  # "image", "text" or "binary"
    "preview_media_type": pl.String,
    "preview_label": pl.String,
    # A preview image converted from the data (a PiCAM image's JPG); null
    # where the data is the image itself, or isn't one.
    "preview_image": pl.Binary,
    "data": pl.Binary,  # missing bytes are zeros; null if never downlinked
    # Step 10's `satellite_files` columns -- null for a file it doesn't know.
    "reported_sha256": pl.String,  # its `sha256`: the latest one reported
    "listing_status": pl.String,
    "sources": pl.String,
    "mention_count": pl.Int64,
    "first_seen_at": pl.Datetime("us", "UTC"),
    "last_seen_at": pl.Datetime("us", "UTC"),
    "last_listed_at": pl.Datetime("us", "UTC"),
}

# Step 10's columns carried into `_FILES_SCHEMA`, as named there.
_SATELLITE_FILE_COLUMNS = {
    "sha256": "reported_sha256",
    "listing_status": "listing_status",
    "sources": "sources",
    "mention_count": "mention_count",
    "first_seen_at": "first_seen_at",
    "last_seen_at": "last_seen_at",
    "last_listed_at": "last_listed_at",
}

_CHUNK_COLUMNS = (
    "packet_id",
    "received_at",
    "bulk_file_offset",
    "bulk_data_len",
    "bulk_data_hex",
)


def _headers(mentions: pl.DataFrame) -> pl.DataFrame:
    """Every downlink header in step 10's mentions, oldest first by when it
    was sent (`h_at`: `sent_at` if `mentions` has it, else `seen_at`).
    """
    sources = [s.value for s in step_10_pipeline.DOWNLINK_HEADER_SOURCES]
    sent_at = "sent_at" if "sent_at" in mentions.columns else "seen_at"
    return (
        mentions.filter(
            pl.col("source").is_in(sources) & pl.col("seen_at").is_not_null()
        )
        .select(
            h_at=pl.col(sent_at),
            h_received_at=pl.col("seen_at"),
            h_path=pl.col("file_path"),
            h_prefix=pl.col("raw_path").map_elements(
                step_10_pipeline.path_sha256_prefix, return_dtype=pl.String
            ),
            h_action=pl.col("action"),
            h_offset=pl.col("downlink_offset").fill_null(0),
            h_length=pl.col("downlink_length"),
            h_file_size=pl.col("file_size"),
            h_sha256=pl.col("sha256"),
        )
        .sort("h_at")
    )


def _prefix_paths(files: pl.DataFrame) -> pl.DataFrame:
    """Path SHA-256 prefix -> path, from step 10's `satellite_files`. On
    the (24-bit) chance two paths share a prefix, the one mentioned last
    wins, with a warning.
    """
    if files.is_empty():
        return pl.DataFrame(schema={"prefix": pl.String, "prefix_path": pl.String})
    exploded = (
        files.select(
            prefix=pl.col("path_sha256_prefixes").str.json_decode(pl.List(pl.String)),
            prefix_path=pl.col("file_path"),
            last_seen_at=pl.col("last_seen_at"),
        )
        .explode("prefix", empty_as_null=True)
        .drop_nulls("prefix")
    )
    collisions = (
        exploded.group_by("prefix")
        .agg(pl.col("prefix_path"))
        .filter(pl.col("prefix_path").list.len() > 1)
    )
    for prefix, paths in collisions.iter_rows():
        logger.warning(f"Path SHA-256 prefix {prefix} is shared by {paths}.")
    return (
        exploded.sort("last_seen_at", descending=True, nulls_last=True)
        .unique("prefix", keep="first")
        .select("prefix", "prefix_path")
    )


def _microseconds(duration: timedelta) -> int:
    return duration // timedelta(microseconds=1)


def _header_fits(h: str) -> pl.Expr:
    """Whether each chunk is where header `h` (its `{h}_*` columns) would
    have put it: in its byte range, and sent when its downlink got to it --
    see the module docstring. Judged on both clocks, as either can be off: a
    packet's `sent_at` (merged with a copy sent 30 s later, say), or its
    `received_at` (the recordings disagreeing).
    """
    offset = pl.col("bulk_file_offset")
    start, length = pl.col(f"{h}_offset"), pl.col(f"{h}_length")
    due_us = (offset - start) * (1_000_000 / DOWNLINK_BYTES_PER_SEC)

    def on_time(chunk_at: str, header_at: str) -> pl.Expr:
        late_us = (pl.col(chunk_at) - pl.col(header_at)).dt.total_microseconds()
        late_us -= due_us
        return (late_us >= -_microseconds(DOWNLINK_LEAD)) & (
            late_us <= _microseconds(DOWNLINK_SLACK) + DOWNLINK_RATE_SLACK * due_us
        )

    return (
        (offset >= start)
        & (length.is_null() | (offset < start + length))
        & (
            on_time("sent_at", f"{h}_at").fill_null(value=False)
            | on_time("received_at", f"{h}_received_at").fill_null(value=False)
        )
    ).fill_null(value=False)


def attribute_chunks(
    chunks: pl.DataFrame,
    beacons: pl.DataFrame,
    mentions: pl.DataFrame,
    files: pl.DataFrame,
) -> pl.DataFrame:
    """Tie every chunk to the file it's of -- see the module docstring.

    Args:
        chunks: CRC-valid `BULK_FILE_DOWNLINK` rows of `everything_decoded`
            (`_CHUNK_COLUMNS`), plus `sent_at` (else `received_at` is used).
        beacons: CRC-valid `BEACON_EXTENDED_V5PLUS` rows, live or stored:
            `sent_at` (else `received_at`) and
            `bulk_downlink_file_path_sha256_prefix`.
        mentions: step 10's `satellite_file_mentions`, plus `sent_at` for
            when each was sent (else `seen_at` is used).
        files: step 10's `satellite_files`.

    Returns:
        `chunks`, oldest sent first, plus `attribution`, `path_sha256_prefix`,
        `file_path` (null where the prefix matches no known path), the
        header's columns (`h_*`: the one attributing the chunk, else the
        latest), and `file_key` and `downlink_index` -- which file and which
        downlink of it each chunk is part of.
    """
    if "sent_at" not in chunks.columns:
        chunks = chunks.with_columns(sent_at=pl.col("received_at"))
    if "sent_at" not in beacons.columns:
        beacons = beacons.with_columns(sent_at=pl.col("received_at"))
    chunks = chunks.sort("sent_at", "bulk_file_offset")
    headers = _headers(mentions).with_row_index("_h")
    header_columns = [c for c in headers.columns if c.startswith("h_")]
    previous_headers = headers.select(
        (pl.col("_h") + 1).alias("_h"),
        *(pl.col(c).alias(f"p{c}") for c in header_columns),
    )
    beacon_times = beacons.select(
        b_at=pl.col("sent_at"),
        b_prefix=pl.col("bulk_downlink_file_path_sha256_prefix"),
    ).sort("b_at")

    # The latest header sent by `DOWNLINK_LEAD` after the chunk, the one
    # before it, and the first one after that.
    lead = pl.col("sent_at") + DOWNLINK_LEAD
    joined = (
        chunks.with_columns(_lead=lead)
        .join_asof(headers, left_on="_lead", right_on="h_at", strategy="backward")
        .join(previous_headers, on="_h", how="left")
        .join_asof(
            headers.select(hn_at=pl.col("h_at")),
            left_on="_lead",
            right_on="hn_at",
            strategy="forward",
            allow_exact_matches=False,
        )
        .join_asof(
            beacon_times,
            left_on="sent_at",
            right_on="b_at",
            strategy="backward",
            tolerance=BEACON_CONTEXT,
        )
        .join_asof(
            beacon_times.rename({"b_at": "nb_at", "b_prefix": "nb_prefix"}),
            left_on="sent_at",
            right_on="nb_at",
            strategy="forward",
            tolerance=BEACON_CONTEXT,
        )
    )

    # The latest header's own chunk, or the tail of the one before it (sent
    # just after the new one, or before it was sent at all).
    latest_fits = _header_fits("h")
    use_previous = ~latest_fits & (
        (_header_fits("ph") & (pl.col("sent_at") <= pl.col("h_at") + DOWNLINK_LEAD))
        | (pl.col("h_at") > pl.col("sent_at"))
    )
    joined = joined.with_columns(
        _from_header=latest_fits | (use_previous & _header_fits("ph")),
        **{
            c: pl.when(use_previous).then(pl.col(f"p{c}")).otherwise(pl.col(c))
            for c in header_columns
        },
    )

    # A beacon only names this chunk's downlink if that may have started by
    # the time the beacon was sent, and no header since says another did.
    since_start_us = pl.col("bulk_file_offset") * (
        (1 + DOWNLINK_RATE_SLACK) * 1_000_000 / DOWNLINK_BYTES_PER_SEC
    ) + _microseconds(DOWNLINK_SLACK)
    beacon_applies = (
        pl.col("b_prefix").is_not_null()
        & (
            (pl.col("sent_at") - pl.col("b_at")).dt.total_microseconds()
            <= since_start_us
        )
        & (
            pl.col("h_at").is_null()
            | (pl.col("b_at") >= pl.col("h_at"))
            | (pl.col("sent_at") <= pl.col("h_at") + DOWNLINK_LEAD)
        )
    ).fill_null(value=False)
    # See `HEADER_MARGIN`.
    length = pl.col("h_length").fill_null(0)
    header_window_applies = (
        pl.col("b_prefix").is_null()
        & pl.col("nb_prefix").is_null()
        & (pl.col("bulk_file_offset") >= pl.col("h_offset"))
        & (
            pl.col("h_length").is_null()
            | (pl.col("bulk_file_offset") < pl.col("h_offset") + length)
        )
        & (
            pl.col("sent_at")
            <= pl.col("h_at")
            + pl.duration(milliseconds=length * 1000 // MIN_DOWNLINK_BYTES_PER_SEC)
            + HEADER_MARGIN
        )
    ).fill_null(value=False)
    next_beacon_applies = (
        pl.col("nb_prefix").is_not_null()
        & (pl.col("hn_at").is_null() | (pl.col("nb_at") < pl.col("hn_at")))
    ).fill_null(value=False)
    attribution = (
        pl.when(pl.col("_from_header") | header_window_applies)
        .then(pl.lit(Attribution.HEADER.value))
        .when(beacon_applies)
        .then(pl.lit(Attribution.BEACON.value))
        .when(next_beacon_applies)
        .then(pl.lit(Attribution.NEXT_BEACON.value))
        .otherwise(pl.lit(Attribution.UNATTRIBUTED.value))
    )
    attributed = (
        joined.with_columns(attribution=attribution)
        .with_columns(
            path_sha256_prefix=pl.when(
                pl.col("attribution") == Attribution.HEADER.value
            )
            .then(pl.col("h_prefix"))
            .when(pl.col("attribution") == Attribution.BEACON.value)
            .then(pl.col("b_prefix"))
            .when(pl.col("attribution") == Attribution.NEXT_BEACON.value)
            .then(pl.col("nb_prefix")),
        )
        .drop(
            "_lead", "_h", "_from_header", "hn_at", *(f"p{c}" for c in header_columns)
        )
    )

    # A header names its path outright; a beacon's prefix is looked up.
    attributed = attributed.join(
        _prefix_paths(files),
        left_on="path_sha256_prefix",
        right_on="prefix",
        how="left",
    ).with_columns(
        file_path=pl.when(pl.col("attribution") == Attribution.HEADER.value)
        .then(pl.col("h_path"))
        .otherwise(pl.col("prefix_path"))
    )
    return _assign_downlinks(attributed.drop("prefix_path"))


def _assign_downlinks(attributed: pl.DataFrame) -> pl.DataFrame:
    """Add `file_key` (what's reassembled together) and `downlink_index`
    (consecutive chunks of one file, broken by a gap over `DOWNLINK_GAP` or
    a new header -- see the module docstring).
    """
    identity = (
        pl.when(pl.col("file_path").is_not_null())
        .then(pl.col("file_path"))
        .when(pl.col("path_sha256_prefix").is_not_null())
        .then(pl.lit(PREFIX_KEY_PREFIX) + pl.col("path_sha256_prefix"))
    )
    sorted_chunks = attributed.sort("sent_at", "bulk_file_offset").with_columns(
        _identity=identity
    )
    # When the chunk's downlink must have started, at `DOWNLINK_BYTES_PER_SEC`.
    started_us = pl.col("sent_at").dt.timestamp("us") - pl.col("bulk_file_offset") * (
        1_000_000 / DOWNLINK_BYTES_PER_SEC
    )
    new_downlink = (
        (pl.col("sent_at").diff() > DOWNLINK_GAP)
        | (started_us.diff().abs() > _microseconds(DOWNLINK_RESTART))
        | pl.col("_identity").ne_missing(pl.col("_identity").shift())
        | pl.col("h_at").ne_missing(pl.col("h_at").shift())
    ).fill_null(value=True)
    with_index = sorted_chunks.with_columns(
        downlink_index=new_downlink.cum_sum().cast(pl.Int64)
    )
    # An unattributed downlink is reassembled on its own, keyed by when it
    # started.
    started = pl.col("received_at").min().over("downlink_index")
    return with_index.with_columns(
        file_key=pl.coalesce(
            pl.col("_identity"),
            pl.lit(UNIDENTIFIED_KEY_PREFIX)
            + started.dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ"),
        )
    ).drop("_identity")


def _union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by `(start, end)` intervals."""
    total = 0
    reach = None
    for start, end in sorted(intervals):
        if reach is None or start > reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


def _named(attributed: pl.DataFrame) -> pl.DataFrame:
    """Only chunks of a file with a known path: the rest (an unknown path
    hash, or unattributed) are only used to fill named files' gaps.
    """
    if attributed.is_empty():
        return attributed
    return attributed.filter(pl.col("file_path").is_not_null())


def summarize_downlinks(attributed: pl.DataFrame) -> pl.DataFrame:
    """One row per downlink of a named file in `attribute_chunks`' output
    (with `rejected_reason` from `reject_inconsistent_downlinks`, if joined
    on), newest first.
    """
    attributed = _named(attributed)
    if attributed.is_empty():
        return pl.DataFrame(schema=_DOWNLINKS_SCHEMA)
    if "rejected_reason" not in attributed.columns:
        attributed = attributed.with_columns(rejected_reason=pl.lit(None, pl.String))

    end = pl.col("bulk_file_offset") + pl.col("bulk_data_len")
    from_header = pl.col("attribution") == Attribution.HEADER.value
    summary = attributed.group_by("downlink_index").agg(
        file_key=pl.col("file_key").first(),
        file_path=pl.col("file_path").first(),
        path_sha256_prefix=pl.col("path_sha256_prefix").first(),
        attributions=pl.col("attribution").unique(maintain_order=True),
        started_at=pl.col("received_at").min(),
        ended_at=pl.col("received_at").max(),
        chunk_count=pl.len(),
        distinct_offset_count=pl.col("bulk_file_offset").n_unique(),
        offset_min=pl.col("bulk_file_offset").min(),
        offset_end=end.max(),
        _starts=pl.col("bulk_file_offset"),
        _ends=end,
        header_received_at=pl.col("h_received_at").filter(from_header).first(),
        header_action=pl.col("h_action").filter(from_header).first(),
        header_offset=pl.col("h_offset").filter(from_header).first(),
        header_length=pl.col("h_length").filter(from_header).first(),
        header_file_size=pl.col("h_file_size").filter(from_header).first(),
        header_sha256=pl.col("h_sha256").filter(from_header).first(),
        rejected_reason=pl.col("rejected_reason").first(),
    )
    bytes_received = [
        _union_length(list(zip(starts, ends, strict=True)))
        for starts, ends in zip(
            summary["_starts"].to_list(), summary["_ends"].to_list(), strict=True
        )
    ]
    return (
        summary.with_columns(
            attributions=pl.col("attributions").map_elements(
                lambda values: orjson.dumps(list(values)).decode(),
                return_dtype=pl.String,
            ),
            bytes_received=pl.Series(bytes_received, dtype=pl.Int64),
            chunk_count=pl.col("chunk_count").cast(pl.Int64),
            distinct_offset_count=pl.col("distinct_offset_count").cast(pl.Int64),
        )
        .select(pl.col(name).cast(dtype) for name, dtype in _DOWNLINKS_SCHEMA.items())
        .sort("started_at", descending=True)
    )


@dataclass(frozen=True, slots=True)
class _DownlinkBytes:
    """One downlink reassembled on its own: its bytes, and which arrived."""

    index: int
    chunks: pl.DataFrame
    data: np.ndarray
    covered: np.ndarray

    @classmethod
    def of(cls, index: int, chunks: pl.DataFrame) -> _DownlinkBytes:
        result = reassemble_bulk_chunks(chunks)
        covered = np.zeros(result.span_bytes, dtype=bool)
        for segment in result.segments:
            if segment.status is not ByteStatus.MISSING:
                covered[segment.start : segment.end] = True
        return cls(index, chunks, np.frombuffer(result.data, dtype=np.uint8), covered)

    @property
    def covered_bytes(self) -> int:
        return int(np.count_nonzero(self.covered))


def _disagreeing_bytes(a: _DownlinkBytes, b: _DownlinkBytes) -> int:
    """How many bytes both `a` and `b` have, with different values."""
    n = min(len(a.data), len(b.data))
    both = a.covered[:n] & b.covered[:n]
    return int(np.count_nonzero((a.data[:n] != b.data[:n]) & both))


def _consistent_clusters(
    downlinks: list[_DownlinkBytes],
) -> list[list[_DownlinkBytes]]:
    """`downlinks` grouped so each group agrees on every byte its members
    share -- greedily, biggest downlink first.
    """
    clusters: list[list[_DownlinkBytes]] = []
    for downlink in sorted(downlinks, key=lambda d: (-d.covered_bytes, d.index)):
        for cluster in clusters:
            if all(
                _disagreeing_bytes(downlink, member) <= CONSISTENCY_TOLERANCE_BYTES
                for member in cluster
            ):
                cluster.append(downlink)
                break
        else:
            clusters.append([downlink])
    return clusters


def _cluster_data(cluster: list[_DownlinkBytes], file_size: int | None) -> bytes:
    """`cluster`'s chunks reassembled, padded/trimmed to `file_size`."""
    data = reassemble_bulk_chunks(pl.concat([d.chunks for d in cluster])).data
    if file_size is None:
        return data
    return data[:file_size].ljust(file_size, b"\0")


def _carried_file(
    chunks: pl.DataFrame, verified: dict[str, bytes], own_key: str
) -> str | None:
    """The verified file (other than `own_key`) that most of `chunks` match
    byte for byte at their offsets, if any.
    """
    matches: Counter[str] = Counter()
    for offset, hex_str in zip(
        chunks["bulk_file_offset"].to_list(),
        chunks["bulk_data_hex"].to_list(),
        strict=True,
    ):
        payload = bytes.fromhex(hex_str)
        for path, data in verified.items():
            if path != own_key and data[offset : offset + len(payload)] == payload:
                matches[path] += 1
    best = matches.most_common(1)
    if best and best[0][1] >= CARRIED_FILE_MIN_FRACTION * chunks.height:
        return best[0][0]
    return None


@dataclass(slots=True)
class _FileClusters:
    file_key: str
    clusters: list[list[_DownlinkBytes]]
    kept: int | None  # index into `clusters`, once decided


def _file_size_and_sha256s(
    file_key: str, chunks: pl.DataFrame, known: dict[str, dict[str, Any]]
) -> tuple[int | None, list[str]]:
    """A file's size as last reported, and every whole-file SHA-256 the
    satellite reported for it -- from step 10, and the headers it came
    down under.
    """
    row = known.get(file_key) or {}
    header_sizes = chunks.filter(pl.col("attribution") == Attribution.HEADER.value)[
        "h_file_size"
    ].drop_nulls()
    file_size = row.get("file_size") or (
        header_sizes[-1] if len(header_sizes) else None
    )
    sha256s = list(
        dict.fromkeys(
            [
                *cast("list[str]", orjson.loads(row.get("sha256s_seen") or "[]")),
                *chunks["h_sha256"].drop_nulls().to_list(),
            ]
        )
    )
    return file_size, sha256s


def reject_inconsistent_downlinks(
    attributed: pl.DataFrame, files: pl.DataFrame
) -> pl.DataFrame:
    """Which downlinks to leave out of their file, and why -- see the module
    docstring. Returns `downlink_index` and `rejected_reason`, one row per
    rejected downlink.
    """
    schema = {"downlink_index": pl.Int64, "rejected_reason": pl.String}
    if attributed.is_empty():
        return pl.DataFrame(schema=schema)

    known = {row["file_path"]: row for row in files.iter_rows(named=True)}
    attributed = attributed.filter(
        pl.col("bulk_file_offset") + pl.col("bulk_data_len")
        <= MAX_REASSEMBLY_SPAN_BYTES
    )
    verified: dict[str, bytes] = {}
    undecided: list[_FileClusters] = []

    # First pass: every file with one consistent set of downlinks, or with
    # one set the satellite's SHA-256 vouches for, is settled -- and
    # verified files become what the rest are checked against.
    for (file_key,), chunks in attributed.group_by("file_key"):
        key = cast("str", file_key)
        file_size, sha256s = _file_size_and_sha256s(key, chunks, known)
        downlinks = [
            _DownlinkBytes.of(cast("int", index), group)
            for (index,), group in chunks.group_by("downlink_index")
        ]
        clusters = _consistent_clusters(downlinks)
        datas = [_cluster_data(cluster, file_size) for cluster in clusters]
        verified_at = next(
            (
                i
                for i, data in enumerate(datas)
                if hashlib.sha256(data).hexdigest() in sha256s
            ),
            None,
        )
        if verified_at is not None:
            verified[key] = datas[verified_at]
        if len(clusters) > 1:
            undecided.append(_FileClusters(key, clusters, verified_at))

    # Second pass: of the rest, prefer a set not carrying another file's
    # bytes, then the one covering the most.
    rejected: list[dict[str, Any]] = []
    for file in undecided:
        carried = [
            _carried_file(pl.concat([d.chunks for d in c]), verified, file.file_key)
            for c in file.clusters
        ]
        kept = file.kept
        if kept is None:
            kept = min(
                range(len(file.clusters)),
                key=lambda i: (
                    carried[i] is not None,
                    -sum(d.covered_bytes for d in file.clusters[i]),
                ),
            )
        for i, cluster in enumerate(file.clusters):
            if i == kept:
                continue
            reason = (
                f"carries {carried[i]}'s content"
                if carried[i] is not None
                else "disagrees with the file's other downlinks"
            )
            rejected.extend(
                {"downlink_index": d.index, "rejected_reason": reason} for d in cluster
            )
    return pl.DataFrame(rejected, schema=schema)


@dataclass(slots=True)
class _FileBytes:
    """A file's bytes so far, and which of them are known."""

    data: np.ndarray
    covered: np.ndarray
    size: int | None  # as the satellite reported it, if it ever did

    @classmethod
    def of(cls, chunks: pl.DataFrame, size: int | None) -> _FileBytes:
        result = reassemble_bulk_chunks(chunks)
        length = max(result.span_bytes, size or 0)
        data = np.zeros(length, dtype=np.uint8)
        data[: result.span_bytes] = np.frombuffer(result.data, dtype=np.uint8)
        covered = np.zeros(length, dtype=bool)
        for segment in result.segments:
            if segment.status is not ByteStatus.MISSING:
                covered[segment.start : segment.end] = True
        return cls(data, covered, size)

    def try_add(self, chunks: pl.DataFrame) -> bool:
        """Add all of `chunks` if each fits, none disagrees with a byte
        already known (or with each other), and together they fill at least
        one gap -- else add none of them.
        """
        trial_data, trial_covered = self.data.copy(), self.covered.copy()
        fills = False
        for offset, hex_str in zip(
            chunks["bulk_file_offset"].to_list(),
            chunks["bulk_data_hex"].to_list(),
            strict=True,
        ):
            payload = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
            end = offset + len(payload)
            if self.size is not None and end > self.size:
                return False
            if end > len(trial_data):
                extra = end - len(trial_data)
                trial_data = np.concatenate(
                    [trial_data, np.zeros(extra, dtype=np.uint8)]
                )
                trial_covered = np.concatenate(
                    [trial_covered, np.zeros(extra, dtype=bool)]
                )
            known = trial_covered[offset:end]
            if np.any(trial_data[offset:end][known] != payload[known]):
                return False
            fills = fills or not bool(known.all())
            trial_data[offset:end] = payload
            trial_covered[offset:end] = True
        if fills:
            self.data, self.covered = trial_data, trial_covered
        return fills

    @property
    def sha256_if_complete(self) -> str | None:
        """The hash of the whole file, once every byte of it is known."""
        if self.size is None or not self.covered[: self.size].all():
            return None
        return hashlib.sha256(self.data[: self.size].tobytes()).hexdigest()


def rescue_nearby_chunks(attributed: pl.DataFrame, files: pl.DataFrame) -> pl.DataFrame:
    """Fill files' gaps with unattributed chunks received near their
    downlinks -- see the module docstring.

    `attributed` is `attribute_chunks`' output with `rejected_reason` joined
    on (see `reject_inconsistent_downlinks`). Returns it with every rescued
    chunk moved into its file: `attribution` `time_window`, the file's
    `file_key`/`file_path`/`path_sha256_prefix`, and a `downlink_index` of
    its own per (file, unattributed downlink it came from).
    """
    if attributed.is_empty():
        return attributed

    is_named = ~pl.col("file_key").str.starts_with(UNIDENTIFIED_KEY_PREFIX)
    kept = attributed.filter(is_named & pl.col("rejected_reason").is_null())
    loose = attributed.filter(pl.col("attribution") == Attribution.UNATTRIBUTED.value)
    if kept.is_empty() or loose.is_empty():
        return attributed

    # `h_at` is the latest header before each chunk (whether or not the
    # chunk is from it): a downlink is one header's "epoch".
    windows = kept.group_by("file_key", "downlink_index").agg(
        _start=pl.col("sent_at").min(),
        _end=pl.col("sent_at").max(),
        _h_at=pl.col("h_at").first(),
        _target_path=pl.col("file_path").first(),
    )
    seconds_away = pl.max_horizontal(
        (pl.col("_start") - pl.col("sent_at")).dt.total_seconds(),
        (pl.col("sent_at") - pl.col("_end")).dt.total_seconds(),
        pl.lit(0.0),
    )
    candidates = loose.join_where(
        windows.select(
            _target=pl.col("file_key"),
            _start=pl.col("_start"),
            _end=pl.col("_end"),
            _h_at=pl.col("_h_at"),
            _target_path=pl.col("_target_path"),
        ),
        pl.col("sent_at") >= pl.col("_start") - RESCUE_WINDOW,
        pl.col("sent_at") <= pl.col("_end") + RESCUE_WINDOW,
    )
    # A header between them means a new downlink started; and the file's own
    # header only sent its `offset`..`offset + length` -- see the module
    # docstring for what either takes.
    offset = pl.col("bulk_file_offset")
    in_own_headers_range = pl.col("h_path").ne_missing(pl.col("_target_path")) | (
        (offset >= pl.col("h_offset"))
        & (
            pl.col("h_length").is_null()
            | (offset < pl.col("h_offset") + pl.col("h_length"))
        )
    ).fill_null(value=True)
    candidates = candidates.with_columns(
        _away=seconds_away,
        _same_epoch=pl.col("h_at").eq_missing(pl.col("_h_at")) & in_own_headers_range,
    )
    # A chunk near more than one file's downlinks could be either's.
    unambiguous = (
        candidates.group_by("packet_id")
        .agg(pl.col("_target").n_unique().alias("_targets"))
        .filter(pl.col("_targets") == 1)
    )
    candidates = (
        candidates.join(unambiguous, on="packet_id")
        .sort("_away")
        .unique("packet_id", keep="first")
    )

    known = {row["file_path"]: row for row in files.iter_rows(named=True)}
    rescued: list[pl.DataFrame] = []
    for (target,), offered in candidates.group_by("_target"):
        key = cast("str", target)
        mine = kept.filter(pl.col("file_key") == key)
        size, sha256s = _file_size_and_sha256s(key, mine, known)
        groups: list[pl.DataFrame] = [
            group.sort("sent_at")
            for _, group in offered.sort("_away").group_by(
                "downlink_index", maintain_order=True
            )
        ]
        same_epoch = [g for g in groups if g["_same_epoch"].all()]
        across_header = [g for g in groups if not g["_same_epoch"].all()]

        file_bytes = _FileBytes.of(mine, size)
        taken = [g for g in same_epoch if file_bytes.try_add(g)]
        completed = file_bytes.sha256_if_complete
        if completed is not None and sha256s and completed not in sha256s:
            logger.info(
                f"Not rescuing chunks into {key}: they'd complete it with the "
                "wrong SHA-256."
            )
            taken, file_bytes = [], _FileBytes.of(mine, size)
        across = [g for g in across_header if file_bytes.try_add(g)]
        if across and file_bytes.sha256_if_complete in sha256s:
            taken += across
        if not taken:
            continue
        first = mine.row(0, named=True)
        rescued.extend(
            group.select("packet_id", "downlink_index").with_columns(
                _file_key=pl.lit(key),
                _file_path=pl.lit(first["file_path"], pl.String),
                _prefix=pl.lit(first["path_sha256_prefix"], pl.String),
            )
            for group in taken
        )
    if not rescued:
        return attributed

    # Each (file, source downlink) is a downlink of its own.
    next_index = cast("int", attributed["downlink_index"].max()) + 1
    moves = pl.concat(rescued).with_columns(
        _new_index=pl.struct("_file_key", "downlink_index").rank("dense").cast(pl.Int64)
        + next_index
        - 1
    )
    moved = pl.col("_new_index").is_not_null()
    return (
        attributed.join(moves.drop("downlink_index"), on="packet_id", how="left")
        .with_columns(
            attribution=pl.when(moved)
            .then(pl.lit(Attribution.TIME_WINDOW.value))
            .otherwise(pl.col("attribution")),
            file_key=pl.coalesce("_file_key", "file_key"),
            file_path=pl.when(moved).then("_file_path").otherwise("file_path"),
            path_sha256_prefix=pl.when(moved)
            .then("_prefix")
            .otherwise("path_sha256_prefix"),
            downlink_index=pl.coalesce("_new_index", "downlink_index"),
        )
        .drop("_file_key", "_file_path", "_prefix", "_new_index")
    )


def encode_segments(segments: list[ByteSegment]) -> str:
    """`segments` as stored in `downlinked_files.segments`."""
    return orjson.dumps(
        [
            [
                s.start,
                s.end,
                _SEGMENT_STATUS_CODES[s.status],
                s.min_copies,
                s.max_copies,
            ]
            for s in segments
        ]
    ).decode()


def decode_segments(segments_json: str) -> tuple[ByteSegment, ...]:
    """`downlinked_files.segments` back into `ByteSegment`s."""
    return tuple(
        ByteSegment(start, end, _SEGMENT_STATUSES[code], min_copies, max_copies)
        for start, end, code, min_copies, max_copies in cast(
            "list[list[Any]]", orjson.loads(segments_json)
        )
    )


def _content_kind(data: bytes, segments: list[ByteSegment]) -> str:
    """ "text" if the bytes that arrived are (almost) all printable ASCII --
    a `.run` script, a log -- else "binary".
    """
    sample = bytearray()
    for segment in segments:
        if segment.status is ByteStatus.MISSING:
            continue
        sample += data[segment.start : segment.end]
        if len(sample) >= _TEXT_SNIFF_BYTES:
            break
    if not sample:
        return "binary"
    printable = sum(
        1 for byte in sample[:_TEXT_SNIFF_BYTES] if byte in _PRINTABLE_BYTES
    )
    return (
        "text"
        if printable / min(len(sample), _TEXT_SNIFF_BYTES)
        >= _TEXT_MIN_PRINTABLE_FRACTION
        else "binary"
    )


_PICAM_START = b"START_CAM:"
_JPEG_SOI = b"\xff\xd8\xff"


def _detect_preview(file_key: str, data: bytes) -> DetectedImage | None:
    """An image to preview `data` as, if it is one -- see
    `file_reassembly.detect_image`.

    Plus, unlike there (where it's how a still-downloading image is told
    apart), a PiCAM image without its `@FACE` end line: seen in flight in
    complete, SHA-256-verified files, where the camera's output was cut
    short on board. What's there still decodes to the start of a JPEG,
    which browsers render as far as it goes.
    """
    try:
        image = detect_image(data)
        if image is None and data.startswith(_PICAM_START):
            jpg = parse_picam_ascii_to_jpg_bytes(
                data.decode("ascii", errors="replace"), enable_logs=False
            )
            if jpg.startswith(_JPEG_SOI):
                image = DetectedImage(
                    data=jpg,
                    media_type="image/jpeg",
                    suffix=".jpg",
                    label="Detected PiCAM image (no end-of-image line)",
                    is_converted=True,
                )
    except Exception as e:  # noqa: BLE001 -- a preview is never worth failing the step
        logger.debug(f"No preview for {file_key}: {type(e).__name__}: {e}")
        return None
    return image


def _reassemble_one(
    file_key: str,
    all_chunks: pl.DataFrame,
    known: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One row of `downlinked_files` from every chunk of one file (rejected
    downlinks' included, for the count -- they're left out of the bytes).
    """
    all_chunks = all_chunks.filter(
        pl.col("bulk_file_offset") + pl.col("bulk_data_len")
        <= MAX_REASSEMBLY_SPAN_BYTES
    )
    chunks = all_chunks.filter(pl.col("rejected_reason").is_null())
    file_size, expected = _file_size_and_sha256s(file_key, all_chunks, known)
    result = reassemble_bulk_chunks(chunks)
    data = result.data
    segments = list(result.segments)

    if file_size is not None and file_size > len(data):
        segments.append(ByteSegment(len(data), file_size, ByteStatus.MISSING, 0, 0))
        data += bytes(file_size - len(data))

    good = sum(s.length for s in segments if s.status is ByteStatus.GOOD)
    missing = sum(s.length for s in segments if s.status is ByteStatus.MISSING)
    conflicting = sum(s.length for s in segments if s.status is ByteStatus.CONFLICTING)
    is_complete = bool(data) and good == len(data)
    reassembled = hashlib.sha256(data).hexdigest()

    image = _detect_preview(file_key, data)

    return {
        "file_key": file_key,
        "file_path": all_chunks["file_path"].drop_nulls().first(),
        "path_sha256_prefix": all_chunks["path_sha256_prefix"].drop_nulls().first(),
        "file_size": file_size,
        "byte_length": len(data),
        "good_bytes": good,
        "missing_bytes": missing,
        "conflict_bytes": conflicting,
        "received_fraction": (good + conflicting) / len(data) if data else 0.0,
        "is_complete": is_complete,
        "sha256": reassembled,
        "expected_sha256s": orjson.dumps(expected).decode(),
        "sha256_verified": (reassembled in expected)
        if is_complete and expected
        else None,
        "first_downlinked_at": chunks["received_at"].min(),
        "last_downlinked_at": chunks["received_at"].max(),
        "downlink_count": chunks["downlink_index"].n_unique(),
        "rejected_downlink_count": all_chunks["downlink_index"].n_unique()
        - chunks["downlink_index"].n_unique(),
        "chunk_count": chunks.height,
        "rescued_chunk_count": int(
            (chunks["attribution"] == Attribution.TIME_WINDOW.value).sum()
        ),
        "segments": encode_segments(segments),
        "content_kind": "image" if image else _content_kind(data, segments),
        "preview_media_type": image.media_type if image else None,
        "preview_label": image.label if image else None,
        "preview_image": image.data if image and image.is_converted else None,
        "data": data,
    }


def reassemble_downlinked_files(
    attributed: pl.DataFrame, files: pl.DataFrame
) -> pl.DataFrame:
    """One row per named file in `attribute_chunks`' output (`file_key`),
    plus one per file in step 10's `files` never downlinked (no `data`, zero
    downlinks) -- newest downlink first, then newest mentioned. See
    `_FILES_SCHEMA`. Downlinks with a `rejected_reason` (see
    `reject_inconsistent_downlinks`) are left out.
    """
    rows: list[dict[str, Any]] = []
    attributed = _named(attributed)
    if not attributed.is_empty():
        if "rejected_reason" not in attributed.columns:
            attributed = attributed.with_columns(
                rejected_reason=pl.lit(None, pl.String)
            )
        known = {row["file_path"]: row for row in files.iter_rows(named=True)}
        rows = [
            _reassemble_one(cast("str", file_key), group, known)
            for (file_key,), group in attributed.group_by("file_key")
            # A file all of whose downlinks were rejected has nothing to show.
            if group["rejected_reason"].is_null().any()
        ]
    downlinked = pl.DataFrame(rows, schema=_FILES_SCHEMA).drop(
        _SATELLITE_FILE_COLUMNS.values()
    )
    if files.is_empty():
        return downlinked.with_columns(
            pl.lit(None, _FILES_SCHEMA[name]).alias(name)
            for name in _SATELLITE_FILE_COLUMNS.values()
        ).select(*_FILES_SCHEMA)

    satellite = files.select(
        "file_path",
        *(pl.col(src).alias(dst) for src, dst in _SATELLITE_FILE_COLUMNS.items()),
        _file_size=pl.col("file_size"),
        _prefix=pl.col("path_sha256_prefix"),
        _sha256s=pl.col("sha256s_seen"),
    )
    never = pl.col("file_key").is_null()  # in step 10, but never downlinked
    return (
        downlinked.join(satellite, on="file_path", how="full", coalesce=True)
        .with_columns(
            file_key=pl.coalesce("file_key", "file_path"),
            file_size=pl.coalesce("file_size", "_file_size"),
            path_sha256_prefix=pl.coalesce("path_sha256_prefix", "_prefix"),
            expected_sha256s=pl.when(never)
            .then(pl.col("_sha256s").fill_null("[]"))
            .otherwise("expected_sha256s"),
            **{
                name: pl.when(never).then(0).otherwise(name)
                for name in (
                    "downlink_count",
                    "rejected_downlink_count",
                    "chunk_count",
                    "rescued_chunk_count",
                )
            },
        )
        .select(pl.col(name).cast(dtype) for name, dtype in _FILES_SCHEMA.items())
        .sort("last_downlinked_at", "last_seen_at", descending=True, nulls_last=True)
    )


def _clock_offsets(decoded: pl.LazyFrame) -> pl.DataFrame:
    """The satellite's clock minus `received_at`, by `received_at`: a
    rolling median over the packets with a `transmitted_at_sat_clock`.
    """
    return (
        decoded.select(
            "received_at",
            _offset=(
                pl.col("transmitted_at_sat_clock") - pl.col("received_at")
            ).dt.total_microseconds(),
        )
        .drop_nulls()
        .sort("received_at")
        .collect()
        .with_columns(
            _offset=pl.col("_offset").rolling_median_by(
                "received_at", window_size=CLOCK_OFFSET_WINDOW
            )
        )
    )


def _with_sent_at(packets: pl.DataFrame, offsets: pl.DataFrame) -> pl.DataFrame:
    """`packets` plus `sent_at`: `transmitted_at_sat_clock`, else
    `received_at` moved onto the satellite's clock by the nearest offset in
    `offsets` (see `_clock_offsets`), if any.
    """
    return (
        packets.sort("received_at")
        .join_asof(
            offsets,
            on="received_at",
            strategy="nearest",
            tolerance=CLOCK_OFFSET_WINDOW,
        )
        .with_columns(
            sent_at=pl.coalesce(
                "transmitted_at_sat_clock",
                pl.col("received_at")
                + pl.duration(microseconds=pl.col("_offset").fill_null(0)),
            )
        )
        .drop("_offset", "transmitted_at_sat_clock")
    )


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def _log_summary(attributed: pl.DataFrame, downlinked: pl.DataFrame) -> None:
    by_attribution = (
        attributed.group_by("attribution")
        .agg(pl.len(), pl.col("bulk_data_len").sum())
        .sort("len", descending=True)
    )
    logger.info(
        "Chunks by attribution: "
        + ", ".join(
            f"{name} {count:,} ({size / 1e6:,.1f} MB)"
            for name, count, size in by_attribution.iter_rows()
        )
        + "."
    )
    named = downlinked.filter(pl.col("downlink_count") > 0)
    unnamed = attributed.filter(pl.col("file_path").is_null()).height
    logger.info(
        f"{named.height:,} named file(s) downlinked: "
        f"{named['is_complete'].sum():,} complete, "
        f"{named['sha256_verified'].sum():,} verified by SHA-256 "
        f"({(named['sha256_verified'] == False).sum():,} mismatched); "  # noqa: E712
        f"{downlinked.height - named.height:,} known file(s) never "
        f"downlinked; {unnamed:,} chunk(s) of no named file left out. "
        f"{named['rejected_downlink_count'].sum():,} downlink(s) rejected as "
        "not matching the rest of their file; "
        f"{named['rescued_chunk_count'].sum():,} unattributed chunk(s) rescued "
        "into files' gaps."
    )


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `file_downlink_time_periods.parquet` and `downlinked_files.parquet`
    from steps 3 and 10's outputs, in `data_dir`.
    """
    decoded_path = data_dir / step_3_pipeline.OUTPUT_FILENAME
    files_path = data_dir / step_10_pipeline.OUTPUT_FILENAME
    mentions_path = data_dir / step_10_pipeline.MENTIONS_OUTPUT_FILENAME
    for path, step in (
        (decoded_path, "step_3"),
        (files_path, "step_10"),
        (mentions_path, "step_10"),
    ):
        if not path.exists():
            msg = f"{path} not found -- run {step} first."
            raise FileNotFoundError(msg)

    logger.info(f"Reading {decoded_path}")
    decoded = pl.scan_parquet(decoded_path).filter(pl.col("csp_crc_valid"))
    offsets = _clock_offsets(decoded)
    chunks = _with_sent_at(
        decoded.filter(pl.col("packet_type") == "BULK_FILE_DOWNLINK")
        .select(*_CHUNK_COLUMNS, "transmitted_at_sat_clock")
        .collect(),
        offsets,
    )
    beacons = (
        decoded.filter(
            (pl.col("packet_type") == "BEACON_EXTENDED_V5PLUS")
            & pl.col("unix_epoch_time_ms").is_between(1, MAX_VALID_EPOCH_MS)
        )
        .select(
            "received_at",
            "bulk_downlink_file_path_sha256_prefix",
            # Stored ones too: sent long ago, but each says what was being
            # downlinked when it was stamped, as a live one would have.
            sent_at=pl.from_epoch(
                pl.col("unix_epoch_time_ms") + _BEACON_STAMP_TO_TRANSMIT_MS,
                time_unit="ms",
            )
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us"),
        )
        .collect()
    )
    responses = _with_sent_at(
        decoded.filter(pl.col("packet_type") == "TCMD_RESPONSE")
        .select("received_at", "transmitted_at_sat_clock")
        .collect(),
        offsets,
    ).unique("received_at")
    files = pl.read_parquet(files_path)
    mentions = pl.read_parquet(mentions_path).join(
        responses.select(seen_at="received_at", sent_at="sent_at"),
        on="seen_at",
        how="left",
    )

    attributed = attribute_chunks(chunks, beacons, mentions, files)
    attributed = attributed.join(
        reject_inconsistent_downlinks(attributed, files),
        on="downlink_index",
        how="left",
    )
    attributed = rescue_nearby_chunks(attributed, files)
    downlinks = summarize_downlinks(attributed)
    downlinked = reassemble_downlinked_files(attributed, files)

    _write_parquet_atomic(downlinks, data_dir / DOWNLINK_TIME_PERIODS_OUTPUT_FILENAME)
    out_path = data_dir / DOWNLINKED_FILES_OUTPUT_FILENAME
    _write_parquet_atomic(downlinked, out_path)

    _log_summary(attributed, downlinked)
    logger.info(
        f"Done. {downlinks.height:,} downlink(s) of {downlinked.height:,} "
        f"file(s) written to {out_path}."
    )
