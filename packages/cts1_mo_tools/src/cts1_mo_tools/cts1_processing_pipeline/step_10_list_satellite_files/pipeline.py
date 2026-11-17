"""Step 10: list every file known to exist on the satellite.

Nothing downlinks a file system snapshot, so the satellite's files are
pieced together from every telecommand response (reassembled by step 5)
that names one:

  - Directory listings (`fs_list_directory_json`, the firmware's
    `LFS_list_directory_json_dict`): a JSON object of `"/dir/name": size`
    pairs, with `"/dir/./"`, `"/dir/../"` and any subdirectories
    (`"/dir/sub/"`) as `null`, in littlefs's (name) order. See "Listing
    quirks" below.
  - Bulk downlink headers (`bulk_downlink_start*`), each with the `file`,
    its `file_size` and SHA-256.
  - ADCS SD card file copies (`adcs_get_latest_sd_file_blob*`), likewise:
    the copy the OBC makes of the latest ADCS SD card file, and then
    downlinks.
  - File maps (`get_file_map*`): `file`, `size`, SHA-256.
  - MPI analyses (`analyze_mpi_data*`): `file`, `size`, SHA-256.
  - Any other response with an `action` and a `file`.

Each such mention is one row of `satellite_file_mentions.parquet`, and
`satellite_files.parquet` has one row per file path: its latest known size
and SHA-256 (and every one seen), when it was first/last mentioned, by
which sources, and whether the latest complete listing of its directory
covering its name (see below) still has it.

A file is keyed by its normalized path (`normalize_path`): no leading `/`,
no empty or `.` segments (`camera/x.img`), the way it's named in every
response but the listings. The v5+ beacons'
`bulk_downlink_file_path_sha256_prefix` is the first 3 bytes of SHA-256 of
the path exactly as given in the downlink telecommand (see the firmware's
`extended_beacon_blob`), which the header echoes back as its `file` -- so
each mention also keeps its `raw_path`, and a file's
`path_sha256_prefixes` cover every form a header named it by, along with
its normalized one (`path_sha256_prefix`). Step 11 uses those to tell which
file a downlink is of.

Listing quirks (from the firmware's `LFS_list_directory_json_dict`):

  - It takes an `offset` and a `count`, and nothing in the response says
    whether either was used, so a listing may be one page of the
    directory. Since littlefs lists in name order, a page with the `./`
    entry (offset 0) still covers every name up to its last entry, and one
    without it every name between its first and last entries. A file is
    only judged against listings covering its name -- e.g. a `count`-limited
    page of `/ADCS` says nothing about the files sorting after its last
    entry. (Seen: 61-, 37- and 21-entry listings each followed by a page of
    just their first 28, 6 and 9 entries.)
  - The keys are built from the directory argument as typed (only a single
    trailing `/` trimmed, and a leading one added), so `//`, `./` and the
    like are normalized away (`normalize_path`).
  - An entry of a type other than file/directory still emits its `,`, so a
    listing isn't guaranteed to be valid JSON: entries are matched one by
    one, and whether a listing is complete comes from step 5 (every frame
    arrived), not from parsing it.
  - A listing that overflows the response buffer, or of a directory that
    doesn't exist, comes back as a plain-text error instead, and is
    ignored.

A response with a frame that never arrived has that frame filled with `?`s
by step 5, so everything here is parsed tolerantly (see
`file_reassembly.parse_response_properties`), and a path with a `?` in it
is dropped as garbled. Only a complete listing counts towards whether a
file is still listed.

Reads step 5's output only; parquet-in-parquet-out, recomputed from
scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_DATA_DIR",
    "DOWNLINK_HEADER_SOURCES",
    "MENTIONS_OUTPUT_FILENAME",
    "OUTPUT_FILENAME",
    "ListingStatus",
    "MentionSource",
    "extract_file_mentions",
    "normalize_path",
    "path_sha256_prefix",
    "run",
    "summarize_satellite_files",
]

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import orjson
import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.file_reassembly import (
    SHA256_HEX_LEN,
    parse_response_properties,
)
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

DEFAULT_DATA_DIR = step_5_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "satellite_files.parquet"
MENTIONS_OUTPUT_FILENAME = "satellite_file_mentions.parquet"

# How many bytes of SHA-256(path) the v5+ beacon carries -- see
# `decode_beacon_extended_v5plus_packet`.
PATH_SHA256_PREFIX_BYTES = 3


class MentionSource(StrEnum):
    """What kind of telecommand response a file was mentioned in."""

    LISTING = "listing"
    BULK_DOWNLINK_HEADER = "bulk_downlink_header"
    ADCS_FILE_HEADER = "adcs_file_header"
    FILE_MAP = "file_map"
    MPI_ANALYSIS = "mpi_analysis"
    OTHER = "other"


# By `action` prefix. Order matters only in that none is a prefix of another.
_ACTION_SOURCES: tuple[tuple[str, MentionSource], ...] = (
    ("bulk_downlink_start", MentionSource.BULK_DOWNLINK_HEADER),
    ("adcs_get_latest_sd_file_blob", MentionSource.ADCS_FILE_HEADER),
    ("get_file_map", MentionSource.FILE_MAP),
    ("analyze_mpi_data", MentionSource.MPI_ANALYSIS),
)

# Sources that are a file being downlinked -- what step 11 reads as the
# header of the chunks that follow.
DOWNLINK_HEADER_SOURCES = frozenset(
    {MentionSource.BULK_DOWNLINK_HEADER, MentionSource.ADCS_FILE_HEADER}
)


class ListingStatus(StrEnum):
    """How a file stands against the latest complete listing of its
    directory -- see `summarize_satellite_files`.
    """

    LISTED = "listed"
    # The latest covering listing is newer than every mention of the file
    # and doesn't have it: most likely deleted since.
    NOT_IN_LATEST_LISTING = "not_in_latest_listing"
    # Mentioned after the latest covering listing (which doesn't have it):
    # most likely created since.
    NEWER_THAN_LATEST_LISTING = "newer_than_latest_listing"
    # No complete listing of its directory has covered its name.
    NEVER_LISTED = "never_listed"


# A listing's entries. Matched pair by pair rather than parsed as JSON, so a
# listing with a missing frame (see the module docstring) still gives up
# every entry outside the gap.
_LISTING_ENTRY_RE = re.compile(r'"(/(?:[^"\\]|\\.)*)"\s*:\s*(-?\d+|null)')
# What a listing response starts with, after any whitespace.
_LISTING_START_RE = re.compile(r'\s*\{\s*"/')
_LISTED_DIR_SUFFIX = "/./"
_PARENT_DIR_SUFFIX = "/../"
_SHA256_RE = re.compile(rf"[0-9a-fA-F]{{{SHA256_HEX_LEN}}}")

_MENTIONS_SCHEMA = {
    "file_path": pl.String,  # normalized -- see `normalize_path`
    "raw_path": pl.String,  # exactly as the response gave it
    "source": pl.String,  # a `MentionSource`
    "action": pl.String,
    "seen_at": pl.Datetime("us", "UTC"),  # the response's first frame
    "tcmd_sent_at": pl.Datetime("us", "UTC"),
    "file_size": pl.Int64,
    "sha256": pl.String,  # lowercase; whole-file only, never a chunk's
    "crc16": pl.String,
    # For a downlink header: the byte range of the file being downlinked
    # (`offset`, and `length` -- or the rest of the file if not given).
    "downlink_offset": pl.Int64,
    "downlink_length": pl.Int64,
    # For a listing: the directory listed (normalized, "" for the root);
    # whether every frame of the listing arrived; and the range of entry
    # names it covers (see the module docstring) -- `listing_first_name`
    # null for a listing starting at the top (offset 0).
    "listing_directory": pl.String,
    "listing_complete": pl.Boolean,
    "listing_first_name": pl.String,
    "listing_last_name": pl.String,
    "response_complete": pl.Boolean,
}

_FILES_SCHEMA = {
    "file_path": pl.String,
    "directory": pl.String,
    "file_name": pl.String,
    "path_sha256_prefix": pl.String,  # of `file_path`
    # JSON array: of `file_path`, then of every other form a downlink
    # header (or other action) named the file by -- see the module docstring.
    "path_sha256_prefixes": pl.String,
    "file_size": pl.Int64,  # from the latest mention with a size
    "file_sizes_seen": pl.String,  # JSON array, ascending
    "sha256": pl.String,  # from the latest mention with one
    "sha256s_seen": pl.String,  # JSON array
    "sources": pl.String,  # JSON array of `MentionSource`s
    "mention_count": pl.Int64,
    "first_seen_at": pl.Datetime("us", "UTC"),
    "last_seen_at": pl.Datetime("us", "UTC"),
    # The latest complete listing of its directory covering its name.
    "last_listed_at": pl.Datetime("us", "UTC"),
    "listing_status": pl.String,  # a `ListingStatus`
}


def normalize_path(path: str) -> str | None:
    """`path` the way the satellite names it outside listings -- no leading
    `/`, no empty or `.` segments, `..` resolved (a listing's keys echo its
    directory argument as typed) -- or None if it's empty or garbled by a
    missing frame.
    """
    if "?" in path:
        return None
    segments: list[str] = []
    for segment in path.strip().split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    return "/".join(segments) or None


def path_sha256_prefix(path: str) -> str:
    """The hex of the first bytes of SHA-256(`path`), as a v5+ beacon
    reports the file being downlinked (`path` already normalized).
    """
    return hashlib.sha256(path.encode()).digest()[:PATH_SHA256_PREFIX_BYTES].hex()


def _directory_of(path: str) -> str:
    return path.rpartition("/")[0]


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _whole_sha256(properties: dict[str, Any]) -> str | None:
    """The whole file's SHA-256 (never a `chunk_sha256`), if it's all there."""
    for key in ("file_sha256", "sha256"):
        value = _str_or_none(properties.get(key))
        if value is not None and _SHA256_RE.fullmatch(value):
            return value.lower()
    return None


def _source_of(action: str) -> MentionSource:
    for prefix, source in _ACTION_SOURCES:
        if action.startswith(prefix):
            return source
    return MentionSource.OTHER


def _entry_name(key: str) -> str:
    """An entry's own name (its last path segment), as littlefs sorts it."""
    return key.rstrip("/").rpartition("/")[2]


def _listing_directory(keys: list[str]) -> str | None:
    """Which directory a listing is of: from its `./` entry if it has one,
    else (a page past the first) the parent all its entries share.
    """
    for key in keys:
        if key.endswith(_LISTED_DIR_SUFFIX):
            return normalize_path(key.removesuffix(_LISTED_DIR_SUFFIX)) or ""
    parents = {
        normalize_path(key.rstrip("/").rpartition("/")[0]) or ""
        for key in keys
        if "?" not in key
    }
    return parents.pop() if len(parents) == 1 else None


def _listing_mentions(
    text: str, *, response_complete: bool, base: dict[str, Any]
) -> list[dict[str, Any]]:
    """One mention per file in a directory listing response."""
    entries = [(m.group(1), m.group(2)) for m in _LISTING_ENTRY_RE.finditer(text)]
    keys = [key for key, _ in entries]
    # The range of names this listing covers (see the module docstring),
    # from its first and last real entries -- "." and ".." always come
    # first, and only on the page starting at offset 0.
    named = [
        key
        for key in keys
        if not key.endswith((_LISTED_DIR_SUFFIX, _PARENT_DIR_SUFFIX)) and "?" not in key
    ]
    starts_at_top = any(key.endswith(_LISTED_DIR_SUFFIX) for key in keys)
    listing = {
        "listing_directory": _listing_directory(keys),
        "listing_complete": response_complete and text.rstrip().endswith("}"),
        "listing_first_name": (
            None if starts_at_top or not named else _entry_name(named[0])
        ),
        "listing_last_name": _entry_name(named[-1]) if named else None,
    }

    mentions: list[dict[str, Any]] = []
    for key, raw_size in entries:
        if key.endswith("/"):  # ".", ".." or a subdirectory
            continue
        path = normalize_path(key)
        if path is None:
            continue
        mentions.append(
            base
            | listing
            | {
                "file_path": path,
                "raw_path": key,
                "source": MentionSource.LISTING.value,
                "file_size": None if raw_size == "null" else int(raw_size),
            }
        )
    return mentions


def _action_mention(text: str, base: dict[str, Any]) -> dict[str, Any] | None:
    """The mention in a response with an `action` and a `file`, if any."""
    properties, _parsed_whole = parse_response_properties(text)
    action = _str_or_none(properties.get("action"))
    file = _str_or_none(properties.get("file"))
    if action is None or file is None:
        return None
    path = normalize_path(file)
    if path is None:
        return None
    crc16 = _str_or_none(properties.get("file_crc16")) or _str_or_none(
        properties.get("crc16")
    )
    file_size = (
        _int_or_none(properties.get("file_size"))
        if "file_size" in properties
        else _int_or_none(properties.get("size"))
    )
    source = _source_of(action)
    downlink: dict[str, Any] = {}
    if source in DOWNLINK_HEADER_SOURCES:
        offset = _int_or_none(properties.get("offset")) or 0
        length = _int_or_none(properties.get("length"))
        if length is None and file_size is not None:
            length = max(file_size - offset, 0)
        downlink = {"downlink_offset": offset, "downlink_length": length}
    return (
        base
        | downlink
        | {
            "file_path": path,
            "raw_path": file,
            "source": source.value,
            "action": action,
            "file_size": file_size,
            "sha256": _whole_sha256(properties),
            "crc16": crc16,
        }
    )


def extract_file_mentions(tcmd_responses: pl.DataFrame) -> pl.DataFrame:
    """Every file named in `tcmd_responses` (step 5's
    `reassembled_tcmd_responses.parquet`), one row per (response, file),
    oldest first -- see the module docstring for which responses count.
    """
    rows: list[dict[str, Any]] = []
    if not tcmd_responses.is_empty():
        for text, seen_at, tcmd_sent_at, complete in zip(
            tcmd_responses["tcmd_response_text"].to_list(),
            tcmd_responses["first_received_at"].to_list(),
            tcmd_responses["tcmd_sent_at"].to_list(),
            tcmd_responses["is_complete"].to_list(),
            strict=True,
        ):
            if not text:
                continue
            base: dict[str, Any] = {
                "seen_at": seen_at,
                "tcmd_sent_at": tcmd_sent_at,
                "response_complete": bool(complete),
            }
            if _LISTING_START_RE.match(text):
                rows.extend(
                    _listing_mentions(text, response_complete=bool(complete), base=base)
                )
            elif '"action"' in text and '"file"' in text:
                mention = _action_mention(text, base)
                if mention is not None:
                    rows.append(mention)

    return (
        pl.DataFrame(rows, schema=_MENTIONS_SCHEMA)
        if rows
        else pl.DataFrame(schema=_MENTIONS_SCHEMA)
    ).sort("seen_at", "file_path", nulls_last=True)


def _json_list(values: list[Any]) -> str:
    return orjson.dumps(values).decode()


@dataclass(frozen=True, slots=True)
class _Listing:
    """One complete directory listing (or page of one) -- see
    `_listing_mentions` for the name range it covers.
    """

    seen_at: datetime
    paths: frozenset[str]
    first_name: str | None  # None: from the top of the directory
    last_name: str | None  # None: an empty directory (covers everything)

    def covers(self, name: str) -> bool:
        if self.first_name is not None and name < self.first_name:
            return False
        return self.last_name is None or name <= self.last_name


def _listings_by_directory(mentions: pl.DataFrame) -> dict[str, list[_Listing]]:
    """Every complete listing, by directory, newest first."""
    listings = (
        mentions.filter(
            (pl.col("source") == MentionSource.LISTING.value)
            & pl.col("listing_complete")
            & pl.col("listing_directory").is_not_null()
        )
        .group_by(
            "listing_directory",
            "seen_at",
            "listing_first_name",
            "listing_last_name",
        )
        .agg(pl.col("file_path"))
        .sort("seen_at", descending=True)
    )
    by_directory: dict[str, list[_Listing]] = {}
    for directory, seen_at, first_name, last_name, paths in listings.iter_rows():
        by_directory.setdefault(directory, []).append(
            _Listing(seen_at, frozenset(paths), first_name, last_name)
        )
    return by_directory


def _latest_covering_listing(path: str, listings: list[_Listing]) -> _Listing | None:
    name = path.rpartition("/")[2]
    return next((listing for listing in listings if listing.covers(name)), None)


def _listing_status(
    path: str, last_seen_at: datetime, latest: _Listing | None
) -> ListingStatus:
    if latest is None:
        return ListingStatus.NEVER_LISTED
    if path in latest.paths:
        return ListingStatus.LISTED
    if last_seen_at > latest.seen_at:
        return ListingStatus.NEWER_THAN_LATEST_LISTING
    return ListingStatus.NOT_IN_LATEST_LISTING


def summarize_satellite_files(mentions: pl.DataFrame) -> pl.DataFrame:
    """One row per file path in `mentions` (`extract_file_mentions`'s
    output), newest-mentioned first -- see `_FILES_SCHEMA` for the columns.

    A file's `listing_status` compares it with the latest *complete*
    listing of its directory that covers its name (a listing may be one
    page of the directory -- see the module docstring): still in it
    (`listed`), or not -- and then whether the file was mentioned after
    that listing (so likely created since) or not (so likely deleted since).
    """
    if mentions.is_empty():
        return pl.DataFrame(schema=_FILES_SCHEMA)

    listings = _listings_by_directory(mentions)
    rows: list[dict[str, Any]] = []
    for (path,), group in mentions.sort("seen_at").group_by(
        "file_path", maintain_order=True
    ):
        assert isinstance(path, str)
        sizes = group["file_size"].drop_nulls()
        sha256s = group["sha256"].drop_nulls()
        last_seen_at = group["seen_at"].max()
        directory = _directory_of(path)
        latest = _latest_covering_listing(path, listings.get(directory, []))
        # Listings name every file with a leading "/" whatever was typed, so
        # only other responses' raw paths are forms a downlink could use.
        raw_paths = group.filter(pl.col("source") != MentionSource.LISTING.value)[
            "raw_path"
        ].to_list()
        prefixes = dict.fromkeys(path_sha256_prefix(p) for p in [path, *raw_paths])
        rows.append(
            {
                "file_path": path,
                "directory": directory,
                "file_name": path.rpartition("/")[2],
                "path_sha256_prefix": path_sha256_prefix(path),
                "path_sha256_prefixes": _json_list(list(prefixes)),
                "file_size": sizes[-1] if len(sizes) else None,
                "file_sizes_seen": _json_list(sorted(set(sizes.to_list()))),
                "sha256": sha256s[-1] if len(sha256s) else None,
                "sha256s_seen": _json_list(list(dict.fromkeys(sha256s.to_list()))),
                "sources": _json_list(sorted(set(group["source"].to_list()))),
                "mention_count": group.height,
                "first_seen_at": group["seen_at"].min(),
                "last_seen_at": last_seen_at,
                "last_listed_at": latest.seen_at if latest else None,
                "listing_status": _listing_status(
                    path,
                    last_seen_at,  # pyright: ignore[reportArgumentType]
                    latest,
                ).value,
            }
        )
    return pl.DataFrame(rows, schema=_FILES_SCHEMA).sort(
        "last_seen_at", "file_path", descending=[True, False], nulls_last=True
    )


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `satellite_file_mentions.parquet` and
    `satellite_files.parquet` from step 5's output, in `data_dir`.
    """
    tcmd_path = data_dir / step_5_pipeline.OUTPUT_FILENAME
    if not tcmd_path.exists():
        msg = f"{tcmd_path} not found -- run step_5 first."
        raise FileNotFoundError(msg)

    logger.info(f"Reading {tcmd_path}")
    mentions = extract_file_mentions(
        pl.read_parquet(
            tcmd_path,
            columns=[
                "first_received_at",
                "tcmd_sent_at",
                "is_complete",
                "tcmd_response_text",
            ],
        )
    )
    files = summarize_satellite_files(mentions)

    _write_parquet_atomic(mentions, data_dir / MENTIONS_OUTPUT_FILENAME)
    files_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(files, files_path)

    status_counts = files["listing_status"].value_counts(sort=True).rows()
    logger.info(
        f"Done. {files.height:,} file(s) on the satellite, from "
        f"{mentions.height:,} mention(s), written to {files_path} -- by "
        "listing status: "
        + ", ".join(f"{status} {count:,}" for status, count in status_counts)
        + "."
    )
