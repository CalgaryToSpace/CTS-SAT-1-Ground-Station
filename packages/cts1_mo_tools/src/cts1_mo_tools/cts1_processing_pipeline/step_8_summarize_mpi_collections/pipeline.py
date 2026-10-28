"""Step 8: summarize MPI collections, alongside the satellite's own
analyses of its MPI data files.

Step 6 recovers MPI frames from bulk file downlinks and labels each with
its MPI collection (`step_6_pipeline.assign_mpi_collections`): runs of
frames from one recording, each identified by its first frame. This step
summarizes each collection from its frames: its start/end, frames present
vs expected from the counters, and its largest gap.

The satellite can also summarize an MPI data file itself, on board, with
the `analyze_mpi_data` telecommand: its response (reassembled by step 5)
gives the file's name, size, SHA-256, frame counts, `earliest`/`latest`
timestamps, and the strong signals above a DN threshold. Every distinct
one is written to `mpi_analyses.parquet` (`extract_all_mpi_analyses`),
with how many of each strong signal's frames were downlinked
(`count_downlinked_strong_signal_frames`). Each file's primary analysis --
the latest version, then the largest -- is what its collection uses
(`primary_mpi_analyses`), along with the strong signal count at every
threshold the file was analyzed with.

Every frame collection overlapping an analysis's `earliest`..`latest` is
merged into it, and the collection is then timed by the analysis -- the
whole file on board, not just the frames that happened to be downlinked --
with the file's metadata alongside. An analysis with no frames downlinked
yet is still a collection. Each collection also gets the integration
period(s) and board temperature range its frames reported, and where the
satellite was at its start/end and whether it was sunlit, propagated from
step 7's TLE history (see `orbit.py`). The result is
`mpi_collections.parquet` (`summarize_mpi_collections`).

This step reads the output of steps 5, 6 and 7 (5's and 7's are optional:
without them, the collections just aren't merged with any analyses, or
placed in orbit). Like steps 2-6, it's parquet-in-parquet-out and
reprocesses from scratch on every run.
"""

from __future__ import annotations

__all__ = [
    "ANALYSES_OUTPUT_FILENAME",
    "COLLECTIONS_OUTPUT_FILENAME",
    "DEFAULT_DATA_DIR",
    "MPI_ANALYSIS_ACTION_PREFIX",
    "RESYNC_TOLERANCE",
    "count_downlinked_strong_signal_frames",
    "extract_all_mpi_analyses",
    "extract_mpi_analyses",
    "primary_mpi_analyses",
    "run",
    "sort_mpi_analyses_by_preference",
    "summarize_mpi_collections",
]

import itertools
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import orjson
import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import MAX_VALID_EPOCH_MS
from cts1_mo_tools.cts1_processing_pipeline import orbit
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_7_fetch_tle_history import (
    pipeline as step_7_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATA_DIR = step_6_pipeline.DEFAULT_DATA_DIR
COLLECTIONS_OUTPUT_FILENAME = "mpi_collections.parquet"
ANALYSES_OUTPUT_FILENAME = "mpi_analyses.parquet"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _epoch_ms_to_datetime(epoch_ms: float) -> datetime:
    return _EPOCH + timedelta(milliseconds=epoch_ms)


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
    # Every distinct integration period the frames reported, as a JSON
    # array, e.g. "[260]".
    "integration_periods": pl.String,
    # Every distinct source of the time sync the frames' times were taken
    # under (step 6's `time_sync_source`: the firmware's letter for it, e.g.
    # "G" for GNSS over UART or "E" for the EPS's RTC), as a JSON array,
    # e.g. '["E","G"]'.
    "time_sync_sources": pl.String,
    # When the satellite's clock was last set (synced), as of the first and
    # the last frame received: each frame's `collected_at` minus step 6's
    # `ms_since_time_sync`.
    "time_synced_at": pl.Datetime("us", "UTC"),
    "last_time_synced_at": pl.Datetime("us", "UTC"),
    # How many times the clock was resynced during the recording, as seen in
    # the frames received: between consecutive frames, or (for a collection
    # timed by an analysis) between its start and the first frame received.
    # A resync after the last frame received can't be seen.
    "time_resyncs_during": pl.Int64,
    # How long the clock had gone since it was last set, at
    # `collection_started_at` -- or, when it was resynced between then and
    # the first frame received, at that first frame instead.
    "ms_since_time_sync_at_start": pl.Int64,
    # The range of the MPI's board temperature over the frames.
    "board_temperature_min_c": pl.Float64,
    "board_temperature_max_c": pl.Float64,
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
    # For each of `analysis_strong_signals`, in order: how many of its frames
    # were downlinked (and recovered) -- a JSON array, e.g. "[1,0]". See
    # `count_downlinked_strong_signal_frames`.
    "analysis_strong_signal_frames_downlinked": pl.String,
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
    # How many frame collections (see
    # `step_6_pipeline.assign_mpi_collections`) were merged into this one --
    # 0 for an analysis with no frames downlinked yet, more than 1 when one
    # file holds several recordings.
    "frame_collection_count": pl.Int64,
    # Where the satellite was at `collection_started_at`/
    # `collection_ended_at`, and whether it was in sunlight over that time --
    # see `orbit.py`. Null without a TLE from before the collection started.
    "start_latitude_deg": pl.Float64,
    "start_longitude_deg": pl.Float64,
    "end_latitude_deg": pl.Float64,
    "end_longitude_deg": pl.Float64,
    # The least/most of the Sun visible from the satellite over the
    # collection (sampled every `_SUN_SAMPLE_STEP`): 0 in Earth's umbra, 1 in
    # full sunlight.
    "sun_fraction_min": pl.Float64,
    "sun_fraction_max": pl.Float64,
    # "yes" (in full sunlight throughout), "no" (in the umbra throughout), or
    # "partly".
    "sunlit": pl.String,
    # The epoch of the TLE `collection_started_at` was propagated from.
    "orbit_tle_epoch": pl.Datetime("us", "UTC"),
    **_FILE_ANALYSES_SCHEMA,
}

# How far apart two frames' sync times (see `time_synced_at`) must be to
# count as a resync -- each is exact to the millisecond, give or take
# rounding.
RESYNC_TOLERANCE = timedelta(seconds=1)

# How often the satellite's lighting is sampled over a collection.
_SUN_SAMPLE_STEP = timedelta(seconds=1)

# The `action` of an `analyze_mpi_data` telecommand's response -- any
# version, e.g. "analyze_mpi_data_v1".
MPI_ANALYSIS_ACTION_PREFIX = "analyze_mpi_data"
_ANALYSIS_VERSION_RE = re.compile(r"_v(\d+)$")
# How far outside an MPI analysis's `earliest`..`latest` a frame collection
# can reach and still be that analysis's: frames are timed from the same
# timestamps, but may be extrapolated up to a block past them.
_ANALYSIS_MATCH_TOLERANCE = timedelta(seconds=60)  # Step 6's longest block.


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
            orjson.dumps(strong_signals).decode()
            if isinstance(strong_signals, list)
            else None
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
            properties = orjson.loads(text)
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


def count_downlinked_strong_signal_frames(
    all_analyses: pl.DataFrame, frames: pl.DataFrame
) -> pl.DataFrame:
    """`all_analyses` (from `extract_all_mpi_analyses`) with
    `analysis_strong_signal_frames_downlinked` filled in: for each strong
    signal, how many distinct frames in `frames` (`mpi_frames.parquet`) start
    within its `bytes` range and were collected during the analysis's
    `earliest`..`latest` (give or take `_ANALYSIS_MATCH_TOLERANCE`, as for
    merging collections) -- i.e. its frames that were downlinked and
    recovered.
    """
    if all_analyses.is_empty():
        return all_analyses
    counts: list[str | None] = []
    for started_at, ended_at, signals_json in all_analyses.select(
        "analysis_started_at", "analysis_ended_at", "analysis_strong_signals"
    ).iter_rows():
        signals = cast("list[Any]", orjson.loads(signals_json) if signals_json else [])
        window = (
            frames.filter(
                pl.col("collected_at").is_between(
                    started_at - _ANALYSIS_MATCH_TOLERANCE,
                    ended_at + _ANALYSIS_MATCH_TOLERANCE,
                )
            )
            if signals and not frames.is_empty()
            else None
        )
        row_counts: list[int | None] = []
        for signal in signals:
            byte_range = (
                cast("dict[str, Any]", signal).get("bytes")
                if isinstance(signal, dict)
                else None
            )
            if not (
                isinstance(byte_range, list)
                and len(byte_range := cast("list[object]", byte_range)) == 2  # noqa: PLR2004
                and (start := _int_field(byte_range[0])) is not None
                and (end := _int_field(byte_range[1])) is not None
            ):
                row_counts.append(None)
                continue
            row_counts.append(
                0
                if window is None
                else window.filter(
                    pl.col("file_offset").is_between(start, end, "left")
                )["frame_counter"].n_unique()
            )
        counts.append(orjson.dumps(row_counts).decode() if signals else None)
    return all_analyses.with_columns(
        analysis_strong_signal_frames_downlinked=pl.Series(counts, dtype=pl.String)
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
    `step_6_pipeline.assign_mpi_collections`), oldest first -- see
    `_FRAME_COLLECTIONS_SCHEMA`.
    """
    if frames.is_empty():
        return pl.DataFrame(schema=_FRAME_COLLECTIONS_SCHEMA)
    collection = "collection_started_at"
    gap_frames = pl.col("frame_counter").diff().over(collection) - 1
    has_gap = gap_frames > 0
    synced_at = pl.col("collected_at") - pl.duration(
        milliseconds=pl.col("ms_since_time_sync")
    )
    gaps = frames.sort(collection, "frame_counter").with_columns(
        synced_at=synced_at,
        is_resync=(synced_at - synced_at.shift().over(collection)) > RESYNC_TOLERANCE,
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
            integration_periods=pl.format(
                "[{}]",
                pl.col("integration_period")
                .drop_nulls()
                .unique()
                .sort()
                .cast(pl.String)
                .str.join(","),
            ),
            time_sync_sources=pl.format(
                "[{}]",
                pl.format(
                    '"{}"', pl.col("time_sync_source").drop_nulls().unique().sort()
                ).str.join(","),
            ),
            time_synced_at=pl.col("synced_at").drop_nulls().first(),
            last_time_synced_at=pl.col("synced_at").drop_nulls().last(),
            time_resyncs_during=pl.col("is_resync").sum(),
            board_temperature_min_c=pl.col("board_temperature_c").min(),
            board_temperature_max_c=pl.col("board_temperature_c").max(),
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
            frames_missing=pl.col("frames_expected") - pl.col("frames_present"),
            # A frame collection starts at its first frame.
            ms_since_time_sync_at_start=(
                pl.col(collection) - pl.col("time_synced_at")
            ).dt.total_milliseconds(),
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


def _merged_time_syncs(
    started_at: datetime, collections: list[dict[str, Any]]
) -> dict[str, Any]:
    """The time sync columns of frame collections (oldest first) merged into
    one collection starting at `started_at` -- see `_FRAME_COLLECTIONS_SCHEMA`.
    """
    synced = [c for c in collections if c["time_synced_at"] is not None]
    if not synced:
        return dict.fromkeys(
            (
                "time_synced_at",
                "last_time_synced_at",
                "time_resyncs_during",
                "ms_since_time_sync_at_start",
            )
        )
    first, last = synced[0], synced[-1]
    resyncs = sum(c["time_resyncs_during"] for c in synced) + sum(
        after["time_synced_at"] - before["last_time_synced_at"] > RESYNC_TOLERANCE
        for before, after in itertools.pairwise(synced)
    )
    # The clock was set after the recording started, but before its first
    # frame received: its age at the start is unknown, so it's given at that
    # first frame instead.
    resynced_first = first["time_synced_at"] - started_at > RESYNC_TOLERANCE
    age_from = first["collection_started_at"] if resynced_first else started_at
    return {
        "time_synced_at": first["time_synced_at"],
        "last_time_synced_at": last["last_time_synced_at"],
        "time_resyncs_during": resyncs + resynced_first,
        "ms_since_time_sync_at_start": max(
            0, round((age_from - first["time_synced_at"]).total_seconds() * 1000)
        ),
    }


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
        "integration_periods": orjson.dumps(
            sorted(
                {p for c in collections for p in orjson.loads(c["integration_periods"])}
            )
        ).decode(),
        "time_sync_sources": orjson.dumps(
            sorted(
                {s for c in collections for s in orjson.loads(c["time_sync_sources"])}
            )
        ).decode(),
        **_merged_time_syncs(analysis["analysis_started_at"], collections),
        "board_temperature_min_c": min(
            (t for c in collections if (t := c["board_temperature_min_c"]) is not None),
            default=None,
        ),
        "board_temperature_max_c": max(
            (t for c in collections if (t := c["board_temperature_max_c"]) is not None),
            default=None,
        ),
        **{
            key: largest_gap[key]
            for key in (
                "largest_gap_frames",
                "largest_gap_started_at",
                "largest_gap_duration_ms",
            )
        },
    }


def _sunlit(
    sun_fraction_min: float | None, sun_fraction_max: float | None
) -> str | None:
    if sun_fraction_min is None or sun_fraction_max is None:
        return None
    if sun_fraction_min >= 1:
        return "yes"
    return "no" if sun_fraction_max <= 0 else "partly"


def _with_orbit(collections: pl.DataFrame, tle_history: pl.DataFrame) -> pl.DataFrame:
    """`collections` with their orbit columns filled in: each start/end, and
    each `_SUN_SAMPLE_STEP` between, propagated from the TLE right before it.
    """
    starts = collections["collection_started_at"].to_list()
    ends = collections["collection_ended_at"].to_list()
    at_start = orbit.ephemeris(tle_history, starts)
    at_end = orbit.ephemeris(tle_history, ends)
    # Every sample time of every collection, start and end included.
    samples = [
        (i, t)
        for i, (start, end) in enumerate(zip(starts, ends, strict=True))
        for t in [
            *(
                start + k * _SUN_SAMPLE_STEP
                for k in range(int((end - start) / _SUN_SAMPLE_STEP))
            ),
            end,
        ]
    ]
    lighting = (
        orbit.ephemeris(tle_history, [t for _, t in samples])
        .with_columns(_collection=pl.Series([i for i, _ in samples], dtype=pl.Int64))
        .group_by("_collection")
        .agg(
            sun_fraction_min=pl.col("sun_fraction").min(),
            sun_fraction_max=pl.col("sun_fraction").max(),
        )
        .sort("_collection")
    )
    sun = pl.DataFrame({"_collection": range(collections.height)}).join(
        lighting, on="_collection", how="left"
    )
    return collections.with_columns(
        start_latitude_deg=at_start["latitude_deg"],
        start_longitude_deg=at_start["longitude_deg"],
        end_latitude_deg=at_end["latitude_deg"],
        end_longitude_deg=at_end["longitude_deg"],
        sun_fraction_min=sun["sun_fraction_min"],
        sun_fraction_max=sun["sun_fraction_max"],
        sunlit=pl.Series(
            [
                _sunlit(lo, hi)
                for lo, hi in zip(
                    sun["sun_fraction_min"], sun["sun_fraction_max"], strict=True
                )
            ],
            dtype=pl.String,
        ),
        orbit_tle_epoch=at_start["tle_epoch"],
    )


def summarize_mpi_collections(
    frames: pl.DataFrame,
    analyses: pl.DataFrame | None = None,
    tle_history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per MPI collection, oldest first -- see `_COLLECTIONS_SCHEMA`.

    Each frame collection in `frames` (as labelled by
    `step_6_pipeline.assign_mpi_collections`) that overlaps an MPI analysis
    in `analyses` (from `extract_mpi_analyses`) is merged into it: its start/end become
    the analysis's `earliest`/`latest` -- the timestamps in the whole file
    on board, rather than just whichever frames were downlinked -- and its
    file metadata comes along. Every analysis is a collection, even with no
    frames downlinked yet; every other frame collection stays as it was.

    Each collection is then placed in orbit from `tle_history` (step 7's
    `tle_history.parquet`) -- see `_with_orbit`. Without it, the orbit
    columns are null.
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
    collections = (
        pl.DataFrame(rows, schema=_COLLECTIONS_SCHEMA)
        .with_columns(
            collection_duration_ms=(
                pl.col("collection_ended_at") - pl.col("collection_started_at")
            ).dt.total_milliseconds()
        )
        .sort("collection_started_at")
    )
    if tle_history is None or tle_history.is_empty():
        return collections
    return _with_orbit(collections, tle_history)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `mpi_analyses.parquet` and `mpi_collections.parquet` from
    step 6's (and, if they're there, steps 5 and 7's) output.

    Reads `mpi_frames.parquet`, `reassembled_tcmd_responses.parquet` and
    `tle_history.parquet` from `data_dir` (where steps 6, 5 and 7 write
    them) and writes the result back into the same directory --
    parquet-in-parquet-out, no database involved.
    """
    frames_path = data_dir / step_6_pipeline.OUTPUT_FILENAME
    if not frames_path.exists():
        msg = f"{frames_path} not found -- run step_6 first."
        raise FileNotFoundError(msg)
    logger.info(f"Reading {frames_path}")
    frames = pl.read_parquet(frames_path)

    tcmd_path = data_dir / step_5_pipeline.OUTPUT_FILENAME
    if tcmd_path.exists():
        logger.info(f"Reading {tcmd_path}")
        all_analyses = extract_all_mpi_analyses(pl.read_parquet(tcmd_path))
    else:
        logger.warning(f"{tcmd_path} not found -- no MPI analyses (run step_5).")
        all_analyses = pl.DataFrame(schema=_ALL_ANALYSES_SCHEMA)
    all_analyses = count_downlinked_strong_signal_frames(all_analyses, frames)
    analyses_path = data_dir / ANALYSES_OUTPUT_FILENAME
    _write_parquet_atomic(all_analyses, analyses_path)

    tle_path = data_dir / step_7_pipeline.OUTPUT_FILENAME
    if tle_path.exists():
        logger.info(f"Reading {tle_path}")
        tle_history = pl.read_parquet(tle_path)
    else:
        logger.warning(f"{tle_path} not found -- no orbit positions (run step_7).")
        tle_history = None
    collections = summarize_mpi_collections(
        frames, primary_mpi_analyses(all_analyses), tle_history
    )
    collections_path = data_dir / COLLECTIONS_OUTPUT_FILENAME
    _write_parquet_atomic(collections, collections_path)

    timed_by_analysis = collections.filter(
        pl.col("collection_time_source") == "analysis"
    ).height
    logger.info(
        f"Done. {all_analyses.height:,} distinct MPI analysis response(s) "
        f"written to {analyses_path}; {collections.height:,} MPI "
        f"collection(s) written to {collections_path}, {timed_by_analysis:,} "
        "of them timed by an analysis."
    )
