"""How an MPI collection's fields (one row of `mpi_collections.parquet`) are
shown: grouped, labelled and formatted -- shared by the MPI Data page's
collections table and the MPI Vis page's metadata panel, so both read alike.

Each group's `columns` are the table's; its `details` are only in the
metadata panel, which shows every field.
"""

from __future__ import annotations

__all__ = [
    "COLUMN_GROUPS",
    "Column",
    "ColumnGroup",
    "cell_text",
    "collection_cells",
    "other_fields",
]

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)
from cts1_mo_tools.time_str import duration_str

from . import mpi_collections

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Column:
    # A field of the row, or (starting "_") one computed from `sources`.
    field: str
    label: str
    numeric: bool = False
    # The row's fields a computed column shows in full -- not listed again.
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ColumnGroup:
    label: str
    columns: tuple[Column, ...]
    details: tuple[Column, ...] = ()


COLUMN_GROUPS: tuple[ColumnGroup, ...] = (
    ColumnGroup(
        "Collection",
        (
            Column("collection_started_at", "Started (UTC)"),
            Column("collection_ended_at", "Ended (UTC)"),
            Column("collection_duration_ms", "Duration", numeric=True),
            Column("collection_time_source", "Times From"),
        ),
        (Column("frame_collection_count", "Frame Collections Merged", numeric=True),),
    ),
    ColumnGroup(
        "Orbit (SGP4)",
        (
            Column(
                "_start_position",
                "Start Lat/Lon",
                sources=("start_latitude_deg", "start_longitude_deg"),
            ),
            Column(
                "_end_position",
                "End Lat/Lon",
                sources=("end_latitude_deg", "end_longitude_deg"),
            ),
            Column("sunlit", "Sunlit"),
        ),
        (
            Column("sun_fraction_min", "Sun Fraction Min", numeric=True),
            Column("sun_fraction_max", "Sun Fraction Max", numeric=True),
            Column("orbit_tle_epoch", "TLE Epoch (UTC)"),
        ),
    ),
    ColumnGroup(
        "On-Board Analysis (analyze_mpi_data)",
        (
            Column("analysis_file", "File"),
            Column("analysis_file_size", "File Size", numeric=True),
            Column("analysis_frame_count", "Frames", numeric=True),
            Column("analysis_strong_signal_counts", "Strong Signals"),
            Column("analysis_uart_lost_bytes", "UART Lost", numeric=True),
            Column("analysis_reason_for_stopping", "Stop Reason"),
            Column("analysis_count", "Analyses", numeric=True),
            Column("analysis_sha256", "SHA-256"),
            Column("analysis_received_at", "Received (UTC)"),
        ),
        (
            Column("analysis_started_at", "Earliest (UTC)"),
            Column("analysis_ended_at", "Latest (UTC)"),
            Column("analysis_valid_frame_count", "Valid Frames", numeric=True),
            Column("analysis_bad_frame_count", "Bad Frames", numeric=True),
            Column(
                "analysis_background_frame_count", "Background Frames", numeric=True
            ),
            Column("analysis_warmup_frame_count", "Warm-Up Frames", numeric=True),
            Column("analysis_first_frame_byte", "First Frame Byte", numeric=True),
            Column("analysis_last_frame_byte", "Last Frame Byte", numeric=True),
            Column("analysis_time_sync_count", "Time Syncs", numeric=True),
            Column("analysis_mpi_start_count", "MPI Starts", numeric=True),
            Column(
                "analysis_malformed_time_sync_count",
                "Malformed Time Syncs",
                numeric=True,
            ),
            Column(
                "analysis_strong_threshold_dn", "Strong Threshold (DN)", numeric=True
            ),
            Column("analysis_strong_frame_count", "Strong Frames", numeric=True),
            Column("analysis_strong_signal_count", "Strong Signal Count", numeric=True),
            Column("analysis_strong_signals", "Strong Signals (JSON)"),
            Column("analysis_action", "Action"),
            Column("analysis_version", "Version", numeric=True),
            Column("analysis_tcmd_sent_at", "Telecommand Sent (UTC)"),
        ),
    ),
    ColumnGroup(
        "Frames Received (Downlinked)",
        (
            Column("frames_present", "Present", numeric=True),
            Column("frames_expected", "Expected", numeric=True),
            Column("frames_missing", "Missing", numeric=True),
            Column("_downlinked_percent", "Downlinked", numeric=True),
            Column("integration_periods", "Integration Period(s)", numeric=True),
            Column("time_sync_sources", "Time Sync Source(s)", numeric=True),
            Column("_time_sync_age", "Time Sync Age", numeric=True),
            Column(
                "_temperature_range",
                "MPI Temp (°C)",
                numeric=True,
                sources=("board_temperature_min_c", "board_temperature_max_c"),
            ),
            Column("first_frame_counter", "First Counter", numeric=True),
            Column("last_frame_counter", "Last Counter", numeric=True),
            Column("largest_gap_frames", "Largest Gap (Frames)", numeric=True),
            Column("largest_gap_started_at", "Largest Gap Start (UTC)"),
            Column("largest_gap_duration_ms", "Largest Gap Duration", numeric=True),
        ),
        (
            Column("time_synced_at", "Time Synced At (UTC)"),
            Column("last_time_synced_at", "Last Time Synced At (UTC)"),
            Column("time_resyncs_during", "Resyncs During", numeric=True),
            Column("ms_since_time_sync_at_start", "Time Sync Age (ms)", numeric=True),
        ),
    ),
)


def _seconds(duration_ms: int) -> str:
    return f"{duration_ms / 1000:,.2f} s"


# How the fields that need more than `str()` are shown.
_CELL_FORMATTERS: dict[str, Callable[[Any], Any]] = {
    "collection_duration_ms": lambda duration_ms: duration_str(duration_ms // 1000),
    "largest_gap_duration_ms": _seconds,
    "analysis_file_size": lambda size: f"{size:,} B",
    "analysis_uart_lost_bytes": lambda size: f"{size:,} B",
    "analysis_sha256": lambda sha256: sha256[:12],
    "analysis_strong_signal_counts": mpi_collections.strong_signal_counts_text,
    "integration_periods": mpi_collections.integration_periods_text,
    "time_sync_sources": mpi_collections.time_sync_sources_text,
}


def cell_text(field: str, value: Any) -> Any:
    """`value` (of `field`) as shown -- "" for a null."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"{value:%Y-%m-%d %H:%M:%S.%f}"[:-3]
    formatter = _CELL_FORMATTERS.get(field)
    return formatter(value) if formatter else value


def _temperature_range_cell(row: dict[str, Any]) -> str:
    low, high = row.get("board_temperature_min_c"), row.get("board_temperature_max_c")
    if low is None or high is None:
        return ""
    return f"{low:.1f}" if low == high else f"{low:.1f} to {high:.1f}"


def _time_sync_age_cell(row: dict[str, Any]) -> str:
    """How long since the clock was last set, at the recording's start --
    e.g. "16m 5s" -- flagged if it was resynced during the recording, when
    the age is at the first frame received instead if that's where it was.
    """
    age_ms = row.get("ms_since_time_sync_at_start")
    if age_ms is None:
        return ""
    age = duration_str(age_ms // 1000)
    resyncs = row.get("time_resyncs_during") or 0
    if not resyncs:
        return age
    synced_at, started_at = row.get("time_synced_at"), row["collection_started_at"]
    resynced_first = (
        synced_at is not None
        and synced_at - started_at > step_8_pipeline.RESYNC_TOLERANCE
    )
    at = "first frame" if resynced_first else "start"
    times = "once" if resyncs == 1 else f"{resyncs} times"
    return f"⚠ resynced {times} during; {age} at {at}"


def _time_source_cell(row: dict[str, Any]) -> str:
    """ "Times From": the analysis's full action name when it's from one --
    e.g. "analyze_mpi_data_v1".
    """
    source = row.get("collection_time_source") or ""
    action = row.get("analysis_action")
    return action if source == "analysis" and action else source


def collection_cells(row: dict[str, Any]) -> dict[str, Any]:
    """Every group's columns and details, by field, as shown for `row` --
    "" where it has no value. `.get`: an `mpi_collections.parquet` from
    before a field was added hasn't it.
    """
    cells = {
        column.field: cell_text(column.field, row.get(column.field))
        for group in COLUMN_GROUPS
        for column in (*group.columns, *group.details)
    }
    return cells | {
        "collection_time_source": _time_source_cell(row),
        "_start_position": mpi_collections.position_text(
            row.get("start_latitude_deg"), row.get("start_longitude_deg")
        ),
        "_end_position": mpi_collections.position_text(
            row.get("end_latitude_deg"), row.get("end_longitude_deg")
        ),
        "_temperature_range": _temperature_range_cell(row),
        "_time_sync_age": _time_sync_age_cell(row),
        "_downlinked_percent": mpi_collections.downlinked_percent_text(row),
    }


def other_fields(row: dict[str, Any]) -> dict[str, Any]:
    """`row`'s non-null fields no group shows -- e.g. one step 8 has added
    since the groups were last updated -- so nothing's left out.
    """
    shown = {
        name
        for group in COLUMN_GROUPS
        for column in (*group.columns, *group.details)
        for name in (column.field, *column.sources)
    }
    return {k: v for k, v in row.items() if k not in shown and v is not None}
