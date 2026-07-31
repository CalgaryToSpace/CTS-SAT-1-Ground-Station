"""
cts1_satnogs_interval.py
=========================
Fetch upcoming SatNOGS observations for a satellite and print a merged
coverage summary showing overlapping time windows with station counts.

Usage (uv):
    uv run cts1_satnogs_interval 69015
    uv run cts1_satnogs_interval 69015 --hours 6
    uv run cts1_satnogs_interval 69015 --hours 6 --sort stations
    uv run cts1_satnogs_interval 69015 \\
        --start 2026-07-10T00:00:00Z --end 2026-07-12T00:00:00Z
    uv run cts1_satnogs_interval 69015 --sort stations --min-stations 3
"""

from __future__ import annotations

__all__: list[str] = [
    "Args",
    "CoverageWindow",
    "ObsInterval",
    "build_coverage_windows",
    "export_tsv",
    "fetch_all_observations",
    "format_detail",
    "format_summary",
]

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Final, Literal, TypeAlias

import polars as pl
import requests
import tyro
from loguru import logger

from .cts1_agenda_maker.satnogs_data import iter_future_observation_pages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COL_WIDTH: Final[int] = 80

SortOrder: TypeAlias = Literal["time", "stations"]
ObsStatus: TypeAlias = Literal["future", "good", "bad", "unknown", "failed"]

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObsInterval:
    """A single SatNOGS observation expressed as a closed time interval."""

    start: datetime
    end: datetime
    ground_station_id: int | str
    observation_id: int | str


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 string (Z or offset suffix) into an aware UTC datetime.

    Args:
        value: ISO 8601 datetime string, e.g. ``"2026-07-10T00:00:00Z"``.

    Returns:
        Timezone-aware datetime in UTC.

    Raises:
        ValueError: If the string cannot be parsed or carries no timezone info.
    """
    normalised = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalised)
    if dt.tzinfo is None:
        msg = f"Datetime string has no timezone: {value!r}"
        raise ValueError(msg)
    return dt.astimezone(UTC)


def _parse_obs_interval(obs: dict[str, Any]) -> ObsInterval:
    """Parse a raw SatNOGS observation dict into a typed ObsInterval.

    Args:
        obs: Raw observation dict as returned by the SatNOGS API.

    Returns:
        Typed ObsInterval.

    Raises:
        KeyError:   If required fields (``start``, ``end``) are absent.
        ValueError: If datetime fields cannot be parsed.
        TypeError:  If field values have unexpected types.
    """
    start_raw = obs["start"]
    end_raw = obs["end"]
    if not isinstance(start_raw, str):
        msg = f"Expected str for 'start', got {type(start_raw).__name__}"
        raise TypeError(msg)
    if not isinstance(end_raw, str):
        msg = f"Expected str for 'end', got {type(end_raw).__name__}"
        raise TypeError(msg)

    gs_id = obs.get("ground_station", "?")
    obs_id = obs.get("id", "?")
    return ObsInterval(
        start=_parse_iso(start_raw),
        end=_parse_iso(end_raw),
        ground_station_id=gs_id if isinstance(gs_id, (int, str)) else str(gs_id),
        observation_id=obs_id if isinstance(obs_id, (int, str)) else str(obs_id),
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_all_observations(
    norad_cat_id: str,
    *,
    start_gt: datetime,
    start_lt: datetime,
    status: ObsStatus = "future",
) -> list[dict[str, Any]]:
    """Fetch every observation page and return them as a single flat list.

    Delegates pagination to :func:`satnogs_data.iter_future_observation_pages`.

    Args:
        norad_cat_id: NORAD catalog ID of the target satellite.
        start_gt:     Fetch observations that start after this UTC datetime.
        start_lt:     Fetch observations that start before this UTC datetime.
        status:       SatNOGS observation status filter.

    Returns:
        Flat list of raw observation dicts.

    Raises:
        requests.HTTPError: On a non-2xx API response.
    """
    if status != "future":
        logger.warning(
            f"iter_future_observation_pages only queries 'future' observations; "
            f"ignoring requested status={status!r}."
        )

    obs: list[dict[str, Any]] = []
    for page in iter_future_observation_pages(
        norad_cat_id,
        start_gt_filter=start_gt,
        start_lt_filter=start_lt,
    ):
        obs.extend(page)
        logger.info(f"  {len(obs)} observations loaded so far…")
    return obs


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------


def _empty_observations() -> list[ObsInterval]:
    return []


@dataclass(slots=True)
class CoverageWindow:
    """A merged time window in which at least one ground station is listening.

    Attributes:
        start:        UTC start of the combined coverage window.
        end:          UTC end of the combined coverage window.
        observations: Every ObsInterval that contributes to this window.
    """

    start: datetime
    end: datetime
    observations: list[ObsInterval] = field(default_factory=_empty_observations)

    @property
    def duration(self) -> timedelta:
        """Wall-clock length of this coverage window."""
        return self.end - self.start

    @property
    def duration_str(self) -> str:
        """Human-readable duration, e.g. ``'23m 2s'``."""
        total = int(self.duration.total_seconds())
        return f"{total // 60}m {total % 60}s"

    @property
    def station_count(self) -> int:
        """Total distinct ground station IDs appearing anywhere in this window.
        This counts unique stations across the *whole* window duration, not
        the maximum active at any single moment.
        """
        return len({iv.ground_station_id for iv in self.observations})

    @property
    def peak_simultaneous_stations(self) -> int:
        """Maximum number of stations active at exactly the same instant.

        Uses a sweep-line over the individual observation intervals.  Events at
        the same timestamp are processed ends-before-starts so that a pass
        ending at T and a new pass starting at T are *not* counted as
        overlapping.

        Returns:
            Peak simultaneous count (0 for an empty window).
        """
        if not self.observations:
            return 0

        events: list[tuple[datetime, int]] = []
        for iv in self.observations:
            events.append((iv.start, 1))
            events.append((iv.end, -1))
        # -1 sorts before +1 at the same timestamp → end processed before start
        events.sort(key=lambda e: (e[0], e[1]))

        peak = current = 0
        for _, delta in events:
            current += delta
            peak = max(peak, current)
        return peak

    @property
    def obs_ids(self) -> list[int | str]:
        """Observation IDs of all contributing observations."""
        return [iv.observation_id for iv in self.observations]

    def format_start(self) -> str:
        """Full UTC start string, e.g. ``'2026-07-11T06:08:25'``."""
        return self.start.strftime("%Y-%m-%dT%H:%M:%S")

    def format_end(self, *, short: bool = True) -> str:
        """UTC end string, shortened to ``HH:MM:SS`` when on the same UTC date."""
        if short and self.start.date() == self.end.date():
            return self.end.strftime("%H:%M:%S")
        return self.end.strftime("%Y-%m-%dT%H:%M:%S")


def build_coverage_windows(
    observations: list[dict[str, Any]],
) -> list[CoverageWindow]:
    """Merge a flat list of SatNOGS observation dicts into non-overlapping
    CoverageWindows.

    Uses a sweep-line algorithm: observations are sorted by start time and
    merged greedily while they overlap or touch.  Chain overlaps (A∩B and B∩C
    but not A∩C) correctly collapse into one continuous window.

    Malformed observations are logged as warnings and skipped.

    Args:
        observations: Raw observation dicts as returned by the SatNOGS API.

    Returns:
        Chronologically sorted list of non-overlapping CoverageWindow objects.
    """
    intervals: list[ObsInterval] = []
    for obs in observations:
        try:
            intervals.append(_parse_obs_interval(obs))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Skipping malformed observation {obs.get('id')!r}: {exc}")

    if not intervals:
        return []

    intervals.sort(key=lambda iv: iv.start)

    windows: list[CoverageWindow] = []
    cur = CoverageWindow(
        start=intervals[0].start,
        end=intervals[0].end,
        observations=[intervals[0]],
    )

    for iv in intervals[1:]:
        if iv.start <= cur.end:
            cur.end = max(cur.end, iv.end)
            cur.observations.append(iv)
        else:
            windows.append(cur)
            cur = CoverageWindow(start=iv.start, end=iv.end, observations=[iv])

    windows.append(cur)
    return windows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _divider(char: str = "─") -> str:
    return char * _COL_WIDTH


def _filter_and_sort(
    windows: list[CoverageWindow],
    *,
    sort_by: SortOrder,
    min_stations: int,
) -> list[CoverageWindow]:
    """Return windows that meet min_stations, ordered by sort_by."""
    filtered = [w for w in windows if w.peak_simultaneous_stations >= min_stations]
    if sort_by == "stations":
        filtered.sort(key=lambda w: (-w.peak_simultaneous_stations, w.start))
    else:
        filtered.sort(key=lambda w: w.start)
    return filtered


def format_summary(
    windows: list[CoverageWindow],
    *,
    sort_by: SortOrder = "time",
    min_stations: int = 1,
    show_obs_ids: bool = False,
) -> str:
    """Format merged coverage windows as a polars table.

    Args:
        windows:       Merged CoverageWindow objects to display.
        sort_by:       ``"time"`` for chronological; ``"stations"`` for
                       descending station count (ties broken chronologically).
        min_stations:  Exclude windows below this peak simultaneous count.
        show_obs_ids:  Append an ``Obs IDs`` column to the table.

    Returns:
        Multi-line formatted string, ready for :func:`print`.
    """
    filtered = _filter_and_sort(windows, sort_by=sort_by, min_stations=min_stations)
    if not filtered:
        return "(no coverage windows match the filter criteria)"

    rows: list[dict[str, str | int]] = []
    for w in filtered:
        row: dict[str, str | int] = {
            "Stations": w.station_count,
            "Start (UTC)": w.format_start(),
            "End (UTC)": w.format_end(),
            "Duration": w.duration_str,
        }
        if show_obs_ids:
            row["Obs IDs"] = ", ".join(str(i) for i in w.obs_ids)
        rows.append(row)

    df = pl.DataFrame(rows)

    total_coverage: timedelta = sum((w.duration for w in filtered), timedelta())
    total_sec = int(total_coverage.total_seconds())
    max_peak = max(w.peak_simultaneous_stations for w in filtered)

    with pl.Config(
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_cell_alignment="CENTER",
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        table_str = str(df)

    footer = (
        f"  {len(filtered)} window(s)  |  "
        f"Total coverage: "
        f"{total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s  |  "
        f"{max_peak} simultaneous station(s)"
    )
    return (
        f"Combined SatNOGS Coverage Windows (UTC)\n{_divider()}\n{table_str}\n{footer}"
    )


def format_detail(
    windows: list[CoverageWindow],
    *,
    sort_by: SortOrder = "time",
) -> str:
    """Format a verbose per-observation breakdown under each coverage window.

    Args:
        windows: Merged CoverageWindow objects to display.
        sort_by: Same sort semantics as :func:`format_summary`.

    Returns:
        Multi-line formatted string, ready for :func:`print`.
    """
    if sort_by == "stations":
        sorted_windows = sorted(
            windows, key=lambda w: (-w.peak_simultaneous_stations, w.start)
        )
    else:
        sorted_windows = sorted(windows, key=lambda w: w.start)

    lines: list[str] = [_divider("═"), "Detailed Coverage Breakdown", _divider("─")]

    for idx, w in enumerate(sorted_windows, start=1):
        lines.append(
            f"Window {idx:>3}  |  up to {w.peak_simultaneous_stations} stn (peak)  |  "
            f"{w.format_start()} → {w.format_end()}  |  {w.duration_str}"
        )
        for iv in sorted(w.observations, key=lambda o: o.start):
            dur_s = int((iv.end - iv.start).total_seconds())
            lines.append(
                f"          Obs {iv.observation_id!s:<8}  "
                f"GS {iv.ground_station_id!s:<5}  "
                f"{iv.start.strftime('%H:%M:%SZ')} → {iv.end.strftime('%H:%M:%SZ')}  "
                f"({dur_s // 60}m {dur_s % 60}s)"
            )

    lines.append(_divider("═"))
    return "\n".join(lines)


def export_tsv(
    windows: list[CoverageWindow],
    *,
    sort_by: SortOrder = "time",
    min_stations: int = 1,
    output_path: str = "output/satnogs.tsv",
) -> None:
    """Export coverage windows as a TSV file for spreadsheet import.

    Args:
        windows: Coverage windows to export.
        sort_by: Sort order applied before writing rows.
        min_stations: Minimum station count to include.
        output_path: Destination TSV file path.
    """
    filtered = _filter_and_sort(
        windows,
        sort_by=sort_by,
        min_stations=min_stations,
    )

    if not filtered:
        logger.warning("No coverage windows match the filter criteria.")
        return

    rows: list[dict[str, str | int]] = [
        {
            "total_stations": w.station_count,
            "window_start_utc": w.format_start(),
            "window_end_utc": w.format_end(),
            "duration": w.duration_str,
        }
        for w in filtered
    ]

    df = pl.DataFrame(rows)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.write_csv(
        output_file,
        separator="\t",
    )

    logger.info(f"Saved SatNOGS TSV file: {output_path}")


# ---------------------------------------------------------------------------
# CLI  (tyro reads field docstrings as --help text)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Args:
    """Fetch SatNOGS observations and summarise overlapping coverage windows."""

    norad_id: Annotated[str, tyro.conf.Positional]
    """NORAD catalog ID of the satellite (e.g. 69015 for CTS-SAT-1)."""

    start: str | None = None
    """Start of the search window (ISO 8601 + timezone). Defaults to now."""

    end: str | None = None
    """End of the search window (ISO 8601 + timezone). Mutually exclusive with
    --hours."""

    hours: float = 24.0
    """Hours ahead to search from --start. Ignored when --end is provided."""

    sort: SortOrder = "time"
    """Sort by 'time' (chronological) or 'stations' (descending peak simultaneous)."""

    min_stations: int = 1
    """Omit windows whose peak simultaneous station count is below this threshold."""

    detail: bool = False
    """Print a per-observation breakdown after the summary table."""

    obs_ids: bool = False
    """Add an Obs IDs column to the summary table."""

    tsv: bool = False
    """Export coverage windows to output/satnogs.tsv."""

    debug: bool = False
    """Enable debug logging (shows API page URLs)."""


def main() -> None:
    """Entry point: parse CLI, fetch observations, print coverage summary."""
    args = tyro.cli(Args)

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    start_dt = (
        _parse_iso(args.start) if args.start is not None else datetime.now(tz=UTC)
    )
    end_dt = (
        _parse_iso(args.end)
        if args.end is not None
        else start_dt + timedelta(hours=args.hours)
    )

    if end_dt <= start_dt:
        logger.error("--end must be strictly after --start")
        sys.exit(1)

    total_hours = (end_dt - start_dt).total_seconds() / 3600
    sort_by: SortOrder = args.sort
    logger.info(
        f"Fetching future observations for NORAD {args.norad_id} "
        f"over {total_hours:.1f}h window"
    )
    logger.info(f"  From: {start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info(f"  To:   {end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}")

    try:
        observations = fetch_all_observations(
            args.norad_id,
            start_gt=start_dt,
            start_lt=end_dt,
        )
    except requests.HTTPError as exc:
        logger.error(f"HTTP error fetching observations: {exc}")
        sys.exit(1)

    logger.info(f"Fetched {len(observations)} total observations.")

    if not observations:
        print("\nNo observations found in the specified window.")  # noqa: T201
        sys.exit(0)

    windows = build_coverage_windows(observations)
    logger.info(f"Merged into {len(windows)} coverage window(s).")

    if args.tsv:
        export_tsv(
            windows,
            sort_by=sort_by,
            min_stations=args.min_stations,
        )

    print(  # noqa: T201
        f"""\n{
            format_summary(
                windows,
                sort_by=sort_by,
                min_stations=args.min_stations,
                show_obs_ids=args.obs_ids,
            )
        }"""
    )

    if args.detail:
        filtered = [
            w for w in windows if w.peak_simultaneous_stations >= args.min_stations
        ]
        print(f"\n{format_detail(filtered, sort_by=sort_by)}")  # noqa: T201


if __name__ == "__main__":
    main()
