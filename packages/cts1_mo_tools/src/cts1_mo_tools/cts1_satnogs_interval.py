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
    "CoverageWindow",
    "ObsInterval",
    "build_coverage_windows",
    "fetch_all_observations",
    "format_detail",
    "format_summary",
]

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

import requests
from loguru import logger

from .cts1_agenda_maker.satnogs_data import iter_future_observation_pages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COL_WIDTH: Final[int] = 80

SortOrder = Literal["time", "stations"]
ObsStatus = Literal["future", "good", "bad", "unknown", "failed"]

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
    # iter_future_observation_pages only supports "future" status; for other
    # statuses we would need a different helper, so log a warning.
    if status != "future":
        logger.warning(
            f"satnogs_data.iter_future_observation_pages only queries 'future' "
            f"observations; ignoring requested status={status!r}."
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
        """Number of distinct ground stations active within this window."""
        return len({iv.ground_station_id for iv in self.observations})

    @property
    def obs_ids(self) -> list[int | str]:
        """Observation IDs of all contributing observations."""
        return [iv.observation_id for iv in self.observations]

    def format_start(self) -> str:
        """Full UTC start string, e.g. ``'2026-07-11T06:08:25Z'``."""
        return self.start.strftime("%Y-%m-%dT%H:%M:%SZ")

    def format_end(self, *, short: bool = True) -> str:
        """UTC end string, shortened to ``HH:MM:SSZ`` when on the same UTC date."""
        if short and self.start.date() == self.end.date():
            return self.end.strftime("%H:%M:%SZ")
        return self.end.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def format_summary(
    windows: list[CoverageWindow],
    *,
    sort_by: SortOrder = "time",
    min_stations: int = 1,
    show_obs_ids: bool = False,
) -> str:
    """Format merged coverage windows as a human-readable summary table.

    Args:
        windows:       Merged CoverageWindow objects to display.
        sort_by:       ``"time"`` for chronological order; ``"stations"`` for
                       descending station count (ties broken chronologically).
        min_stations:  Exclude windows with fewer than this many stations.
        show_obs_ids:  Append contributing observation IDs to each output line.

    Returns:
        Multi-line formatted string, ready for :func:`print`.
    """
    filtered = [w for w in windows if w.station_count >= min_stations]
    if not filtered:
        return "(no coverage windows match the filter criteria)"

    if sort_by == "stations":
        filtered.sort(key=lambda w: (-w.station_count, w.start))
    else:
        filtered.sort(key=lambda w: w.start)

    total_coverage: timedelta = sum((w.duration for w in filtered), timedelta())
    max_stations = max(w.station_count for w in filtered)

    lines: list[str] = [
        _divider("═"),
        "Combined SatNOGS Coverage Windows (overlapping passes merged, UTC)",
        _divider("─"),
    ]
    for w in filtered:
        station_label = (
            f"{w.station_count:>3} station{'s' if w.station_count != 1 else ' '}"
        )
        line = (
            f"  {station_label}  |  "
            f"{w.format_start()} → {w.format_end()}  |  "
            f"Duration: {w.duration_str}"
        )
        if show_obs_ids:
            line += f"  (Obs: {', '.join(str(i) for i in w.obs_ids)})"
        lines.append(line)

    total_sec = int(total_coverage.total_seconds())
    lines += [
        _divider("─"),
        (
            f"  {len(filtered)} window(s)  |  "
            f"Total coverage: "
            f"{total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s  |  "
            f"Max simultaneous stations: {max_stations}"
        ),
        _divider("═"),
    ]
    return "\n".join(lines)


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
        sorted_windows = sorted(windows, key=lambda w: (-w.station_count, w.start))
    else:
        sorted_windows = sorted(windows, key=lambda w: w.start)

    lines: list[str] = [_divider("═"), "Detailed Coverage Breakdown", _divider("─")]

    for idx, w in enumerate(sorted_windows, start=1):
        lines.append(
            f"Window {idx:>3}  |  {w.station_count} station(s)  |  "
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dt_arg(value: str) -> datetime:
    """argparse type converter for ISO 8601 datetime strings or the literal ``'now'``.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed.
    """
    if value.lower() == "now":
        return datetime.now(tz=UTC)
    try:
        return _parse_iso(value)
    except ValueError as exc:
        msg = (
            f"Cannot parse datetime {value!r}: {exc}. "
            "Use ISO 8601 with timezone, e.g. 2026-07-10T00:00:00Z"
        )
        raise argparse.ArgumentTypeError(msg) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cts1_satnogs_interval",
        description=(
            "Fetch SatNOGS observations for a satellite and display merged "
            "coverage windows with station counts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "norad_id",
        metavar="NORAD_ID",
        help="NORAD catalog ID of the satellite (e.g. 69015 for CTS-SAT-1).",
    )
    parser.add_argument(
        "--start",
        metavar="ISO_DATETIME",
        type=_parse_dt_arg,
        default=None,
        help=(
            "Start of the search window (ISO 8601 with timezone, or 'now'). "
            "Default is now."
        ),
    )
    parser.add_argument(
        "--end",
        metavar="ISO_DATETIME",
        type=_parse_dt_arg,
        default=None,
        help="End of the search window. Mutually exclusive with --hours.",
    )
    parser.add_argument(
        "--hours",
        metavar="N",
        type=float,
        default=24.0,
        help=(
            "Search N hours ahead from --start (default: 24). Ignored when --end is set"
        ),
    )
    parser.add_argument(
        "--sort",
        choices=("time", "stations"),
        default="time",
        dest="sort_by",
        help=(
            "Sort by 'time' (chronological, default) or "
            "'stations' (descending station count)."
        ),
    )
    parser.add_argument(
        "--min-stations",
        metavar="N",
        type=int,
        default=1,
        help="Only show windows with at least N stations (default: 1).",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also print a per-observation breakdown under each window.",
    )
    parser.add_argument(
        "--obs-ids",
        action="store_true",
        help="Append contributing observation IDs to each summary line.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows API page URLs).",
    )
    return parser


def main() -> None:
    """Entry point: parse arguments, fetch observations, and print coverage summary."""
    parser = _build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    start_dt: datetime = args.start if args.start is not None else datetime.now(tz=UTC)
    end_dt: datetime = (
        args.end if args.end is not None else start_dt + timedelta(hours=args.hours)
    )

    if end_dt <= start_dt:
        logger.error("--end must be strictly after --start")
        sys.exit(1)

    total_hours = (end_dt - start_dt).total_seconds() / 3600
    sort_by: SortOrder = args.sort_by
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

    print(  # noqa: T201
        f"\n{format_summary(windows, sort_by=sort_by, min_stations=args.min_stations, show_obs_ids=args.obs_ids)}"  # noqa: E501
    )

    if args.detail:
        filtered = [w for w in windows if w.station_count >= args.min_stations]
        print(f"\n{format_detail(filtered, sort_by=sort_by)}")  # noqa: T201


if __name__ == "__main__":
    main()
