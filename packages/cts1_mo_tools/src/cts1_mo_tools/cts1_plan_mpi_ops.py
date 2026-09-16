"""
Plan MPI data-collection telecommands over geographic regions of interest.

Fetches the latest TLE for a satellite from CelesTrak, propagates the orbit with
satkit (SGP4), finds when the satellite is inside the configured lat/lon boxes,
and schedules repeated MPI start/stop telecommands while it's inside them.

Usage (uv):
    uv run cts1_plan_mpi_ops --duration-hours 24
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import requests
import satkit as sk
import tyro
from loguru import logger

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

TIME_STEP_SEC = 1.0
"""Propagation step used to detect region entry/exit."""


@dataclass(frozen=True, slots=True)
class Args:
    """Plan MPI start/stop telecommands for when the satellite is over a region."""

    norad_id: int = 69015
    """NORAD catalog ID of the satellite (69015 is CTS-SAT-1/FrontierSat)."""

    north_latitude_range: str | None = "55.0 to 65.0"
    """Latitude range (degrees) of the northern region, like "55.0 to 65.0"."""

    north_longitude_range: str | None = "-90 to -120"
    """Longitude range (degrees) of the northern region, like "-90 to -120"."""

    south_latitude_range: str | None = None
    """Latitude range (degrees) of the southern region. None to disable/unbound."""

    south_longitude_range: str | None = None
    """Longitude range (degrees) of the southern region. None to disable/unbound."""

    mpi_collection_duration_sec: int | float = 60
    """Duration of each MPI recording."""

    mpi_break_duration_sec: int | float = 5
    """Gap between stopping one MPI recording and starting the next."""

    mpi_min_collection_duration_sec: int | float = 10
    """Skip recordings shorter than this (when truncated by leaving the region)."""

    start_time: str | None = None
    """ISO-8601 start of the planning window (with timezone). Default: now."""

    duration_hours: float = 24.0
    """Length of the planning window."""

    tle_file: Path | None = None
    """Optional local TLE file to use instead of fetching from CelesTrak."""

    mpi_filename_format: str = "%Y-%m-%d_%H%M%SZ.mpi"
    """strftime format of the MPI filename passed to mpi_enable_active_mode."""

    output_csv: Path | None = None
    """Optional path to write the resulting plan as a CSV."""


@dataclass(frozen=True, slots=True)
class Region:
    name: str
    lat_range: tuple[float, float] | None
    lon_range: tuple[float, float] | None


def _parse_range(range_str: str | None) -> tuple[float, float] | None:
    """Parse a string like "-90 to -120" into a sorted (min, max) tuple."""
    if range_str is None:
        return None

    parts = range_str.lower().split("to")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f'Range must look like "<a> to <b>", got: {range_str!r}'
        raise ValueError(msg)

    a, b = (float(p.strip()) for p in parts)
    return (min(a, b), max(a, b))


def _parse_start_time(start_time: str | None) -> datetime:
    if start_time is None:
        return datetime.now(UTC).replace(microsecond=0)

    val = datetime.fromisoformat(start_time)
    if val.tzinfo is None:
        msg = f"Please specify a timezone offset in the start time: {start_time}"
        raise ValueError(msg)
    return val.astimezone(UTC)


def fetch_tle_lines(norad_id: int) -> list[str]:
    """Fetch the latest TLE for a satellite from CelesTrak."""
    response = requests.get(
        CELESTRAK_GP_URL,
        params={"CATNR": str(norad_id), "FORMAT": "tle"},
        timeout=10,
    )
    response.raise_for_status()
    text = response.text.strip()
    if not text or "No GP data found" in text:
        msg = f"No TLE found on CelesTrak for NORAD ID {norad_id}."
        raise ValueError(msg)
    return text.splitlines()


def propagate_lat_lon(
    tle: sk.TLE, times: list[datetime]
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the TLE to each time; return (latitude_deg, longitude_deg) arrays."""
    sk_times = [sk.time.from_datetime(t) for t in times]
    pos_teme, _ = sk.sgp4(tle, sk_times)  # pyright: ignore[reportUnknownMemberType]
    pos_teme = np.atleast_2d(pos_teme)

    quats = sk.frametransform.qteme2itrf(sk_times)
    assert isinstance(quats, list)

    coords = [sk.itrfcoord(q * p) for q, p in zip(quats, pos_teme, strict=True)]
    lat = np.array([c.latitude_deg for c in coords])
    lon = np.array([c.longitude_deg for c in coords])
    return lat, lon


def _in_region_mask(region: Region, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    mask = np.ones(lat.shape, dtype=bool)
    if region.lat_range is not None:
        mask &= (lat >= region.lat_range[0]) & (lat <= region.lat_range[1])
    if region.lon_range is not None:
        mask &= (lon >= region.lon_range[0]) & (lon <= region.lon_range[1])
    return mask


def _mask_to_intervals(
    mask: np.ndarray, times: list[datetime]
) -> list[tuple[datetime, datetime]]:
    """Convert a boolean mask over sample times into (enter, exit) intervals."""
    intervals: list[tuple[datetime, datetime]] = []
    enter_idx: int | None = None
    for i, inside in enumerate(mask):
        if inside and enter_idx is None:
            enter_idx = i
        elif not inside and enter_idx is not None:
            # Exit at the last sample that was still inside.
            intervals.append((times[enter_idx], times[i - 1]))
            enter_idx = None
    if enter_idx is not None:
        intervals.append((times[enter_idx], times[-1]))
    return intervals


def plan_mpi_events(
    intervals: list[tuple[str, datetime, datetime]],
    *,
    collection: timedelta,
    break_duration: timedelta,
    min_collection: timedelta,
    filename_format: str,
) -> list[dict[str, object]]:
    """Schedule repeated start/stop pairs within each in-region interval.

    The last recording in an interval is cut short at the region exit, and
    skipped entirely if that leaves it shorter than `min_collection`.
    """
    events: list[dict[str, object]] = []
    for region_name, enter, exit_ in intervals:
        start = enter
        while start < exit_:
            stop = min(start + collection, exit_)
            if stop - start < min_collection:
                break
            filename = start.strftime(filename_format)
            events.append(
                {
                    "timestamp": start,
                    "command": f"CTS1+mpi_enable_active_mode({filename})",
                    "region": region_name,
                }
            )
            events.append(
                {
                    "timestamp": stop,
                    "command": "CTS1+mpi_disable_active_mode()",
                    "region": region_name,
                }
            )
            start = stop + break_duration
    return events


def run(args: Args) -> pl.DataFrame:
    logger.info(
        "Input settings:\n"
        + "\n".join(f"  {k} = {getattr(args, k)!r}" for k in args.__dataclass_fields__)
    )

    regions = [
        Region(
            name="north",
            lat_range=_parse_range(args.north_latitude_range),
            lon_range=_parse_range(args.north_longitude_range),
        ),
        Region(
            name="south",
            lat_range=_parse_range(args.south_latitude_range),
            lon_range=_parse_range(args.south_longitude_range),
        ),
    ]
    # A region with neither bound set is disabled (otherwise it'd be the whole globe).
    regions = [r for r in regions if r.lat_range is not None or r.lon_range is not None]
    if not regions:
        msg = "At least one region must have a latitude or longitude range."
        raise ValueError(msg)

    if args.mpi_collection_duration_sec <= 0:
        msg = "mpi_collection_duration_sec must be positive."
        raise ValueError(msg)
    if args.mpi_break_duration_sec < 0:
        msg = "mpi_break_duration_sec must not be negative."
        raise ValueError(msg)

    if args.tle_file is not None:
        tle_lines = args.tle_file.read_text().strip().splitlines()
    else:
        tle_lines = fetch_tle_lines(args.norad_id)
    tle = sk.TLE.from_lines(tle_lines)
    if isinstance(tle, list):
        tle = tle[0]
    logger.info(f"Using TLE (epoch {tle.epoch}):\n" + "\n".join(tle_lines))

    start = _parse_start_time(args.start_time)
    num_steps = int(args.duration_hours * 3600 / TIME_STEP_SEC) + 1
    times = [start + timedelta(seconds=i * TIME_STEP_SEC) for i in range(num_steps)]
    logger.info(f"Propagating {num_steps} steps from {start} ({args.duration_hours} h)")

    lat, lon = propagate_lat_lon(tle, times)

    intervals: list[tuple[str, datetime, datetime]] = []
    for region in regions:
        region_intervals = _mask_to_intervals(_in_region_mask(region, lat, lon), times)
        logger.info(f"Region {region.name}: {len(region_intervals)} passes")
        intervals.extend((region.name, a, b) for a, b in region_intervals)
    intervals.sort(key=lambda x: x[1])

    events = plan_mpi_events(
        intervals,
        collection=timedelta(seconds=args.mpi_collection_duration_sec),
        break_duration=timedelta(seconds=args.mpi_break_duration_sec),
        min_collection=timedelta(seconds=args.mpi_min_collection_duration_sec),
        filename_format=args.mpi_filename_format,
    )

    schema = {
        "timestamp": pl.Datetime("us", "UTC"),
        "command": pl.String,
        "region": pl.String,
        "latitude_deg": pl.Float64,
        "longitude_deg": pl.Float64,
    }
    if not events:
        logger.warning("Satellite never enters any region in the planning window.")
        return pl.DataFrame(schema=schema)

    event_lat, event_lon = propagate_lat_lon(tle, [e["timestamp"] for e in events])  # type: ignore[misc]
    return (
        pl.DataFrame(events)
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            latitude_deg=pl.Series(event_lat).round(4),
            longitude_deg=pl.Series(event_lon).round(4),
        )
        .select(schema.keys())
        .sort("timestamp")
    )


def main() -> None:
    """Entry point."""
    args = tyro.cli(Args)
    df = run(args)

    with pl.Config(
        tbl_rows=-1,
        tbl_width_chars=250,
        fmt_str_lengths=100,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(df)  # noqa: T201

    logger.info(f"Planned {df.height // 2} MPI recordings ({df.height} telecommands).")

    if args.output_csv is not None:
        df.write_csv(args.output_csv)
        logger.info(f"Wrote plan to {args.output_csv}")


if __name__ == "__main__":
    main()
