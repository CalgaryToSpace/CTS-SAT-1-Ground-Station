"""
Plan MPI data-collection and camera telecommands over space-physics regions.

Fetches the latest TLE for a satellite from CelesTrak, propagates the orbit with
satkit (SGP4), and finds when the satellite crosses each configured region. Like
simple_sat_ops' `region_crossings` tool, each region is a band of magnetic
latitude (either hemisphere) within a window of magnetic local time (MLT):

    cusp     70-80 deg, 09-15 MLT   the dayside cusp and cleft
    aurora   62-72 deg, 18-06 MLT   the nightside auroral oval
    steve    55-62 deg, 18-24 MLT   the pre-midnight subauroral (STEVE) zone

Magnetic latitude is quasi-dipole (QD) latitude at the satellite's altitude, and
MLT follows apexpy's convention (see `magcoords.py`). Crossings are also filtered
by lighting (umbra-only by default).

Within each crossing, it schedules camera photos (`camera_setup`, then
`camera_capture` a fixed lead time later), and fills the rest of the crossing with
repeated MPI start/stop telecommands.

Usage (uv):
    uv run cts1_plan_mpi_ops --duration-hours 24
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import polars as pl
import requests
import satkit as sk
import tyro
from loguru import logger

from cts1_mo_tools import magcoords as mc

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

TIME_STEP_SEC = 1.0
"""Propagation step used to detect region entry/exit."""

MAG_STEP_SAMPLES = 5
"""Magnetic coordinates (field-line traces) are computed every this many
propagation steps, and linearly interpolated in between."""

TLE_MAX_AGE_DAYS = 7.0
"""Warn when any part of the window is further than this from the TLE epoch."""

CAMERA_FILENAME_MAX_CHARS = 31
"""`camera_capture` copies the filename into a 32-byte buffer (incl. the null)
in the OBC firmware, and rejects anything longer."""

FS_MKDIR_LEAD = timedelta(seconds=10)
"""How long before the first planned command to create the output directories."""

Lighting = Literal["umbra", "penumbra", "sunlit"]
RegionName = Literal["cusp", "aurora", "steve"]
Hemisphere = Literal["north", "south"]
CameraMode = Literal["auto", "d", "m", "n", "s"]

CAMERA_AUTO_MODE: dict[Lighting, str] = {"umbra": "n", "penumbra": "m", "sunlit": "d"}
"""Camera lighting mode picked by `camera_lighting_mode="auto"`."""


@dataclass(frozen=True, slots=True)
class Args:
    """Plan MPI and camera telecommands for when the satellite crosses a region."""

    norad_id: int = 69015
    """NORAD catalog ID of the satellite (69015 is CTS-SAT-1/FrontierSat)."""

    start_time: str | None = None
    """ISO-8601 start of the planning window (with timezone). Default: now."""

    duration_hours: float = 24.0
    """Length of the planning window."""

    tle_file: Path | None = None
    """Optional local TLE file to use instead of fetching from CelesTrak."""

    # ---- Regions ----

    regions: tuple[RegionName, ...] = ("cusp", "aurora", "steve")
    """Which regions to plan operations in."""

    hemispheres: tuple[Hemisphere, ...] = ("north", "south")
    """Which magnetic hemispheres to plan operations in."""

    cusp_mlat_range: str = "70 to 80"
    """Cusp band of |QD magnetic latitude| (degrees)."""

    cusp_mlt_range: str = "9 to 15"
    """Cusp MLT window (hours). Wraps through midnight when start > end."""

    aurora_mlat_range: str = "62 to 72"
    """Nightside auroral band of |QD magnetic latitude| (degrees)."""

    aurora_mlt_range: str = "18 to 6"
    """Auroral MLT window (hours). Wraps through midnight when start > end."""

    steve_mlat_range: str = "55 to 62"
    """Subauroral (STEVE) band of |QD magnetic latitude| (degrees)."""

    steve_mlt_range: str = "18 to 24"
    """Subauroral MLT window (hours). Wraps through midnight when start > end."""

    min_crossing_duration_sec: int | float = 10
    """Ignore region crossings shorter than this (e.g., grazing a region edge)."""

    lighting: tuple[Lighting, ...] = ("umbra",)
    """Which lighting conditions to plan operations in: "umbra" (Sun fully
    hidden), "penumbra" (Sun partly hidden), and/or "sunlit" (daytime)."""

    # ---- MPI ----

    plan_mpi: bool = True
    """Plan MPI recordings."""

    mpi_collection_duration_sec: int | float = 60
    """Duration of each MPI recording."""

    mpi_break_duration_sec: int | float = 5
    """Gap between stopping one MPI recording and starting the next."""

    mpi_min_collection_duration_sec: int | float = 10
    """Skip recordings shorter than this (when truncated by leaving the region)."""

    mpi_filename_format: str = "mpi_data/%Y-%m/%Y-%m-%d_%H%M%SZ.mpi"
    """strftime format of the MPI filename passed to mpi_enable_active_mode."""

    # ---- Camera ----

    plan_camera: bool = True
    """Plan camera photos."""

    camera_photos_per_crossing: int = 1
    """Number of photos per region crossing, spread evenly through it."""

    camera_setup_lead_sec: int | float = 30
    """Time between camera_setup and camera_capture."""

    camera_capture_duration_sec: int | float = 25
    """Time reserved for camera_capture to finish (the firmware's receive timeout
    is 20 s). No MPI recording is planned from camera_setup until this long after
    camera_capture."""

    camera_lighting_mode: CameraMode = "auto"
    """camera_capture lighting mode: d (daylight), m (medium), n (night), or s
    (solar sail). "auto" picks n in umbra, m in penumbra, and d when sunlit."""

    camera_filename_format: str = "camera/%Y-%m-%d_%H%M%SZ.img"
    """strftime format of the image filename passed to camera_capture."""

    # ---- Output ----

    make_directories: bool = True
    """Add fs_make_directory telecommands for the output folders before the first
    planned command (LittleFS doesn't create parent folders on its own)."""

    output_csv: Path | None = None
    """Optional path to write the resulting plan as a CSV."""

    output_agenda: Path | None = None
    """Optional path to write the plan as a telecommand agenda text file."""


@dataclass(frozen=True, slots=True)
class MagRegion:
    name: RegionName
    mlat_range: tuple[float, float]
    """Band of |QD latitude|: lo <= |lat| < hi, degrees."""

    mlt_range: tuple[float, float]
    """MLT window (hours): lo <= MLT < hi, wrapping through midnight if lo > hi."""


@dataclass(frozen=True, slots=True)
class Crossing:
    label: str
    """Like "aurora-N"."""

    enter: datetime
    exit: datetime


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: datetime
    command: str
    region: str


def _split_range(range_str: str) -> tuple[float, float]:
    """Parse a string like "18 to 6" into (start, end), keeping the order."""
    parts = range_str.lower().split("to")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f'Range must look like "<a> to <b>", got: {range_str!r}'
        raise ValueError(msg)
    a, b = (float(p.strip()) for p in parts)
    return (a, b)


def _parse_mlat_range(range_str: str) -> tuple[float, float]:
    lo, hi = sorted(_split_range(range_str))
    if lo < 0 or hi > 90:  # noqa: PLR2004
        msg = f"Magnetic latitude range must be within 0 to 90 deg, got: {range_str!r}"
        raise ValueError(msg)
    return (lo, hi)


def _parse_mlt_range(range_str: str) -> tuple[float, float]:
    lo, hi = _split_range(range_str)
    if not (0 <= lo <= 24 and 0 <= hi <= 24):  # noqa: PLR2004
        msg = f"MLT range must be within 0 to 24 h, got: {range_str!r}"
        raise ValueError(msg)
    return (lo, hi)


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


@dataclass(frozen=True, slots=True)
class Ephemeris:
    """Satellite state sampled at each propagation time."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    altitude_km: np.ndarray

    qd_latitude_deg: np.ndarray
    """Quasi-dipole magnetic latitude."""

    mlt_hours: np.ndarray
    """Magnetic local time."""

    sun_fraction: np.ndarray
    """Fraction of the Sun visible from the satellite: 0 in umbra, 1 in full sun."""


def propagate(
    tle: sk.TLE,
    times: Sequence[datetime],
    igrf: mc.IgrfModel,
    *,
    mag_step_samples: int = 1,
) -> Ephemeris:
    """Propagate the TLE to each time; return its position, magnetic coordinates,
    and lighting.

    Magnetic coordinates are traced every `mag_step_samples` samples (and at the
    last one), and linearly interpolated in between.
    """
    sk_times = [sk.time.from_datetime(t) for t in times]
    pos_teme, _ = sk.sgp4(tle, sk_times)  # pyright: ignore[reportUnknownMemberType]
    pos_teme = np.atleast_2d(pos_teme)

    q_itrf = sk.frametransform.qteme2itrf(sk_times)
    q_gcrf = sk.frametransform.qteme2gcrf(sk_times)
    q_gcrf2itrf = sk.frametransform.qgcrf2itrf(sk_times)
    assert isinstance(q_itrf, list)
    assert isinstance(q_gcrf, list)
    assert isinstance(q_gcrf2itrf, list)

    pos_itrf_km = (
        np.array([q * p for q, p in zip(q_itrf, pos_teme, strict=True)]) / 1000.0
    )
    lat, lon, alt = mc.ecef_to_geodetic(pos_itrf_km)

    # Shadow check is done in GCRF, the frame satkit gives the Sun position in.
    sun_gcrf = np.atleast_2d(sk.sun.pos_gcrf(sk_times))
    sun_fraction = np.array(
        [
            sk.sun.shadowfunc(sun, q * p)
            for q, p, sun in zip(q_gcrf, pos_teme, sun_gcrf, strict=True)
        ]
    )
    sun_itrf = np.array([q * s for q, s in zip(q_gcrf2itrf, sun_gcrf, strict=True)])

    # Field-line tracing is the slow part, so do it on a coarser grid.
    n = len(times)
    idx = np.unique(np.append(np.arange(0, n, max(mag_step_samples, 1)), n - 1))
    qd = mc.quasi_dipole(igrf, pos_itrf_km[idx], alt[idx])
    if idx.size == n:
        qdlat, apex_lon = qd.qdlat_deg, qd.apex_lon_deg
    else:
        t_sec = np.array([(t - times[0]).total_seconds() for t in times])
        qdlat = np.interp(t_sec, t_sec[idx], qd.qdlat_deg)
        unwrapped = np.degrees(np.unwrap(np.radians(qd.apex_lon_deg)))
        apex_lon = np.interp(t_sec, t_sec[idx], unwrapped)

    return Ephemeris(
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_km=alt,
        qd_latitude_deg=qdlat,
        mlt_hours=mc.magnetic_local_time(igrf, apex_lon, sun_itrf),
        sun_fraction=sun_fraction,
    )


def in_region_mask(
    region: MagRegion, hemisphere: Hemisphere, eph: Ephemeris
) -> np.ndarray:
    qd = eph.qd_latitude_deg
    abs_qd = np.abs(qd)
    mask = (abs_qd >= region.mlat_range[0]) & (abs_qd < region.mlat_range[1])
    mask &= (qd >= 0) if hemisphere == "north" else (qd < 0)

    mlt = eph.mlt_hours
    lo, hi = region.mlt_range
    if lo <= hi:
        mask &= (mlt >= lo) & (mlt < hi)
    else:
        mask &= (mlt >= lo) | (mlt < hi)
    return mask


def classify_lighting(sun_fraction: np.ndarray) -> np.ndarray:
    """Lighting class name of each sun fraction."""
    return np.where(
        sun_fraction <= 0.0,
        "umbra",
        np.where(sun_fraction < 1.0, "penumbra", "sunlit"),
    )


def _lighting_mask(lighting: Sequence[Lighting], eph: Ephemeris) -> np.ndarray:
    return np.isin(classify_lighting(eph.sun_fraction), list(lighting))


def _mask_to_intervals(
    mask: np.ndarray, times: Sequence[datetime]
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


def _subtract_intervals(
    start: datetime,
    end: datetime,
    blocked: Sequence[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """The parts of [start, end] not covered by any of the (sorted) blocked
    intervals."""
    free: list[tuple[datetime, datetime]] = []
    cursor = start
    for b_start, b_end in blocked:
        if b_end <= cursor or b_start >= end:
            continue
        if b_start > cursor:
            free.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < end:
        free.append((cursor, end))
    return free


@dataclass(frozen=True, slots=True)
class CameraPlan:
    events: list[Event]
    blackouts: list[tuple[datetime, datetime]]
    """Times when the camera is in use (from camera_setup to the end of capture)."""


def plan_camera_events(  # noqa: PLR0913
    crossings: Sequence[Crossing],
    *,
    photos_per_crossing: int,
    setup_lead: timedelta,
    capture_duration: timedelta,
    lighting_mode_at: Callable[[datetime], str],
    filename_format: str,
) -> CameraPlan:
    """Schedule `photos_per_crossing` photos spread evenly through each crossing.

    Each photo is a camera_setup at `capture - setup_lead`, then the
    camera_capture. Photos whose camera use would overlap an earlier one's are
    skipped.
    """
    events: list[Event] = []
    blackouts: list[tuple[datetime, datetime]] = []
    busy_until: datetime | None = None
    for crossing in sorted(crossings, key=lambda c: c.enter):
        span = crossing.exit - crossing.enter
        for i in range(photos_per_crossing):
            capture = crossing.enter + span * ((i + 0.5) / photos_per_crossing)
            capture = capture.replace(microsecond=0)
            setup = capture - setup_lead
            if busy_until is not None and setup < busy_until:
                logger.debug(f"Skipping photo at {capture}: camera still busy.")
                continue
            filename = capture.strftime(filename_format)
            mode = lighting_mode_at(capture)
            events.append(Event(setup, "CTS1+camera_setup()", crossing.label))
            events.append(
                Event(
                    capture,
                    f"CTS1+camera_capture({filename},{mode})",
                    crossing.label,
                )
            )
            busy_until = capture + capture_duration
            blackouts.append((setup, busy_until))
    return CameraPlan(events=events, blackouts=blackouts)


def plan_mpi_events(  # noqa: PLR0913
    crossings: Sequence[Crossing],
    *,
    collection: timedelta,
    break_duration: timedelta,
    min_collection: timedelta,
    filename_format: str,
    blackouts: Sequence[tuple[datetime, datetime]] = (),
) -> list[Event]:
    """Schedule repeated start/stop pairs within each crossing.

    Recordings stay `break_duration` clear of each blackout (camera use) and of
    each other, including across overlapping crossings. The last recording in a
    free stretch is cut short at its end, and skipped entirely if that leaves it
    shorter than `min_collection`.
    """
    padded = sorted((a - break_duration, b + break_duration) for a, b in blackouts)
    events: list[Event] = []
    next_allowed: datetime | None = None
    for crossing in sorted(crossings, key=lambda c: c.enter):
        for free_start, free_end in _subtract_intervals(
            crossing.enter, crossing.exit, padded
        ):
            start = free_start
            if next_allowed is not None:
                start = max(start, next_allowed)
            while start < free_end:
                stop = min(start + collection, free_end)
                if stop - start < min_collection:
                    break
                filename = start.strftime(filename_format)
                events.append(
                    Event(
                        start,
                        f"CTS1+mpi_enable_active_mode({filename})",
                        crossing.label,
                    )
                )
                events.append(
                    Event(stop, "CTS1+mpi_disable_active_mode()", crossing.label)
                )
                start = stop + break_duration
                next_allowed = start
    return events


def _filename_args(events: Sequence[Event]) -> list[str]:
    """Filenames passed to the file-creating telecommands in `events`."""
    prefixes = ("CTS1+mpi_enable_active_mode(", "CTS1+camera_capture(")
    return [
        e.command.removeprefix(prefix).rstrip(")").split(",")[0]
        for e in events
        for prefix in prefixes
        if e.command.startswith(prefix)
    ]


def plan_make_directory_events(events: Sequence[Event]) -> list[Event]:
    """fs_make_directory for every folder (and parent folder) the planned files
    go into, 1 s apart, ending `FS_MKDIR_LEAD` before the first event."""
    dirs: list[str] = []
    for name in _filename_args(events):
        parent = PurePosixPath(name).parent
        for d in reversed([parent, *parent.parents]):
            if str(d) not in {".", "/"} and str(d) not in dirs:
                dirs.append(str(d))
    if not dirs:
        return []

    first = min(e.timestamp for e in events)
    start = first - FS_MKDIR_LEAD - timedelta(seconds=len(dirs) - 1)
    return [
        Event(start + timedelta(seconds=i), f"CTS1+fs_make_directory({d})", "-")
        for i, d in enumerate(dirs)
    ]


def _format_settings(args: Args) -> list[str]:
    return [f"  {k} = {getattr(args, k)!r}" for k in args.__dataclass_fields__]


def write_agenda(df: pl.DataFrame, args: Args, output_file: Path) -> None:
    """Write the plan as an agenda file, with tssent equal to tsexec."""
    header = [
        "# MPI/camera operations plan generated by cts1_plan_mpi_ops.",
        f"# Generated at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "# Input settings:",
        *(f"#{line}" for line in _format_settings(args)),
        "",
    ]

    lines: list[str] = []
    for row in df.iter_rows(named=True):
        ts: datetime = row["timestamp"]
        ts_ms = int(ts.timestamp() * 1000)
        command = f"{row['command']}@tssent={ts_ms}@tsexec={ts_ms}!"
        comment = (
            f"# {ts.strftime('%Y-%m-%dT%H:%M:%SZ')} region={row['region']} "
            f"lat={row['latitude_deg']:.4f} lon={row['longitude_deg']:.4f} "
            f"mlat={row['qd_latitude_deg']:.2f} mlt={row['mlt_hours']:.2f} "
            f"lighting={row['lighting']} sun_fraction={row['sun_fraction']:.2f}"
        )
        lines.append(f"{command}  {comment}")

    output_file.write_text("\n".join(header + lines) + "\n")


def _build_regions(args: Args) -> list[MagRegion]:
    all_regions = {
        "cusp": MagRegion(
            "cusp",
            _parse_mlat_range(args.cusp_mlat_range),
            _parse_mlt_range(args.cusp_mlt_range),
        ),
        "aurora": MagRegion(
            "aurora",
            _parse_mlat_range(args.aurora_mlat_range),
            _parse_mlt_range(args.aurora_mlt_range),
        ),
        "steve": MagRegion(
            "steve",
            _parse_mlat_range(args.steve_mlat_range),
            _parse_mlt_range(args.steve_mlt_range),
        ),
    }
    return [all_regions[name] for name in dict.fromkeys(args.regions)]


def _validate(args: Args) -> None:
    checks = [
        (len(args.regions) > 0, "At least one region must be selected."),
        (len(args.hemispheres) > 0, "At least one hemisphere must be selected."),
        (len(args.lighting) > 0, "At least one lighting condition must be selected."),
        (args.plan_mpi or args.plan_camera, "Enable plan_mpi and/or plan_camera."),
        (
            args.mpi_collection_duration_sec > 0,
            "mpi_collection_duration_sec must be positive.",
        ),
        (
            args.mpi_break_duration_sec >= 0,
            "mpi_break_duration_sec must not be negative.",
        ),
        (
            args.camera_photos_per_crossing >= 0,
            "camera_photos_per_crossing must not be negative.",
        ),
        (
            args.camera_setup_lead_sec >= 0,
            "camera_setup_lead_sec must not be negative.",
        ),
        (
            args.camera_capture_duration_sec >= 0,
            "camera_capture_duration_sec must not be negative.",
        ),
        (args.duration_hours > 0, "duration_hours must be positive."),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def _load_tle(args: Args) -> sk.TLE:
    if args.tle_file is not None:
        tle_lines = args.tle_file.read_text().strip().splitlines()
    else:
        tle_lines = fetch_tle_lines(args.norad_id)
    tle = sk.TLE.from_lines(tle_lines)
    if isinstance(tle, list):
        tle = tle[0]
    logger.info(f"Using TLE (epoch {tle.epoch}):\n" + "\n".join(tle_lines))
    return tle


def _warn_about_epochs(
    tle: sk.TLE, start: datetime, end: datetime, year: float
) -> None:
    epoch: datetime = tle.epoch.datetime()
    reach_days = max(abs(end - epoch), abs(start - epoch)) / timedelta(days=1)
    if reach_days > TLE_MAX_AGE_DAYS:
        logger.warning(
            f"The window reaches {reach_days:.1f} days from the TLE epoch (more than "
            f"{TLE_MAX_AGE_DAYS:.0f}); crossing times lose accuracy as a TLE ages."
        )
    if not (mc.IGRF_YEAR_FIRST <= year <= mc.IGRF_YEAR_LAST):
        logger.warning(
            f"{year:.1f} is outside IGRF-14's {mc.IGRF_YEAR_FIRST:.0f}-"
            f"{mc.IGRF_YEAR_LAST:.0f} span; the field is extrapolated."
        )


OUTPUT_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "command": pl.String,
    "region": pl.String,
    "latitude_deg": pl.Float64,
    "longitude_deg": pl.Float64,
    "altitude_km": pl.Float64,
    "qd_latitude_deg": pl.Float64,
    "mlt_hours": pl.Float64,
    "lighting": pl.String,
    "sun_fraction": pl.Float64,
}


def run(args: Args) -> pl.DataFrame:
    logger.info("Input settings:\n" + "\n".join(_format_settings(args)))
    _validate(args)
    regions = _build_regions(args)

    tle = _load_tle(args)

    start = _parse_start_time(args.start_time)
    end = start + timedelta(hours=args.duration_hours)
    year = mc.decimal_year(start + (end - start) / 2)
    igrf = mc.IgrfModel.at_year(year)
    _warn_about_epochs(tle, start, end, year)

    num_steps = int(args.duration_hours * 3600 / TIME_STEP_SEC) + 1
    times = [start + timedelta(seconds=i * TIME_STEP_SEC) for i in range(num_steps)]
    logger.info(f"Propagating {num_steps} steps from {start} ({args.duration_hours} h)")

    eph = propagate(tle, times, igrf, mag_step_samples=MAG_STEP_SAMPLES)
    lit = _lighting_mask(args.lighting, eph)
    logger.info(
        f"Satellite is in {'/'.join(args.lighting)} for {lit.mean():.1%} of the "
        "planning window."
    )

    crossings: list[Crossing] = []
    for region in regions:
        for hemi in dict.fromkeys(args.hemispheres):
            label = f"{region.name}-{hemi[0].upper()}"
            region_intervals = [
                (a, b)
                for a, b in _mask_to_intervals(
                    in_region_mask(region, hemi, eph) & lit, times
                )
                if (b - a).total_seconds() >= args.min_crossing_duration_sec
            ]
            logger.info(
                f"Region {label} (|mlat| {region.mlat_range[0]}-{region.mlat_range[1]}"
                f" deg, MLT {region.mlt_range[0]}-{region.mlt_range[1]} h): "
                f"{len(region_intervals)} crossings"
            )
            crossings.extend(Crossing(label, a, b) for a, b in region_intervals)
    crossings.sort(key=lambda c: c.enter)

    lighting_class = classify_lighting(eph.sun_fraction)

    def camera_mode_at(t: datetime) -> str:
        if args.camera_lighting_mode != "auto":
            return args.camera_lighting_mode
        i = min(round((t - start).total_seconds() / TIME_STEP_SEC), num_steps - 1)
        return CAMERA_AUTO_MODE[lighting_class[i]]

    camera = CameraPlan(events=[], blackouts=[])
    if args.plan_camera and args.camera_photos_per_crossing > 0:
        camera = plan_camera_events(
            crossings,
            photos_per_crossing=args.camera_photos_per_crossing,
            setup_lead=timedelta(seconds=args.camera_setup_lead_sec),
            capture_duration=timedelta(seconds=args.camera_capture_duration_sec),
            lighting_mode_at=camera_mode_at,
            filename_format=args.camera_filename_format,
        )

    mpi_events: list[Event] = []
    if args.plan_mpi:
        mpi_events = plan_mpi_events(
            crossings,
            collection=timedelta(seconds=args.mpi_collection_duration_sec),
            break_duration=timedelta(seconds=args.mpi_break_duration_sec),
            min_collection=timedelta(seconds=args.mpi_min_collection_duration_sec),
            filename_format=args.mpi_filename_format,
            blackouts=camera.blackouts,
        )

    events = camera.events + mpi_events
    if not events:
        logger.warning("Satellite never crosses any selected region in the window.")
        return pl.DataFrame(schema=OUTPUT_SCHEMA)

    too_long = [
        n for n in _filename_args(camera.events) if len(n) > CAMERA_FILENAME_MAX_CHARS
    ]
    if too_long:
        logger.warning(
            f"{len(too_long)} camera filenames (like {too_long[0]!r}, "
            f"{len(too_long[0])} chars) are longer than the "
            f"{CAMERA_FILENAME_MAX_CHARS} chars that the OBC firmware's camera_capture "
            "accepts (32-byte buffer). Those commands will fail unless the firmware "
            "buffer is enlarged, or a shorter --camera-filename-format is used."
        )

    if args.make_directories:
        events = plan_make_directory_events(events) + events

    events.sort(key=lambda e: e.timestamp)
    event_eph = propagate(tle, [e.timestamp for e in events], igrf)
    return pl.DataFrame(
        {
            "timestamp": [e.timestamp for e in events],
            "command": [e.command for e in events],
            "region": [e.region for e in events],
            "latitude_deg": event_eph.latitude_deg.round(4),
            "longitude_deg": event_eph.longitude_deg.round(4),
            "altitude_km": event_eph.altitude_km.round(1),
            "qd_latitude_deg": event_eph.qd_latitude_deg.round(2),
            "mlt_hours": event_eph.mlt_hours.round(2),
            "lighting": classify_lighting(event_eph.sun_fraction).tolist(),
            "sun_fraction": event_eph.sun_fraction.round(4),
        },
        schema=OUTPUT_SCHEMA,
    )


def main() -> None:
    """Entry point."""
    args = tyro.cli(Args)
    df = run(args)

    with pl.Config(
        tbl_rows=-1,
        tbl_width_chars=300,
        fmt_str_lengths=100,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(df)  # noqa: T201

    cmd = pl.col("command")
    n_mpi = df.filter(cmd.str.starts_with("CTS1+mpi_enable_active_mode(")).height
    n_cam = df.filter(cmd.str.starts_with("CTS1+camera_capture(")).height
    logger.info(
        f"Planned {n_mpi} MPI recordings and {n_cam} camera photos "
        f"({df.height} telecommands)."
    )

    if args.output_csv is not None:
        df.write_csv(args.output_csv)
        logger.info(f"Wrote plan to {args.output_csv}")

    if args.output_agenda is not None:
        write_agenda(df, args, args.output_agenda)
        logger.info(f"Wrote agenda to {args.output_agenda}")


if __name__ == "__main__":
    main()
