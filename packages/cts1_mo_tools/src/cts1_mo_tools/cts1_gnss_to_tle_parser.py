# pyright: standard
"""
Method:
1. Read the GNSS log and keep only the good position and velocity readings.
2. Rotate each reading from the Earth-fixed frame into a non-rotating frame.
3. Let satkit search for the TLE whose predictions best match all the
   readings (a least squares fit).
4. Compare the TLE with the readings again, and stop with an error if it's
   off by more than 1 km on average.

Source: Vallado and Crawford, "SGP4 Orbit Determination", AIAA 2008-6770.
"""

import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import satkit as sk
import tyro
from loguru import logger

# GPS time started on this date and is 18 seconds ahead of UTC
GPS_START = datetime(1980, 1, 6, tzinfo=UTC)
LEAP_SECONDS = 18

MAX_ERROR_KM = 1.0


def line_is_good(line: str) -> bool:
    """
    Check the CRC at the end of a NovAtel line to make sure it wasn't corrupted.
    """
    text, _, crc = line.strip().partition("*")
    text = text[1:]
    try:
        expected = int(crc, 16)
    except ValueError:
        return False
    actual = zlib.crc32(text.encode(), 0xFFFFFFFF) ^ 0xFFFFFFFF
    return actual == expected


def load_gnss_file(
    path: Path,
) -> tuple[list[datetime], list[list[float]], list[list[float]]]:
    """
    Read the good #BESTXYZA lines from a GNSS log. Returns the times (UTC),
    positions (km) and velocities (km/s). Positions and velocities are in the
    Earth-fixed frame (ITRF).
    """
    times = []
    positions = []
    velocities = []

    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("#BESTXYZA,") or not line_is_good(line):
            continue

        header, _, data = line.partition(";")
        header = header.split(",")
        data = data.split(",")

        if (
            header[4] != "FINESTEERING"
            or data[0] != "SOL_COMPUTED"
            or data[8] != "SOL_COMPUTED"
        ):
            continue

        # GPS time is a week number plus seconds into the week
        week = int(header[5])
        seconds = float(header[6])
        times.append(GPS_START + timedelta(weeks=week, seconds=seconds - LEAP_SECONDS))

        # metres to km
        positions.append(
            [float(data[2]) / 1000, float(data[3]) / 1000, float(data[4]) / 1000]
        )
        velocities.append(
            [float(data[10]) / 1000, float(data[11]) / 1000, float(data[12]) / 1000]
        )

    if len(times) == 0:
        msg = f"no good GNSS fixes in {path}"
        raise ValueError(msg)

    return times, positions, velocities


def to_satkit_time(t: datetime) -> sk.time:
    """Turn a Python datetime into a satkit time."""
    return sk.time(
        t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6
    )


def tle_position(tle: sk.TLE, t: datetime) -> np.ndarray:
    """Where the TLE says the satellite is at time t, in the Earth-fixed frame (km)."""
    st = to_satkit_time(t)
    pos, vel = sk.sgp4(tle, st)
    # SGP4 gives TEME, rotate it to the Earth-fixed frame
    pos, vel = sk.frametransform.transform_state(
        from_frame=sk.frame.TEME,
        to_frame=sk.frame.ITRF,
        tm=st,
        pos=np.ravel(pos),
        vel=np.ravel(vel),
    )
    return np.array(pos) / 1000


def fit_tle(
    times: list[datetime], positions: list[list[float]], velocities: list[list[float]]
) -> tuple[sk.TLE, float]:
    """
    Fit a TLE to the GNSS points.
    Returns the TLE and its average (RMS) position error against the GNSS points in km.
    """
    sk_times = []
    states = []
    for i in range(len(times)):
        st = to_satkit_time(times[i])
        # non-rotating frame (GCRF) in metres
        pos, vel = sk.frametransform.transform_state(
            from_frame=sk.frame.ITRF,
            to_frame=sk.frame.GCRF,
            tm=st,
            pos=np.array(positions[i]) * 1000,
            vel=np.array(velocities[i]) * 1000,
        )
        sk_times.append(st)
        states.append(np.concatenate([pos, vel]))

    best_tle = None
    best_rms = float("inf")
    for epoch in [sk_times[0], sk_times[len(sk_times) // 2], sk_times[-1]]:
        tle, _ = sk.TLE.fit_from_states(states, sk_times, epoch)
        rms = rms_error_km(tle, times, positions)
        if rms < best_rms:
            best_tle = tle
            best_rms = rms

    if best_tle is None or best_rms > MAX_ERROR_KM:
        msg = f"bad TLE fit, average error is {best_rms:.3f} km"
        raise RuntimeError(msg)

    return best_tle, best_rms


def rms_error_km(
    tle: sk.TLE, times: list[datetime], positions: list[list[float]]
) -> float:
    """Average (RMS) distance between the TLE and the GNSS points, in km."""
    errors = [
        np.linalg.norm(tle_position(tle, times[i]) - np.array(positions[i]))
        for i in range(len(times))
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def convert_gnss_to_tle(gnss_file: Path) -> None:
    """Read a GNSS log, print the fitted TLE and plot the data."""
    times, positions, velocities = load_gnss_file(gnss_file)
    logger.info(f"loaded {len(times)} GNSS points")

    tle, rms = fit_tle(times, positions, velocities)
    line1, line2 = tle.to_2line()
    logger.info(f"average error: {rms * 1000:.0f} m")
    logger.info(f"fitted TLE:\n{line1}\n{line2}")

    # height above the equator radius
    heights = [np.linalg.norm(p) - 6378.137 for p in positions]
    plt.plot(np.array(times), heights, ".")  # dots, so gaps in the data are visible
    plt.ylabel("altitude (km)")
    plt.xticks(rotation=45)
    plt.show()

    # ground track as latitude vs longitude
    lats = []
    lons = []
    for p in positions:
        coord = sk.itrfcoord(np.array(p) * 1000)
        lats.append(coord.latitude_deg)
        lons.append(coord.longitude_deg)
    plt.scatter(lons, lats, s=2)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("longitude (deg)")
    plt.ylabel("latitude (deg)")
    plt.show()


def main() -> None:
    """Command-line entry point."""
    tyro.cli(convert_gnss_to_tle)


if __name__ == "__main__":
    main()
