from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger
from scipy.optimize import least_squares
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

MU = 398600.8  # Gravity constant
EARTH_SPIN = 7.292115e-5
MIN_HEADER_FIELDS = 7
MIN_MESSAGE_FIELDS = 16
ANTIMERIDIAN_DEG = 180

# Time and coordinate conversion


def julian_date(date: datetime) -> float:
    # converts a normal calendar date/time into a "Julian date"
    year, month, day = date.year, date.month, date.day
    hour = date.hour + date.minute / 60 + date.second / 3600
    return (
        367 * year
        - int(7 * (year + int((month + 9) / 12)) / 4)
        + int(275 * month / 9)
        + day
        + 1721013.5
        + hour / 24
    )


def gmst(julian_date_value: float) -> float:
    # Greenwich Mean Sidereal Time
    centuries = (julian_date_value - 2451545.0) / 36525  # centuries since year 2000
    seconds = (
        67310.54841
        + (876600 * 3600 + 8640184.812866) * centuries
        + 0.093104 * centuries**2
        - 6.2e-6 * centuries**3
    )
    degrees = (seconds / 240) % 360
    return np.radians(degrees)


def ecef_to_teme(
    pos: np.ndarray, vel: np.ndarray, julian_date_value: float
) -> tuple[np.ndarray, np.ndarray]:
    # rotates a position + velocity from ECEF into TEME
    angle = gmst(julian_date_value)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # rotation matrix by "angle" around Earth's axis
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    pos_teme = rot @ pos
    # velocity correction (the ECEF frame itself is spinning)
    vel_teme = rot @ vel + EARTH_SPIN * np.array([-pos_teme[1], pos_teme[0], 0])
    return pos_teme, vel_teme


# Initial orbit guess


def rv2coe(
    pos: np.ndarray, vel: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    # position + velocity -> classical orbital elements
    # just for initial guessing, not for actual orbit propagation
    r_mag, v_mag = np.linalg.norm(pos), np.linalg.norm(vel)

    h_vec = np.cross(pos, vel)  # specific angular momentum vector
    h_mag = np.linalg.norm(h_vec)

    node_vec = np.cross([0, 0, 1], h_vec)  # ascending node direction
    node_mag = np.linalg.norm(node_vec)

    ecc_vec = np.cross(vel, h_vec) / MU - pos / r_mag  # eccentricity vector
    ecc = np.linalg.norm(ecc_vec)

    sma = -MU / (v_mag**2 - 2 * MU / r_mag)  # semi-major axis

    incl = np.arccos(h_vec[2] / h_mag)

    raan = np.arccos(np.clip(node_vec[0] / node_mag, -1, 1))
    if node_vec[1] < 0:
        raan = 2 * np.pi - raan

    argp = np.arccos(np.clip(np.dot(node_vec, ecc_vec) / (node_mag * ecc), -1, 1))
    if ecc_vec[2] < 0:
        argp = 2 * np.pi - argp

    true_anom = np.arccos(np.clip(np.dot(ecc_vec, pos) / (ecc * r_mag), -1, 1))
    if np.dot(pos, vel) < 0:
        true_anom = 2 * np.pi - true_anom

    ecc_anom = 2 * np.arctan2(
        np.sqrt(1 - ecc) * np.sin(true_anom / 2),
        np.sqrt(1 + ecc) * np.cos(true_anom / 2),
    )
    mean_anom = (ecc_anom - ecc * np.sin(ecc_anom)) % (2 * np.pi)

    mean_motion = np.sqrt(MU / sma**3) * 60

    return incl, raan, ecc, argp, mean_anom, mean_motion


# TLE Fit


def fit_tle(
    times: list[datetime],
    positions: list[list[float]],
    velocities: list[list[float]],
    sat_num: int = 99999,
) -> tuple[str, str]:

    jds = [julian_date(t) for t in times]

    r_teme, v_teme = zip(
        *[
            ecef_to_teme(np.array(p), np.array(v), j)
            for j, p, v in zip(jds, positions, velocities, strict=True)
        ],
        strict=True,
    )

    epoch = jds[0] - 2433281.5  # TLE epoch

    guess = rv2coe(r_teme[0], v_teme[0])  # starting guess for the fit

    def residuals(params: np.ndarray) -> list[float]:
        incl, raan, ecc, argp, mean_anom, mean_motion = params
        sat = Satrec()
        sat.sgp4init(
            WGS72,
            "i",
            sat_num,
            epoch,
            0,
            0,
            0,
            ecc,
            argp,
            incl,
            mean_anom,
            mean_motion,
            raan,
        )
        errors = []
        for j, p_true, v_true in zip(jds, r_teme, v_teme, strict=True):
            jd_int = np.floor(j)
            _, p_pred, v_pred = sat.sgp4(jd_int, j - jd_int)
            errors += list(np.array(p_pred) - p_true) + list(np.array(v_pred) - v_true)
        return errors

    fitted = least_squares(residuals, guess, method="lm").x

    incl, raan, ecc, argp, mean_anom, mean_motion = fitted
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        sat_num,
        epoch,
        0,
        0,
        0,
        ecc,
        argp,
        incl,
        mean_anom,
        mean_motion,
        raan,
    )
    sat.classification = "U"  # 'U' = unclassified
    sat.intldesg = "25999A"  # placeholder designator
    sat.elnum = 1  # element set number
    sat.revnum = 1  # orbit number at epoch
    return export_tle(sat)


# Reading GNSS logs

GnssFixes = tuple[list[datetime], list[list[float]], list[list[float]]]


def load_gnss_file(path: Path) -> GnssFixes:
    times: list[datetime] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    with path.open(encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line.startswith("#BESTXYZA,"):
                continue

            header, _, message = line.partition(";")
            header_fields = header.split(",")
            message_fields = message.split(",")
            too_few_header_fields = len(header_fields) < MIN_HEADER_FIELDS
            too_few_message_fields = len(message_fields) < MIN_MESSAGE_FIELDS
            if too_few_header_fields or too_few_message_fields:
                continue

            time_status = header_fields[4]
            pos_status, vel_status = message_fields[0], message_fields[8]
            if (
                time_status != "FINESTEERING"
                or pos_status != "SOL_COMPUTED"
                or vel_status != "SOL_COMPUTED"
            ):
                continue

            gps_week = int(header_fields[5])
            gps_seconds = float(header_fields[6])
            fix_time = datetime(1980, 1, 6, tzinfo=UTC) + timedelta(
                weeks=gps_week, seconds=gps_seconds - 18
            )

            times.append(fix_time)
            positions.append(
                [float(message_fields[i]) / 1000 for i in (2, 3, 4)]
            )  # ECEF X,Y,Z position, m -> km
            velocities.append(
                [float(message_fields[i]) / 1000 for i in (10, 11, 12)]
            )  # ECEF X,Y,Z velocity, m/s -> km/s

    if not times:
        msg = f"no valid GNSS fixes found in {path}"
        raise ValueError(msg)

    return times, positions, velocities


# Entry point


def convert_gnss_to_tle(gnss_file: Path, *, sat_num: int = 99999) -> None:
    times, positions, velocities = load_gnss_file(gnss_file)  # read log file
    line1, line2 = fit_tle(times, positions, velocities, sat_num=sat_num)  # fit TLE
    logger.info(line1)
    logger.info(line2)

    # Plotting
    altitude = [np.linalg.norm(p) - 6378 for p in positions]  # above mean Earth radius
    plt.plot(times, altitude)
    plt.ylabel("altitude (km)")
    plt.xticks(rotation=45)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(
        [p[0] for p in positions], [p[1] for p in positions], [p[2] for p in positions]
    )  # 3D orbit shape
    plt.show()

    # Optional Mapping
    import cartopy.crs as ccrs  # noqa: PLC0415
    import cartopy.feature as cfeature  # noqa: PLC0415
    from pyproj import Transformer  # noqa: PLC0415

    to_latlon = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    x = [p[0] * 1000 for p in positions]
    y = [p[1] * 1000 for p in positions]
    z = [p[2] * 1000 for p in positions]
    lon, lat, _ = to_latlon.transform(x, y, z)
    lon = np.array(lon)
    lat = np.array(lat)

    # line breaks whenever it wraps around the map edge, or jumps a real gap in the data
    lon_wraps = np.abs(np.diff(lon)) > ANTIMERIDIAN_DEG
    seconds_between_fixes = np.array(
        [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    )
    real_gap = seconds_between_fixes > 3 * np.median(seconds_between_fixes)
    break_points = np.where(lon_wraps | real_gap)[0] + 1
    lon = np.insert(lon, break_points, np.nan)
    lat = np.insert(lat, break_points, np.nan)

    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    ax.plot(lon, lat, transform=ccrs.PlateCarree(), color="red", linewidth=2)
    plt.show()


def main() -> None:
    tyro.cli(convert_gnss_to_tle)


if __name__ == "__main__":
    main()
