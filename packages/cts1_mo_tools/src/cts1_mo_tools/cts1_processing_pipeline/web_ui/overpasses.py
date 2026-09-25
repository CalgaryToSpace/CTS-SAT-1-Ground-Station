"""Overpass prediction for the web UI's Overpasses page: fetches the
satellite's latest TLE from the SatNOGS DB, propagates it with SGP4 (via
`satkit`), and finds every window where it's above a ground station's
horizon.

Pure data layer: no NiceGUI/rendering concerns live here -- see
`overpasses_page` for the page itself.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_NORAD_ID",
    "RAO",
    "GroundStation",
    "Overpass",
    "SatnogsTle",
    "compass_direction",
    "compute_overpasses",
    "fetch_satnogs_tle",
]

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import requests
import satkit as sk

DEFAULT_NORAD_ID: Final = "69015"

SATNOGS_DB_TLE_URL: Final = "https://db.satnogs.org/api/tle/"

# SatNOGS DB only picks up a new TLE a few times a day, so there's no point
# re-fetching it on every page load/refresh.
_TLE_CACHE_TTL_SEC: Final = 30 * 60

# Coarse elevation sampling step: must be well under the shortest pass
# worth finding (a LEO pass that just grazes the horizon still lasts
# a minute or so), after which each horizon crossing is refined by bisection.
_COARSE_STEP_SEC: Final = 10.0
_BISECTION_TOLERANCE_SEC: Final = 0.05
# Sampling step for finding each pass's max elevation.
_FINE_STEP_SEC: Final = 1.0


@dataclass(frozen=True, slots=True)
class GroundStation:
    """A ground station's name, geodetic location, and local time zone."""

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    """Height above the WGS-84 ellipsoid, in meters."""
    timezone: str
    """IANA time zone name, for displaying local times."""


RAO: Final = GroundStation(
    name="RAO",
    latitude_deg=50.8684,
    longitude_deg=-114.2910,
    altitude_m=1269.0,
    timezone="America/Edmonton",
)
"""Rothney Astrophysical Observatory -- the uplink station."""


@dataclass(frozen=True, slots=True)
class SatnogsTle:
    """A TLE as served by the SatNOGS DB API."""

    name: str
    line1: str
    line2: str
    source: str
    updated_at: datetime
    """When SatNOGS DB last updated this TLE (not the TLE's own epoch)."""

    @property
    def epoch(self) -> datetime:
        """The TLE's own epoch, parsed from line 1 (columns 19-32)."""
        field = self.line1[18:32]
        two_digit_year = int(field[:2])
        year = 2000 + two_digit_year if two_digit_year < 57 else 1900 + two_digit_year  # noqa: PLR2004
        day_of_year = float(field[2:])
        return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1)


@dataclass(frozen=True, slots=True)
class Overpass:
    """One pass of the satellite above a ground station's horizon (or above
    whatever minimum elevation `compute_overpasses` was asked for).
    """

    aos: datetime
    """Acquisition of signal: when the satellite rises above the horizon."""
    los: datetime
    """Loss of signal: when the satellite sets below the horizon."""
    max_elevation_at: datetime
    max_elevation_deg: float
    aos_azimuth_deg: float
    los_azimuth_deg: float

    @property
    def duration(self) -> timedelta:
        return self.los - self.aos


_tle_cache: dict[str, tuple[float, SatnogsTle]] = {}
_tle_cache_lock = threading.Lock()


def fetch_satnogs_tle(norad_id: str) -> SatnogsTle:
    """The latest TLE SatNOGS DB has for `norad_id`.

    Cached for `_TLE_CACHE_TTL_SEC` per NORAD ID. Blocking -- call it off
    the event loop.

    Raises:
        requests.RequestException: On a network/HTTP error.
        LookupError: If SatNOGS DB has no TLE for `norad_id`.
    """
    now = time.monotonic()
    with _tle_cache_lock:
        cached = _tle_cache.get(norad_id)
    if cached is not None and now - cached[0] < _TLE_CACHE_TTL_SEC:
        return cached[1]

    r = requests.get(
        SATNOGS_DB_TLE_URL,
        params={"norad_cat_id": norad_id, "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    entries: list[dict[str, Any]] = r.json()
    if not entries:
        msg = f"SatNOGS DB has no TLE for NORAD ID {norad_id}."
        raise LookupError(msg)

    entry = entries[0]
    tle = SatnogsTle(
        name=str(entry["tle0"]).removeprefix("0 ").strip(),
        line1=str(entry["tle1"]),
        line2=str(entry["tle2"]),
        source=str(entry.get("tle_source") or "unknown"),
        updated_at=datetime.fromisoformat(str(entry["updated"])),
    )
    with _tle_cache_lock:
        _tle_cache[norad_id] = (now, tle)
    return tle


def _enu_rotation(station: GroundStation) -> npt.NDArray[np.float64]:
    """Rotation matrix from ITRF (ECEF) to the station's local East-North-Up."""
    lat = np.radians(station.latitude_deg)
    lon = np.radians(station.longitude_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


class _LookAngleCalculator:
    """Azimuth/elevation of a TLE'd satellite from one ground station."""

    def __init__(self, tle: SatnogsTle, station: GroundStation) -> None:
        self._tle: Any = sk.TLE.from_lines([tle.name, tle.line1, tle.line2])
        station_coord = sk.itrfcoord(
            latitude_deg=station.latitude_deg,
            longitude_deg=station.longitude_deg,
            altitude=station.altitude_m,
        )
        self._station_itrf = np.asarray(station_coord.vector, dtype=np.float64)
        self._enu_rotation = _enu_rotation(station)

    def look_angles(
        self, times: list[datetime]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(azimuth_deg, elevation_deg) arrays, one entry per time in `times`."""
        sk_times = [sk.time.from_datetime(t) for t in times]
        pos_teme, _vel = sk.sgp4(self._tle, sk_times)  # pyright: ignore[reportUnknownMemberType]
        pos_teme = np.asarray(pos_teme, dtype=np.float64).reshape(-1, 3)
        teme_to_itrf: Any = sk.frametransform.qteme2itrf(sk_times)
        if isinstance(teme_to_itrf, sk.quaternion):  # A lone time gets a lone one.
            teme_to_itrf = [teme_to_itrf]
        pos_itrf = np.array(
            [q * p for q, p in zip(teme_to_itrf, pos_teme, strict=True)],
            dtype=np.float64,
        )
        enu = (pos_itrf - self._station_itrf) @ self._enu_rotation.T
        east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
        elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
        azimuth = np.degrees(np.arctan2(east, north)) % 360.0
        return azimuth, elevation

    def elevation_at(self, t: datetime) -> float:
        return float(self.look_angles([t])[1][0])

    def azimuth_at(self, t: datetime) -> float:
        return float(self.look_angles([t])[0][0])


def _refine_crossing(
    calc: _LookAngleCalculator,
    below: datetime,
    above: datetime,
    min_elevation_deg: float,
) -> datetime:
    """Bisect for when elevation crosses `min_elevation_deg`, between a time
    it's below and a time it's above (in either order).
    """
    while abs((above - below).total_seconds()) > _BISECTION_TOLERANCE_SEC:
        mid = below + (above - below) / 2
        if calc.elevation_at(mid) >= min_elevation_deg:
            above = mid
        else:
            below = mid
    return above


def compute_overpasses(
    tle: SatnogsTle,
    station: GroundStation,
    start: datetime,
    end: datetime,
    *,
    min_elevation_deg: float = 0.0,
) -> list[Overpass]:
    """Every pass over `station` that's above `min_elevation_deg` at some
    point in [start, end], oldest first.

    A pass already in progress at `start` (or still in progress at `end`)
    is included with its true AOS (or LOS), searched for past the window's
    edge -- so e.g. `start=now` still reports the full current pass.
    """
    calc = _LookAngleCalculator(tle, station)

    # Pad the window by one pass-length each side so a pass straddling
    # either edge still gets both of its crossings found.
    pad = timedelta(minutes=30)
    n_steps = int(((end - start) + 2 * pad).total_seconds() // _COARSE_STEP_SEC) + 1
    times = [
        start - pad + timedelta(seconds=i * _COARSE_STEP_SEC) for i in range(n_steps)
    ]
    _azimuth, elevation = calc.look_angles(times)
    is_up = elevation >= min_elevation_deg

    overpasses: list[Overpass] = []
    aos: datetime | None = None
    for i in range(1, len(times)):
        if is_up[i] and not is_up[i - 1]:
            aos = _refine_crossing(calc, times[i - 1], times[i], min_elevation_deg)
        elif not is_up[i] and is_up[i - 1] and aos is not None:
            los = _refine_crossing(calc, times[i], times[i - 1], min_elevation_deg)
            if los >= start and aos <= end:
                overpasses.append(_build_overpass(calc, aos, los))
            aos = None
    return overpasses


def _build_overpass(
    calc: _LookAngleCalculator, aos: datetime, los: datetime
) -> Overpass:
    n_steps = int((los - aos).total_seconds() // _FINE_STEP_SEC) + 1
    times = [aos + timedelta(seconds=i * _FINE_STEP_SEC) for i in range(n_steps)]
    times.append(los)
    _azimuth, elevation = calc.look_angles(times)
    peak = int(np.argmax(elevation))
    return Overpass(
        aos=aos,
        los=los,
        max_elevation_at=times[peak],
        max_elevation_deg=float(elevation[peak]),
        aos_azimuth_deg=calc.azimuth_at(aos),
        los_azimuth_deg=calc.azimuth_at(los),
    )


_COMPASS_POINTS: Final = (
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()  # noqa: SIM905
)


def compass_direction(azimuth_deg: float) -> str:
    """16-point compass direction for an azimuth, e.g. 100 -> "E"."""
    return _COMPASS_POINTS[round(azimuth_deg / 22.5) % 16]
