"""IGRF-14 main field, quasi-dipole (QD) latitude, apex longitude, and MLT.

A vectorized numpy port of `src/orbit/magcoords.c` from simple_sat_ops (by
Johnathan K Burchill, GPL-3.0-or-later), which was validated against apexpy
2.1.1 and IRI-2026's Fortran IGRF.

apexpy itself isn't used because it only ships as a source distribution that
needs a Fortran compiler to install.

Conventions (same as apexpy):
- QD latitude is found by tracing the IGRF field line from the point up to its
  apex: `qdlat = acos(sqrt((RE + h) / (RE + h_apex)))`, signed by hemisphere.
- Apex longitude is the centered-dipole longitude of the apex.
- MLT is `(180 + apex_lon - sun_dipole_lon) / 15` hours, where the Sun's
  longitude is taken in the centered-dipole frame.

All positions are Earth-fixed (ITRF/ECEF) Cartesian, in km.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

IGRF_NMAX = 13
IGRF_A_KM = 6371.2
"""IGRF reference radius."""

QD_RE_KM = 6371.009
"""Mean Earth radius used in apexpy's quasi-dipole latitude formula."""

IGRF_YEAR_FIRST = 2025.0
IGRF_YEAR_LAST = 2030.0
"""IGRF-14's secular variation is defined over this span of decimal years."""

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

_TRACE_STEP_FRAC = 0.04
"""Each RK4 step is this fraction of the current radius."""

_TRACE_RMAX_KM = 100.0 * IGRF_A_KM
"""Beyond this radius the field is a dipole; finish with the dipole formula."""

_TRACE_MAX_STEPS = 20000

# IGRF-14: (g/h, n, m, 2025.0 main field nT, 2025-2030 secular variation nT/yr).
# From https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt
_IGRF14: tuple[tuple[str, int, int, float, float], ...] = (
    ("g", 1, 0, -29350.0, 12.6),
    ("g", 1, 1, -1410.3, 10.0),
    ("h", 1, 1, 4545.5, -21.5),
    ("g", 2, 0, -2556.2, -11.2),
    ("g", 2, 1, 2950.9, -5.3),
    ("h", 2, 1, -3133.6, -27.3),
    ("g", 2, 2, 1648.7, -8.3),
    ("h", 2, 2, -814.2, -11.1),
    ("g", 3, 0, 1360.9, -1.5),
    ("g", 3, 1, -2404.2, -4.4),
    ("h", 3, 1, -56.9, 3.8),
    ("g", 3, 2, 1243.8, 0.4),
    ("h", 3, 2, 237.6, -0.2),
    ("g", 3, 3, 453.4, -15.6),
    ("h", 3, 3, -549.6, -3.9),
    ("g", 4, 0, 894.7, -1.7),
    ("g", 4, 1, 799.6, -2.3),
    ("h", 4, 1, 278.6, -1.3),
    ("g", 4, 2, 55.8, -5.8),
    ("h", 4, 2, -134.0, 4.1),
    ("g", 4, 3, -281.1, 5.4),
    ("h", 4, 3, 212.0, 1.6),
    ("g", 4, 4, 12.0, -6.8),
    ("h", 4, 4, -375.4, -4.1),
    ("g", 5, 0, -232.9, 0.6),
    ("g", 5, 1, 369.0, 1.3),
    ("h", 5, 1, 45.3, -0.5),
    ("g", 5, 2, 187.2, 0.0),
    ("h", 5, 2, 220.0, 2.1),
    ("g", 5, 3, -138.7, 0.7),
    ("h", 5, 3, -122.9, 0.5),
    ("g", 5, 4, -141.9, 2.3),
    ("h", 5, 4, 42.9, 1.7),
    ("g", 5, 5, 20.9, 1.0),
    ("h", 5, 5, 106.2, 1.9),
    ("g", 6, 0, 64.3, -0.2),
    ("g", 6, 1, 63.8, -0.3),
    ("h", 6, 1, -18.4, 0.3),
    ("g", 6, 2, 76.7, 0.8),
    ("h", 6, 2, 16.8, -1.6),
    ("g", 6, 3, -115.7, 1.2),
    ("h", 6, 3, 48.9, -0.4),
    ("g", 6, 4, -40.9, -0.8),
    ("h", 6, 4, -59.8, 0.8),
    ("g", 6, 5, 14.9, 0.4),
    ("h", 6, 5, 10.9, 0.7),
    ("g", 6, 6, -60.8, 0.9),
    ("h", 6, 6, 72.8, 0.9),
    ("g", 7, 0, 79.6, -0.1),
    ("g", 7, 1, -76.9, -0.1),
    ("h", 7, 1, -48.9, 0.6),
    ("g", 7, 2, -8.8, -0.1),
    ("h", 7, 2, -14.4, 0.5),
    ("g", 7, 3, 59.3, 0.5),
    ("h", 7, 3, -1.0, -0.7),
    ("g", 7, 4, 15.8, -0.1),
    ("h", 7, 4, 23.5, 0.0),
    ("g", 7, 5, 2.5, -0.8),
    ("h", 7, 5, -7.4, -0.9),
    ("g", 7, 6, -11.2, -0.8),
    ("h", 7, 6, -25.1, 0.5),
    ("g", 7, 7, 14.3, 0.9),
    ("h", 7, 7, -2.2, -0.3),
    ("g", 8, 0, 23.1, -0.1),
    ("g", 8, 1, 10.9, 0.2),
    ("h", 8, 1, 7.2, -0.3),
    ("g", 8, 2, -17.5, 0.0),
    ("h", 8, 2, -12.6, 0.4),
    ("g", 8, 3, 2.0, 0.4),
    ("h", 8, 3, 11.5, -0.3),
    ("g", 8, 4, -21.8, -0.1),
    ("h", 8, 4, -9.7, 0.4),
    ("g", 8, 5, 16.9, 0.3),
    ("h", 8, 5, 12.7, -0.5),
    ("g", 8, 6, 14.9, 0.1),
    ("h", 8, 6, 0.7, -0.6),
    ("g", 8, 7, -16.8, 0.0),
    ("h", 8, 7, -5.2, 0.3),
    ("g", 8, 8, 1.0, 0.3),
    ("h", 8, 8, 3.9, 0.2),
    ("g", 9, 0, 4.7, 0.0),
    ("g", 9, 1, 8.0, 0.0),
    ("h", 9, 1, -24.8, 0.0),
    ("g", 9, 2, 3.0, 0.0),
    ("h", 9, 2, 12.1, 0.0),
    ("g", 9, 3, -0.2, 0.0),
    ("h", 9, 3, 8.3, 0.0),
    ("g", 9, 4, -2.5, 0.0),
    ("h", 9, 4, -3.4, 0.0),
    ("g", 9, 5, -13.1, 0.0),
    ("h", 9, 5, -5.3, 0.0),
    ("g", 9, 6, 2.4, 0.0),
    ("h", 9, 6, 7.2, 0.0),
    ("g", 9, 7, 8.6, 0.0),
    ("h", 9, 7, -0.6, 0.0),
    ("g", 9, 8, -8.7, 0.0),
    ("h", 9, 8, 0.8, 0.0),
    ("g", 9, 9, -12.8, 0.0),
    ("h", 9, 9, 9.8, 0.0),
    ("g", 10, 0, -1.3, 0.0),
    ("g", 10, 1, -6.4, 0.0),
    ("h", 10, 1, 3.3, 0.0),
    ("g", 10, 2, 0.2, 0.0),
    ("h", 10, 2, 0.1, 0.0),
    ("g", 10, 3, 2.0, 0.0),
    ("h", 10, 3, 2.5, 0.0),
    ("g", 10, 4, -1.0, 0.0),
    ("h", 10, 4, 5.4, 0.0),
    ("g", 10, 5, -0.5, 0.0),
    ("h", 10, 5, -9.0, 0.0),
    ("g", 10, 6, -0.9, 0.0),
    ("h", 10, 6, 0.4, 0.0),
    ("g", 10, 7, 1.5, 0.0),
    ("h", 10, 7, -4.2, 0.0),
    ("g", 10, 8, 0.9, 0.0),
    ("h", 10, 8, -3.8, 0.0),
    ("g", 10, 9, -2.6, 0.0),
    ("h", 10, 9, 0.9, 0.0),
    ("g", 10, 10, -3.9, 0.0),
    ("h", 10, 10, -9.0, 0.0),
    ("g", 11, 0, 3.0, 0.0),
    ("g", 11, 1, -1.4, 0.0),
    ("h", 11, 1, 0.0, 0.0),
    ("g", 11, 2, -2.5, 0.0),
    ("h", 11, 2, 2.8, 0.0),
    ("g", 11, 3, 2.4, 0.0),
    ("h", 11, 3, -0.6, 0.0),
    ("g", 11, 4, -0.6, 0.0),
    ("h", 11, 4, 0.1, 0.0),
    ("g", 11, 5, 0.0, 0.0),
    ("h", 11, 5, 0.5, 0.0),
    ("g", 11, 6, -0.6, 0.0),
    ("h", 11, 6, -0.3, 0.0),
    ("g", 11, 7, -0.1, 0.0),
    ("h", 11, 7, -1.2, 0.0),
    ("g", 11, 8, 1.1, 0.0),
    ("h", 11, 8, -1.7, 0.0),
    ("g", 11, 9, -1.0, 0.0),
    ("h", 11, 9, -2.9, 0.0),
    ("g", 11, 10, -0.1, 0.0),
    ("h", 11, 10, -1.8, 0.0),
    ("g", 11, 11, 2.6, 0.0),
    ("h", 11, 11, -2.3, 0.0),
    ("g", 12, 0, -2.0, 0.0),
    ("g", 12, 1, -0.1, 0.0),
    ("h", 12, 1, -1.2, 0.0),
    ("g", 12, 2, 0.4, 0.0),
    ("h", 12, 2, 0.6, 0.0),
    ("g", 12, 3, 1.2, 0.0),
    ("h", 12, 3, 1.0, 0.0),
    ("g", 12, 4, -1.2, 0.0),
    ("h", 12, 4, -1.5, 0.0),
    ("g", 12, 5, 0.6, 0.0),
    ("h", 12, 5, 0.0, 0.0),
    ("g", 12, 6, 0.5, 0.0),
    ("h", 12, 6, 0.6, 0.0),
    ("g", 12, 7, 0.5, 0.0),
    ("h", 12, 7, -0.2, 0.0),
    ("g", 12, 8, -0.1, 0.0),
    ("h", 12, 8, 0.8, 0.0),
    ("g", 12, 9, -0.5, 0.0),
    ("h", 12, 9, 0.1, 0.0),
    ("g", 12, 10, -0.2, 0.0),
    ("h", 12, 10, -0.9, 0.0),
    ("g", 12, 11, -1.2, 0.0),
    ("h", 12, 11, 0.1, 0.0),
    ("g", 12, 12, -0.7, 0.0),
    ("h", 12, 12, 0.2, 0.0),
    ("g", 13, 0, 0.2, 0.0),
    ("g", 13, 1, -0.9, 0.0),
    ("h", 13, 1, -0.9, 0.0),
    ("g", 13, 2, 0.6, 0.0),
    ("h", 13, 2, 0.7, 0.0),
    ("g", 13, 3, 0.7, 0.0),
    ("h", 13, 3, 1.2, 0.0),
    ("g", 13, 4, -0.2, 0.0),
    ("h", 13, 4, -0.3, 0.0),
    ("g", 13, 5, 0.5, 0.0),
    ("h", 13, 5, -1.3, 0.0),
    ("g", 13, 6, 0.1, 0.0),
    ("h", 13, 6, -0.1, 0.0),
    ("g", 13, 7, 0.7, 0.0),
    ("h", 13, 7, 0.2, 0.0),
    ("g", 13, 8, 0.0, 0.0),
    ("h", 13, 8, -0.2, 0.0),
    ("g", 13, 9, 0.3, 0.0),
    ("h", 13, 9, 0.5, 0.0),
    ("g", 13, 10, 0.2, 0.0),
    ("h", 13, 10, 0.6, 0.0),
    ("g", 13, 11, 0.4, 0.0),
    ("h", 13, 11, -0.6, 0.0),
    ("g", 13, 12, -0.5, 0.0),
    ("h", 13, 12, -0.3, 0.0),
    ("g", 13, 13, -0.4, 0.0),
    ("h", 13, 13, -0.5, 0.0),
)


def decimal_year(t: datetime) -> float:
    """Decimal year of a UTC time, close enough for a field model."""
    j2000 = datetime(2000, 1, 1, tzinfo=UTC)
    return 2000.0 + (t - j2000).total_seconds() / 86400.0 / 365.25


@dataclass(frozen=True, slots=True)
class IgrfModel:
    """IGRF-14 at one epoch, plus its centered-dipole frame."""

    year: float

    g: FloatArray
    h: FloatArray
    """Coefficients at `year` (nT), pre-multiplied by the Schmidt factors so they
    pair with Gauss-normalised associated Legendre functions. Shape (14, 14)."""

    dip_x: FloatArray
    dip_y: FloatArray
    dip_z: FloatArray
    """Centered-dipole frame axes in ECEF: z toward the north geomagnetic pole,
    y = (geographic north) x z, x = y x z."""

    @classmethod
    def at_year(cls, year: float) -> "IgrfModel":
        """Build the model for a decimal year (secular variation extrapolated
        linearly outside 2025-2030)."""
        g = np.zeros((IGRF_NMAX + 1, IGRF_NMAX + 1))
        h = np.zeros((IGRF_NMAX + 1, IGRF_NMAX + 1))
        dt = year - 2025.0
        for gh, n, m, v2025, sv in _IGRF14:
            (g if gh == "g" else h)[n, m] = v2025 + sv * dt

        # The dipole moment points along (g11, h11, g10); the north pole is opposite.
        pole = np.array([-g[1, 1], -h[1, 1], -g[1, 0]])
        dip_z = pole / np.linalg.norm(pole)
        dip_y = np.cross([0.0, 0.0, 1.0], dip_z)
        dip_y /= np.linalg.norm(dip_y)
        dip_x = np.cross(dip_y, dip_z)

        # Fold the Schmidt semi-normalisation into the coefficients.
        s = np.zeros((IGRF_NMAX + 1, IGRF_NMAX + 1))
        s[0, 0] = 1.0
        for n in range(1, IGRF_NMAX + 1):
            s[n, 0] = s[n - 1, 0] * (2.0 * n - 1.0) / n
            for k in range(1, n + 1):
                s[n, k] = s[n, k - 1] * np.sqrt(
                    (n - k + 1.0) * (2.0 if k == 1 else 1.0) / (n + k)
                )
        s[0, 0] = 0.0  # No degree-0 term.

        return cls(year=year, g=g * s, h=h * s, dip_x=dip_x, dip_y=dip_y, dip_z=dip_z)

    def zero_above_degree_1(self) -> "IgrfModel":
        """A dipole-only copy (same frame). For testing against closed forms."""
        g = np.zeros_like(self.g)
        h = np.zeros_like(self.h)
        g[1, :2] = self.g[1, :2]
        h[1, :2] = self.h[1, :2]
        return IgrfModel(self.year, g, h, self.dip_x, self.dip_y, self.dip_z)


def igrf_field_ecef(m: IgrfModel, x_km: FloatArray) -> FloatArray:
    """IGRF field (nT) as ECEF Cartesian components at ECEF positions (N, 3) km."""
    x, y, z = x_km[:, 0], x_km[:, 1], x_km[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    c = np.cos(theta)
    s = np.maximum(np.sin(theta), 1e-12)  # Guard the exact pole.
    n_pts = x.shape[0]

    # Gauss-normalised P(n,m)(cos theta) and dP/dtheta.
    p = np.zeros((IGRF_NMAX + 1, IGRF_NMAX + 1, n_pts))
    dp = np.zeros((IGRF_NMAX + 1, IGRF_NMAX + 1, n_pts))
    p[0, 0] = 1.0
    for n in range(1, IGRF_NMAX + 1):
        for k in range(n + 1):
            if k == n:
                p[n, n] = s * p[n - 1, n - 1]
                dp[n, n] = s * dp[n - 1, n - 1] + c * p[n - 1, n - 1]
            elif n == 1:
                p[1, 0] = c
                dp[1, 0] = -s
            else:
                kk = ((n - 1.0) ** 2 - float(k * k)) / (
                    (2.0 * n - 1.0) * (2.0 * n - 3.0)
                )
                p[n, k] = c * p[n - 1, k] - kk * p[n - 2, k]
                dp[n, k] = c * dp[n - 1, k] - s * p[n - 1, k] - kk * dp[n - 2, k]

    ks = np.arange(IGRF_NMAX + 1)
    cosm = np.cos(np.outer(ks, phi))  # (k, N)
    sinm = np.sin(np.outer(ks, phi))

    ar = IGRF_A_KM / r
    arn = ar * ar
    br = np.zeros(n_pts)
    bt = np.zeros(n_pts)
    bp = np.zeros(n_pts)
    for n in range(1, IGRF_NMAX + 1):
        arn = arn * ar  # (a/r)^(n+2)
        g = m.g[n, : n + 1, None]
        h = m.h[n, : n + 1, None]
        gc = g * cosm[: n + 1] + h * sinm[: n + 1]
        br += (n + 1.0) * arn * (gc * p[n, : n + 1]).sum(axis=0)
        bt -= arn * (gc * dp[n, : n + 1]).sum(axis=0)
        gs = ks[: n + 1, None] * (g * sinm[: n + 1] - h * cosm[: n + 1])
        bp += arn * (gs * p[n, : n + 1]).sum(axis=0)
    bp /= s

    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    horiz = br * st + bt * ct
    return np.stack(
        [horiz * cp - bp * sp, horiz * sp + bp * cp, br * ct - bt * st], axis=1
    )


def geodetic_to_ecef(
    lat_deg: FloatArray, lon_deg: FloatArray, alt_km: FloatArray
) -> FloatArray:
    """WGS84 geodetic to ECEF Cartesian (N, 3) km."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return np.stack(
        [
            (n + alt_km) * cl * np.cos(lon),
            (n + alt_km) * cl * np.sin(lon),
            (n * (1.0 - _WGS84_E2) + alt_km) * sl,
        ],
        axis=1,
    )


def ecef_to_geodetic(x_km: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ECEF Cartesian (N, 3) km to WGS84 (lat_deg, lon_deg, alt_km)."""
    x, y, z = x_km[:, 0], x_km[:, 1], x_km[:, 2]
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        sl = np.sin(lat)
        n = WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * sl * sl)
        lat = np.arctan2(z + _WGS84_E2 * n * sl, p)
    sl = np.sin(lat)
    # Height along the normal; valid at the poles, unlike p/cos(lat) - N.
    alt = p * np.cos(lat) + z * sl - WGS84_A_KM * np.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), alt


def dipole_latlon(m: IgrfModel, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Centered-dipole latitude and longitude (deg) of ECEF directions (N, 3)."""
    r = np.linalg.norm(x, axis=1)
    zc = np.clip(x @ m.dip_z / r, -1.0, 1.0)
    return np.degrees(np.arcsin(zc)), np.degrees(np.arctan2(x @ m.dip_y, x @ m.dip_x))


def _trace_dir(m: IgrfModel, x: FloatArray, sgn: FloatArray) -> FloatArray:
    b = igrf_field_ecef(m, x)
    return sgn[:, None] * b / np.linalg.norm(b, axis=1, keepdims=True)


def _rk4_step(
    m: IgrfModel, x: FloatArray, h: FloatArray, sgn: FloatArray
) -> FloatArray:
    hc = h[:, None]
    k1 = _trace_dir(m, x, sgn)
    k2 = _trace_dir(m, x + 0.5 * hc * k1, sgn)
    k3 = _trace_dir(m, x + 0.5 * hc * k2, sgn)
    k4 = _trace_dir(m, x + hc * k3, sgn)
    return x + hc / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass(frozen=True, slots=True)
class QdResult:
    qdlat_deg: FloatArray
    apex_lon_deg: FloatArray
    apex_height_km: FloatArray


def quasi_dipole(  # noqa: PLR0915
    m: IgrfModel, x_km: FloatArray, alt_km: FloatArray
) -> QdResult:
    """QD latitude and apex longitude of ECEF points (N, 3) km at geodetic
    heights `alt_km`, by tracing each IGRF field line up to its apex.

    Field lines that leave 100 Earth radii (QD latitude above about 84 deg) are
    finished with the dipole formula from there.
    """
    x_km = np.atleast_2d(np.asarray(x_km, dtype=np.float64))
    alt_km = np.atleast_1d(np.asarray(alt_km, dtype=np.float64))
    n_pts = int(x_km.shape[0])

    r0 = np.linalg.norm(x_km, axis=1)
    b0 = igrf_field_ecef(m, x_km)
    br = np.einsum("ij,ij->i", b0, x_km) / r0
    # The field points down in the northern magnetic hemisphere. Trace along
    # sgn * B, the direction that climbs away from the Earth.
    north = br < 0.0
    sgn = np.where(br > 0.0, 1.0, -1.0)

    prev = x_km.copy()
    cur = x_km.copy()
    rc = r0.copy()
    rprev = r0.copy()
    have_prev = np.zeros(n_pts, dtype=bool)
    apex = x_km.copy()
    frac = np.full(n_pts, _TRACE_STEP_FRAC)

    active: npt.NDArray[np.intp] = np.arange(n_pts)
    for _ in range(_TRACE_MAX_STEPS):
        if active.size == 0:
            break
        a = active
        nxt = _rk4_step(m, cur[a], frac[a] * rc[a], sgn[a])
        rn = np.linalg.norm(nxt, axis=1)

        # Far out, the line is a dipole line r = L cos^2(lat), whose apex is in
        # the dipole equator at the same dipole longitude.
        far = rn > _TRACE_RMAX_KM
        if far.any():
            dlat, dlon = dipole_latlon(m, nxt[far])
            ell = rn[far] / np.cos(np.radians(dlat)) ** 2
            dl = np.radians(dlon)
            apex[a[far]] = ell[:, None] * (
                np.cos(dl)[:, None] * m.dip_x + np.sin(dl)[:, None] * m.dip_y
            )

        descending = (rn <= rc[a]) & ~far
        # The first step overshot the apex (a short line near the magnetic
        # equator): halve the step and retry.
        shrink = descending & ~have_prev[a] & (frac[a] * rc[a] > 0.01)  # noqa: PLR2004
        frac[a[shrink]] *= 0.5

        # Passed the top: fit a parabola through the last three points and take
        # the position at its peak radius. A start within 10 m of the apex
        # stands in for it (apex already equals the start point).
        top = descending & ~shrink
        fit = top & have_prev[a]
        if fit.any():
            i = a[fit]
            curv = 2.0 * rc[i] - rprev[i] - rn[fit]
            u = np.where(
                curv > 0.0,
                (rn[fit] - rprev[i]) / (2.0 * np.where(curv > 0.0, curv, 1.0)),
                0.0,
            )
            u = u[:, None]
            nx = nxt[fit]
            apex[i] = (
                cur[i]
                + 0.5 * u * (nx - prev[i])
                + 0.5 * u * u * (nx - 2.0 * cur[i] + prev[i])
            )

        advance = ~far & ~descending
        i = a[advance]
        prev[i] = cur[i]
        cur[i] = nxt[advance]
        rprev[i] = rc[i]
        rc[i] = rn[advance]
        have_prev[i] = True

        active = a[advance | shrink]

    _, alon = dipole_latlon(m, apex)
    _, _, h_apex = ecef_to_geodetic(apex)
    h_apex = np.maximum(h_apex, alt_km)
    lat = np.degrees(np.arccos(np.sqrt((QD_RE_KM + alt_km) / (QD_RE_KM + h_apex))))
    return QdResult(
        qdlat_deg=np.where(north, lat, -lat),
        apex_lon_deg=alon,
        apex_height_km=h_apex,
    )


def magnetic_local_time(
    m: IgrfModel, apex_lon_deg: FloatArray, sun_ecef: FloatArray
) -> FloatArray:
    """MLT (hours, [0, 24)) of apex longitudes, given ECEF Sun directions (N, 3)."""
    _, slon = dipole_latlon(m, np.atleast_2d(sun_ecef))
    return np.mod((180.0 + apex_lon_deg - slon) / 15.0, 24.0)
