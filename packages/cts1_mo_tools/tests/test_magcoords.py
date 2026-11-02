"""Tests for `magcoords`, ported from simple_sat_ops' `unit_tests/magcoords_selftest.c`.

Field oracle: IRI-2026's Fortran IGRF at decimal year 2026.75 (geodetic N/E/D, nT).
Coordinate oracle: apexpy 2.1.1 (IGRF-14) at decimal year 2026.75. See the C test
for a discussion of apexpy's own error, which sets the tolerances here.
"""

import numpy as np
import pytest
from cts1_mo_tools import magcoords as mc

# lat, lon, alt_km, Bn, Be, Bd (IRI), apex height km, apex lon, QD lat (apexpy)
REF = np.array(
    [
        [51.08, -114.13, 0.0, 15846.81, 3747.40, 52710.20, 15580.4, -50.1914, 57.4026],
        [
            51.08,
            -114.13,
            500.0,
            12658.06,
            2628.35,
            41493.47,
            17261.9,
            -49.9070,
            57.3707,
        ],
        [
            75.00,
            -100.00,
            500.0,
            3148.16,
            -224.56,
            46123.89,
            299525.3,
            -40.8632,
            81.3804,
        ],
        [78.00, 15.00, 500.0, 5781.15, 889.78, 44661.01, 106985.5, 106.6685, 75.7473],
        [65.00, -147.00, 500.0, 9748.89, 2439.65, 43741.71, 33032.2, -90.1698, 65.3178],
        [
            -70.00,
            130.00,
            500.0,
            -3406.27,
            -1707.57,
            -50684.81,
            719125.1,
            -140.2447,
            -84.4153,
        ],
        [
            -65.00,
            140.00,
            500.0,
            -23.73,
            1171.69,
            -51884.71,
            149310.1,
            -127.3434,
            -77.8727,
        ],
        [
            -75.00,
            0.00,
            500.0,
            13124.47,
            -5872.43,
            -30881.25,
            32719.9,
            43.7140,
            -65.2128,
        ],
        [
            -60.00,
            -60.00,
            500.0,
            14872.64,
            2178.89,
            -22971.56,
            8648.2,
            10.4480,
            -47.4391,
        ],
        [
            -30.00,
            -50.00,
            500.0,
            13300.61,
            -3764.98,
            -12603.92,
            1980.5,
            16.4051,
            -24.8999,
        ],
        [10.00, 0.00, 500.0, 25289.94, -886.21, -1187.82, 503.0, 74.7105, -1.2029],
        [60.00, 100.00, 800.0, 9543.11, -293.76, 41110.36, 16160.2, 174.4645, 55.6564],
        [
            -55.00,
            170.00,
            500.0,
            8860.41,
            6688.53,
            -47602.28,
            24705.1,
            -97.3373,
            -61.9517,
        ],
        [
            82.00,
            -60.00,
            500.0,
            2519.29,
            -1637.00,
            45589.66,
            1638903.5,
            59.3111,
            86.2948,
        ],
    ]
)


@pytest.fixture(scope="module")
def model() -> mc.IgrfModel:
    return mc.IgrfModel.at_year(2026.75)


def _wrap180(d: np.ndarray) -> np.ndarray:
    return np.mod(d + 180.0, 360.0) - 180.0


def test_north_geomagnetic_pole(model: mc.IgrfModel) -> None:
    plat = np.degrees(np.arcsin(model.dip_z[2]))
    plon = np.degrees(np.arctan2(model.dip_z[1], model.dip_z[0]))
    assert abs(plat - 80.8) < 0.1
    assert abs(plon + 72.8) < 0.3


def test_geodetic() -> None:
    lat, _, h = mc.ecef_to_geodetic(np.array([[6378.137, 0.0, 0.0]]))
    assert abs(lat[0]) < 1e-9
    assert abs(h[0]) < 1e-9

    polar = 6378.137 * (1.0 - 1.0 / 298.257223563) + 100.0
    lat, _, h = mc.ecef_to_geodetic(np.array([[0.0, 0.0, polar]]))
    assert abs(lat[0] - 90.0) < 1e-9
    assert abs(h[0] - 100.0) < 1e-6

    x = mc.geodetic_to_ecef(np.array([51.08]), np.array([-114.13]), np.array([512.3]))
    lat, lon, h = mc.ecef_to_geodetic(x)
    assert abs(lat[0] - 51.08) < 1e-9
    assert abs(lon[0] + 114.13) < 1e-9
    assert abs(h[0] - 512.3) < 1e-6


def test_field_matches_iri(model: mc.IgrfModel) -> None:
    lat, lon, alt = REF[:, 0], REF[:, 1], REF[:, 2]
    b = mc.igrf_field_ecef(model, mc.geodetic_to_ecef(lat, lon, alt))
    sl, cl = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    so, co = np.sin(np.radians(lon)), np.cos(np.radians(lon))
    bn = -sl * co * b[:, 0] - sl * so * b[:, 1] + cl * b[:, 2]
    be = -so * b[:, 0] + co * b[:, 1]
    bd = -cl * co * b[:, 0] - cl * so * b[:, 1] - sl * b[:, 2]
    np.testing.assert_allclose(bn, REF[:, 3], atol=0.2)
    np.testing.assert_allclose(be, REF[:, 4], atol=0.2)
    np.testing.assert_allclose(bd, REF[:, 5], atol=0.2)


def test_dipole_closed_form(model: mc.IgrfModel) -> None:
    dip = model.zero_above_degree_1()
    pts = np.array(
        [
            [55.0, -114.0, 500.0],
            [72.0, 20.0, 450.0],
            [-66.0, 140.0, 600.0],
            [-80.0, -40.0, 500.0],
            [30.0, 60.0, 500.0],
        ]
    )
    x = mc.geodetic_to_ecef(pts[:, 0], pts[:, 1], pts[:, 2])
    res = mc.quasi_dipole(dip, x, pts[:, 2])

    dlat, dlon = mc.dipole_latlon(dip, x)
    r = np.linalg.norm(x, axis=1)
    ell = r / np.cos(np.radians(dlat)) ** 2
    dl = np.radians(dlon)
    apex = ell[:, None] * (
        np.cos(dl)[:, None] * dip.dip_x + np.sin(dl)[:, None] * dip.dip_y
    )
    _, _, h_exact = mc.ecef_to_geodetic(apex)
    want = np.degrees(
        np.arccos(np.sqrt((mc.QD_RE_KM + pts[:, 2]) / (mc.QD_RE_KM + h_exact)))
    )
    want = np.where(dlat < 0, -want, want)

    np.testing.assert_allclose(res.qdlat_deg, want, atol=1e-3)
    assert np.all(np.abs(_wrap180(res.apex_lon_deg - dlon)) < 1e-3)
    assert np.all(np.abs(res.apex_height_km - h_exact) < 5e-5 * h_exact)


def test_qd_matches_apexpy(model: mc.IgrfModel) -> None:
    lat, lon, alt = REF[:, 0], REF[:, 1], REF[:, 2]
    res = mc.quasi_dipole(model, mc.geodetic_to_ecef(lat, lon, alt), alt)

    dl = np.abs(res.qdlat_deg - REF[:, 8])
    dlon = np.abs(_wrap180(res.apex_lon_deg - REF[:, 7]))
    rel = (res.apex_height_km - REF[:, 6]) / REF[:, 6]
    far = np.abs(REF[:, 8]) > 79.0
    for i in range(len(REF)):
        if far[i]:
            assert dl[i] < 0.05, i
            assert dlon[i] < 0.5, i
        else:
            assert dl[i] < 0.03, i
            assert dlon[i] < 0.02, i
            assert -2.5e-3 < rel[i] < 5e-4, i


def test_mlt_matches_apexpy(model: mc.IgrfModel) -> None:
    # apexpy's subsolar point at 2026-10-06T12:00Z and its mlon2mlt.
    sslat, sslon = np.radians(-5.22580110557268), np.radians(-2.973390710552394)
    sun = np.array(
        [[np.cos(sslat) * np.cos(sslon), np.cos(sslat) * np.sin(sslon), np.sin(sslat)]]
    )
    apex_lon = np.array([-60.0, 0.0, 95.0, 170.0])
    want = np.array(
        [3.379308573404948, 7.3793085734049475, 13.71264190673828, 18.71264190673828]
    )
    mlt = mc.magnetic_local_time(model, apex_lon, np.repeat(sun, 4, axis=0))
    d = np.abs(np.mod(mlt - want + 36.0, 24.0) - 12.0)
    assert np.all(d < 0.01)
