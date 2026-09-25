from datetime import UTC, datetime, timedelta

import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.overpasses import (
    RAO,
    SatnogsTle,
    _LookAngleCalculator,  # pyright: ignore[reportPrivateUsage]
    compass_direction,
    compute_overpasses,
)

# FrontierSat's TLE as SatNOGS DB served it on 2026-09-25.
TLE = SatnogsTle(
    name="FRONTIERSAT",
    line1="1 69015U 26100AM  26267.70358655  .00007751  00000-0  33174-3 0  9993",
    line2="2 69015  97.3863 163.9374 0009690  60.6092 299.6112 15.23036761 21961",
    source="Space-Track.org",
    updated_at=datetime(2026, 9, 25, 3, 23, 41, tzinfo=UTC),
)
START = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def test_tle_epoch() -> None:
    expected = datetime(2026, 9, 24, 16, 53, 9, 878000, tzinfo=UTC)
    assert abs(TLE.epoch - expected) < timedelta(milliseconds=1)


def test_compute_overpasses_over_rao() -> None:
    passes = compute_overpasses(TLE, RAO, START, START + timedelta(days=2))

    # A ~97° sun-synchronous orbit gets ~6 passes/day over 51°N.
    assert 10 <= len(passes) <= 14
    calc = _LookAngleCalculator(TLE, RAO)
    for p in passes:
        assert p.aos < p.max_elevation_at < p.los
        assert timedelta(minutes=1) < p.duration < timedelta(minutes=15)
        assert 0 < p.max_elevation_deg <= 90
        # AOS/LOS are refined to the horizon crossing itself.
        assert abs(calc.elevation_at(p.aos)) < 0.01
        assert abs(calc.elevation_at(p.los)) < 0.01
        assert calc.elevation_at(p.aos - timedelta(seconds=5)) < 0
        assert calc.elevation_at(p.los + timedelta(seconds=5)) < 0
    assert passes == sorted(passes, key=lambda p: p.aos)


def test_min_elevation_filters_and_shortens_passes() -> None:
    end = START + timedelta(days=2)
    horizon = compute_overpasses(TLE, RAO, START, end)
    above_30 = compute_overpasses(TLE, RAO, START, end, min_elevation_deg=30)

    assert 0 < len(above_30) < len(horizon)
    for p in above_30:
        assert p.max_elevation_deg >= 30
        matching = next(h for h in horizon if h.aos < p.aos and p.los < h.los)
        assert abs(matching.max_elevation_deg - p.max_elevation_deg) < 0.1


def test_pass_in_progress_at_start_is_reported_whole() -> None:
    first = compute_overpasses(TLE, RAO, START, START + timedelta(days=1))[0]
    mid_pass = first.aos + first.duration / 2

    passes = compute_overpasses(TLE, RAO, mid_pass, mid_pass + timedelta(hours=1))

    assert abs(passes[0].aos - first.aos) < timedelta(seconds=0.1)
    assert abs(passes[0].los - first.los) < timedelta(seconds=0.1)


@pytest.mark.parametrize(
    ("azimuth", "expected"),
    [(0, "N"), (11, "N"), (12, "NNE"), (90, "E"), (200, "SSW"), (350, "N")],
)
def test_compass_direction(azimuth: float, expected: str) -> None:
    assert compass_direction(azimuth) == expected
