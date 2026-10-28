from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_7_fetch_tle_history import (
    pipeline as step_7_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.overpasses import (
    RAO,
    Tle,
    _LookAngleCalculator,  # pyright: ignore[reportPrivateUsage]
    compass_direction,
    compute_overpasses,
    load_latest_tle,
)

# FrontierSat's TLE from 2026-09-25, as a row of `tle_history.parquet`:
# 1 69015U 26100AM  26267.70358655  .00007751  00000-0  33174-3 0  9993
# 2 69015  97.3863 163.9374 0009690  60.6092 299.6112 15.23036761 21961
TLE_EPOCH = datetime(2026, 9, 24, 16, 53, 9, 878000, tzinfo=UTC)
TLE_ROW: dict[str, Any] = {
    "norad_id": 69015,
    "epoch": TLE_EPOCH,
    "inclination": 97.3863,
    "raan": 163.9374,
    "eccentricity": 0.0009690,
    "arg_perigee": 60.6092,
    "mean_anomaly": 299.6112,
    "mean_motion": 15.23036761,
    "mean_motion_dot": 0.00007751,
    "bstar": 0.00033174,
    "intl_designator": "26100AM",
    "altitude_km": 500.0,
}
TLE = Tle(TLE_ROW)
START = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def _write_tle_history(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows, schema=step_7_pipeline.TLE_HISTORY_SCHEMA).write_parquet(
        data_dir / step_7_pipeline.OUTPUT_FILENAME
    )


def test_load_latest_tle_picks_latest_epoch_for_the_norad_id(tmp_path: Path) -> None:
    newer = {**TLE_ROW, "epoch": TLE_EPOCH + timedelta(days=1)}
    other_sat = {**TLE_ROW, "norad_id": 1, "epoch": TLE_EPOCH + timedelta(days=2)}
    _write_tle_history(tmp_path, [newer, TLE_ROW, other_sat])

    assert load_latest_tle(tmp_path, "69015").epoch == TLE_EPOCH + timedelta(days=1)


def test_load_latest_tle_raises_without_history(tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        load_latest_tle(tmp_path, "69015")
    _write_tle_history(tmp_path, [{**TLE_ROW, "norad_id": 1}])
    with pytest.raises(LookupError):
        load_latest_tle(tmp_path, "69015")


def test_tle_lines() -> None:
    line1, line2 = TLE.lines()
    assert line1.startswith("1 69015U")
    assert "26267.70358655" in line1
    assert line2.startswith(
        "2 69015  97.3863 163.9374 0009690  60.6092 299.6112 15.2303676"
    )


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
