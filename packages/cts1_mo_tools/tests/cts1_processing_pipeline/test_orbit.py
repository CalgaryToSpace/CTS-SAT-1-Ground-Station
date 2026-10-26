from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.orbit import ephemeris
from cts1_mo_tools.cts1_processing_pipeline.step_7_fetch_tle_history.pipeline import (
    TLE_HISTORY_SCHEMA,
)

# Two real consecutive TLEs, and a GNSS fix the satellite reported between
# them.
_EPOCH_BEFORE = datetime(2026, 10, 5, 22, 30, 27, 475200, tzinfo=UTC)
_EPOCH_AFTER = datetime(2026, 10, 6, 12, 41, 51, 514944, tzinfo=UTC)
_TLE_BEFORE = {
    "norad_id": 69015,
    "epoch": _EPOCH_BEFORE,
    "inclination": 97.38469696044922,
    "raan": 174.98910522460938,
    "eccentricity": 0.0008776999893598258,
    "arg_perigee": 27.310199737548828,
    "mean_anomaly": 332.8594055175781,
    "mean_motion": 15.23169593,
    "mean_motion_dot": 6.928e-05,
    "bstar": 0.00029581,
    "intl_designator": "26100AM",
    "altitude_km": 497.39,
}
_TLE_AFTER = {
    **_TLE_BEFORE,
    "epoch": _EPOCH_AFTER,
    "raan": 175.57080078125,
    "eccentricity": 0.0008688999805599451,
    "arg_perigee": 25.40719985961914,
    "mean_anomaly": 334.7590026855469,
    "mean_motion": 15.23175446,
    "mean_motion_dot": 5.847e-05,
    "bstar": 0.00025009,
}
TLE_HISTORY = pl.DataFrame([_TLE_BEFORE, _TLE_AFTER], schema=TLE_HISTORY_SCHEMA)
GNSS_FIX_AT = datetime(2026, 10, 6, 1, 42, 8, tzinfo=UTC)
_GNSS_FIX_LAT_LON = (9.422838, 133.620262)


def test_propagates_from_the_tle_right_before_each_time() -> None:
    later = _EPOCH_AFTER + timedelta(minutes=1)
    at_fix, at_later = ephemeris(TLE_HISTORY, [GNSS_FIX_AT, later]).rows(named=True)
    assert at_fix["tle_epoch"] == _EPOCH_BEFORE
    assert at_later["tle_epoch"] == _EPOCH_AFTER
    # Within ~1 km of where the satellite's own GNSS said it was.
    assert abs(at_fix["latitude_deg"] - _GNSS_FIX_LAT_LON[0]) < 0.02
    assert abs(at_fix["longitude_deg"] - _GNSS_FIX_LAT_LON[1]) < 0.02
    assert at_fix["sun_fraction"] == 1.0  # Mid-morning, local time.


def test_times_keep_their_order_and_need_an_earlier_tle() -> None:
    too_early = _EPOCH_BEFORE - timedelta(days=1)
    times = [GNSS_FIX_AT + timedelta(seconds=10), too_early, GNSS_FIX_AT]
    result = ephemeris(TLE_HISTORY, times)
    assert result["time"].to_list() == times
    assert result["latitude_deg"][1] is None
    assert result["tle_epoch"][1] is None
    # Northbound: a bit further north 10 s later.
    assert result["latitude_deg"][0] > result["latitude_deg"][2]


def test_no_tles_means_no_positions() -> None:
    result = ephemeris(pl.DataFrame(schema=TLE_HISTORY_SCHEMA), [GNSS_FIX_AT])
    assert result.rows() == [(GNSS_FIX_AT, None, None, None, None)]
