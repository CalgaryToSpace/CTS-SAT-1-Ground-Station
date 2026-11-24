from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.attitude_sparkline import (
    attitude_sparkline_svg,
)

_START = datetime(2026, 11, 14, 8, 33, 30, tzinfo=UTC)
_END = _START + timedelta(minutes=2)


def _beacons(*rows: tuple[datetime, float | None, float | None]) -> pl.DataFrame:
    """(time, pitch, yaw) per beacon; roll is always 0."""
    return pl.DataFrame(
        {
            "unix_epoch_time_ms": [int(t.timestamp() * 1000) for t, _, _ in rows],
            "adcs_estimated_roll_angle_deg": [
                None if p is None else 0.0 for _, p, _ in rows
            ],
            "adcs_estimated_pitch_angle_deg": [p for _, p, _ in rows],
            "adcs_estimated_yaw_angle_deg": [y for _, _, y in rows],
        },
        schema_overrides={
            "adcs_estimated_roll_angle_deg": pl.Float64,
            "adcs_estimated_pitch_angle_deg": pl.Float64,
            "adcs_estimated_yaw_angle_deg": pl.Float64,
        },
    )


def test_no_attitude_in_the_window_is_no_plot() -> None:
    far = _END + timedelta(hours=1)
    assert attitude_sparkline_svg(_beacons((far, 1.0, 1.0)), _START, _END) == ""
    assert attitude_sparkline_svg(_beacons((_START, None, None)), _START, _END) == ""


def test_plot_has_a_point_per_angle_and_breaks_at_yaw_wrap() -> None:
    mid = _START + timedelta(minutes=1)
    svg = attitude_sparkline_svg(
        _beacons((_START, -80.0, 179.0), (mid, 0.0, -179.0), (_END, 60.0, -178.0)),
        _START,
        _END,
    )
    assert svg.startswith("<svg")
    assert svg.count("<circle") == 9
    # Roll and pitch are one line each; yaw is split where it wraps (179 ->
    # -179), leaving one 2-point line and a lone point.
    assert svg.count("<polyline") == 3
