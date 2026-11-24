"""A tiny inline-SVG plot of the ADCS attitude estimate (roll/pitch/yaw)
over one MPI collection's beacon window -- for a cell of the MPI Data
page's collections table.
"""

from __future__ import annotations

__all__ = ["attitude_sparkline_svg"]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    beacons as beacon_summary,
)

if TYPE_CHECKING:
    from datetime import datetime

_WIDTH, _HEIGHT, _PAD = 140, 36, 3  # px
# x/y/z (roll/pitch/yaw) in red/green/blue, as on the MPI Vis page's chart.
_AXES = {
    "roll": "#ef4444",
    "pitch": "#22c55e",
    "yaw": "#3b82f6",
}
# The narrowest the y axis gets, so steady pointing looks steady rather
# than its noise filling the plot.
_MIN_SPAN_DEG = 20.0
_TIME_MS = "unix_epoch_time_ms"


def attitude_sparkline_svg(
    beacons: pl.DataFrame, started_at: datetime, ended_at: datetime
) -> str:
    """`beacons` (from `beacons.load_beacons`) around the collection
    `started_at`..`ended_at` as an SVG: each angle over the beacon window,
    the recording itself shaded. "" when no beacon there has an attitude.
    """
    margin = beacon_summary.BEACON_WINDOW_MARGIN
    t0 = (started_at - margin).timestamp() * 1000
    t1 = (ended_at + margin).timestamp() * 1000
    window = beacons.filter(pl.col(_TIME_MS).is_between(t0, t1))
    series = {
        axis: [
            (t, v)
            for t, v in window.select(
                _TIME_MS, f"adcs_estimated_{axis}_angle_deg"
            ).iter_rows()
            if v is not None
        ]
        for axis in _AXES
    }
    values = [v for points in series.values() for _, v in points]
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < _MIN_SPAN_DEG:
        middle = (high + low) / 2
        low, high = middle - _MIN_SPAN_DEG / 2, middle + _MIN_SPAN_DEG / 2

    def x(t: float) -> float:
        return _PAD + (t - t0) / (t1 - t0) * (_WIDTH - 2 * _PAD)

    def y(v: float) -> float:
        return _HEIGHT - _PAD - (v - low) / (high - low) * (_HEIGHT - 2 * _PAD)

    start_x, end_x = (x(at.timestamp() * 1000) for at in (started_at, ended_at))
    recording = (
        f'<rect x="{start_x:.1f}" y="0" width="{end_x - start_x:.1f}" '
        f'height="{_HEIGHT}" fill="rgba(128,128,128,0.18)"/>'
    )
    marks: list[str] = []
    for axis, color in _AXES.items():
        points = series[axis]
        # A new line wherever the angle wraps through +-180.
        runs: list[list[tuple[float, float]]] = []
        for i, (t, v) in enumerate(points):
            if i == 0 or abs(v - points[i - 1][1]) > 180:  # noqa: PLR2004
                runs.append([])
            runs[-1].append((t, v))
        for run in runs:
            if len(run) > 1:
                path = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in run)
                marks.append(
                    f'<polyline points="{path}" fill="none" stroke="{color}" '
                    'stroke-width="1.2"/>'
                )
        marks += [
            f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="1.5" fill="{color}"/>'
            for t, v in points
        ]
    title = (
        f"Attitude (roll red, pitch green, yaw blue): {low:.0f}° to {high:.0f}°; "
        "the recording shaded"
    )
    return (
        f'<svg width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'role="img" aria-label="{title}"><title>{title}</title>'
        + recording
        + "".join(marks)
        + "</svg>"
    )
