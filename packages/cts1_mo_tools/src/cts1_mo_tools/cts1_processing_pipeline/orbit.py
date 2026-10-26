"""Where the satellite was, and whether it was sunlit, at given times.

Each time is propagated (SGP4, with satkit) from the TLE published right
before it -- the latest one in step 7's TLE history with an epoch at or
before that time, picked with a backward `join_asof` -- so a time months
ago isn't propagated from today's TLE. A time before the first TLE in the
history gets nulls.
"""

from __future__ import annotations

__all__ = ["EPHEMERIS_SCHEMA", "ephemeris", "satkit_tle"]

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import satkit as sk

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

# One row per time asked for, in the order asked for.
EPHEMERIS_SCHEMA = {
    "time": pl.Datetime("us", "UTC"),
    "tle_epoch": pl.Datetime("us", "UTC"),  # The TLE it was propagated from.
    "latitude_deg": pl.Float64,  # Geodetic (WGS-84) sub-satellite point.
    "longitude_deg": pl.Float64,
    # How much of the Sun is visible from the satellite: 0 in Earth's umbra,
    # 1 in full sunlight, in between in its penumbra.
    "sun_fraction": pl.Float64,
}


def satkit_tle(row: dict[str, Any]) -> sk.TLE:
    """One row of `tle_history.parquet` (see
    `step_7_pipeline.TLE_HISTORY_SCHEMA`) as a satkit TLE.
    """
    tle = sk.TLE()
    tle.epoch = sk.time.from_datetime(row["epoch"])
    tle.inclination = float(row["inclination"])
    tle.raan = float(row["raan"])
    tle.eccen = float(row["eccentricity"])
    tle.arg_of_perigee = float(row["arg_perigee"])
    tle.mean_anomaly = float(row["mean_anomaly"])
    tle.mean_motion = float(row["mean_motion"])
    tle.mean_motion_dot = float(row["mean_motion_dot"])
    tle.mean_motion_dot_dot = 0.0  # Not in the dataset.
    tle.bstar = float(row["bstar"])
    tle.satnum = int(row["norad_id"])
    return tle


def _quaternions(result: object) -> list[sk.quaternion]:
    """A frame transform's result for a list of times: a list of
    quaternions -- except for just one time, when it's bare.
    """
    return cast("list[sk.quaternion]", result if isinstance(result, list) else [result])


def _propagate(tle: sk.TLE, times: list[datetime]) -> list[tuple[float, float, float]]:
    """(latitude, longitude, sun fraction) at each of `times`."""
    sk_times = [sk.time.from_datetime(t) for t in times]
    pos_teme, _ = sk.sgp4(tle, sk_times)  # pyright: ignore[reportUnknownMemberType]
    pos_teme = np.atleast_2d(pos_teme)
    q_itrf = _quaternions(sk.frametransform.qteme2itrf(sk_times))
    q_gcrf = _quaternions(sk.frametransform.qteme2gcrf(sk_times))
    # The Sun's position comes in GCRF, so the shadow check is done there.
    sun_gcrf = np.atleast_2d(sk.sun.pos_gcrf(sk_times))
    results: list[tuple[float, float, float]] = []
    for p, qi, qg, sun in zip(pos_teme, q_itrf, q_gcrf, sun_gcrf, strict=True):
        coord = sk.itrfcoord(qi * p)
        results.append(
            (coord.latitude_deg, coord.longitude_deg, sk.sun.shadowfunc(sun, qg * p))
        )
    return results


def ephemeris(tle_history: pl.DataFrame, times: Sequence[datetime]) -> pl.DataFrame:
    """The satellite's position and lighting at each of `times` (UTC), from
    `tle_history` (step 7's `tle_history.parquet`) -- see `EPHEMERIS_SCHEMA`.
    """
    if not times or tle_history.is_empty():
        return pl.DataFrame(
            {"time": list(times)}, schema={"time": EPHEMERIS_SCHEMA["time"]}
        ).with_columns(
            pl.lit(None, dtype).alias(name)
            for name, dtype in EPHEMERIS_SCHEMA.items()
            if name != "time"
        )
    matched = (
        pl.DataFrame({"time": list(times)}, schema={"time": EPHEMERIS_SCHEMA["time"]})
        .with_row_index("_order")
        .sort("time")
        .join_asof(
            tle_history.sort("epoch").with_columns(tle_epoch=pl.col("epoch")),
            left_on="time",
            right_on="epoch",
            strategy="backward",
        )
    )
    propagated: list[dict[str, Any]] = []
    for (tle_epoch,), group in matched.filter(
        pl.col("tle_epoch").is_not_null()
    ).group_by("tle_epoch"):
        group_times = group["time"].to_list()
        for order, (lat, lon, sun) in zip(
            group["_order"],
            _propagate(satkit_tle(group.row(0, named=True)), group_times),
            strict=True,
        ):
            propagated.append(
                {
                    "_order": order,
                    "tle_epoch": tle_epoch,
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "sun_fraction": sun,
                }
            )
    results = pl.DataFrame(
        propagated,
        schema={"_order": pl.UInt32, **dict(list(EPHEMERIS_SCHEMA.items())[1:])},
    )
    return (
        matched.select("_order", "time")
        .join(results, on="_order", how="left")
        .sort("_order")
        .select(*EPHEMERIS_SCHEMA)
    )
