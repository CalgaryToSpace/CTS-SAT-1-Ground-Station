"""The "Overpasses" page: upcoming passes of the satellite over the uplink
ground station (RAO by default), predicted from SatNOGS DB's latest TLE --
see `overpasses` for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_overpasses_page"]

import zoneinfo
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from loguru import logger
from nicegui import run, ui

from . import overpasses
from .layout import page_shell

DEFAULT_DAYS_AHEAD = 2.0
COUNTDOWN_REFRESH_INTERVAL_SEC = 1.0

_DATETIME_FORMAT = "%a %Y-%m-%d %H:%M:%S"


@dataclass(slots=True)
class _Prediction:
    """The last successful prediction this page rendered."""

    tle: overpasses.SatnogsTle
    station: overpasses.GroundStation
    passes: list[overpasses.Overpass]


def _duration_str(delta: timedelta) -> str:
    total_sec = int(delta.total_seconds())
    days, rem = divmod(total_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _direction_str(azimuth_deg: float) -> str:
    return f"{overpasses.compass_direction(azimuth_deg)} ({azimuth_deg:.0f}°)"


def _pass_rows(
    prediction: _Prediction, tz: zoneinfo.ZoneInfo, now: datetime
) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "status": (
                "In progress"
                if p.aos <= now < p.los
                else "Done"
                if p.los <= now
                else "Upcoming"
            ),
            "aos_utc": f"{p.aos:{_DATETIME_FORMAT}}",
            "los_utc": f"{p.los:{_DATETIME_FORMAT}}",
            "aos_local": f"{p.aos.astimezone(tz):{_DATETIME_FORMAT}}",
            "los_local": f"{p.los.astimezone(tz):{_DATETIME_FORMAT}}",
            "duration": _duration_str(p.duration),
            "max_elevation": f"{p.max_elevation_deg:.1f}°",
            "direction": (
                f"{_direction_str(p.aos_azimuth_deg)} → "
                f"{_direction_str(p.los_azimuth_deg)}"
            ),
        }
        for i, p in enumerate(prediction.passes, start=1)
    ]


def _passes_table(prediction: _Prediction) -> None:
    station = prediction.station
    tz = zoneinfo.ZoneInfo(station.timezone)
    now = datetime.now(UTC)

    with ui.card().classes("w-full"):
        ui.label(f"Uplink Overpasses over {station.name}").classes("text-lg font-bold")
        ui.label(
            f"{station.latitude_deg:.4f}°, {station.longitude_deg:.4f}°, "
            f"{station.altitude_m:.0f} m -- AOS/LOS are when "
            f"{prediction.tle.name} crosses the horizon (or the minimum "
            "elevation set above)."
        ).classes("text-caption text-grey")

        if not prediction.passes:
            ui.label("No passes in the selected window.").classes(
                "text-caption text-grey mt-2"
            )
            return

        columns = [
            {"name": "index", "label": "#", "field": "index", "align": "right"},
            {"name": "status", "label": "Status", "field": "status"},
            {"name": "aos_local", "label": f"AOS ({tz.key})", "field": "aos_local"},
            {"name": "los_local", "label": f"LOS ({tz.key})", "field": "los_local"},
            {"name": "aos_utc", "label": "AOS (UTC)", "field": "aos_utc"},
            {"name": "los_utc", "label": "LOS (UTC)", "field": "los_utc"},
            {"name": "duration", "label": "Duration", "field": "duration"},
            {
                "name": "max_elevation",
                "label": "Max Elevation",
                "field": "max_elevation",
                "align": "right",
            },
            {
                "name": "direction",
                "label": "Direction (AOS → LOS)",
                "field": "direction",
            },
        ]
        for column in columns:  # Quasar right-aligns by default.
            column.setdefault("align", "left")
        table = ui.table(
            columns=columns,
            rows=_pass_rows(prediction, tz, now),
            row_key="index",
            pagination=0,
        ).classes("w-full")
        table.add_slot(
            "body-cell-status",
            r"""
                <q-td :props="props">
                    <q-badge :color="{
                        'In progress': 'positive',
                        'Upcoming': 'primary',
                        'Done': 'grey',
                    }[props.value] || 'grey'">
                        {{ props.value }}
                    </q-badge>
                </q-td>
            """,
        )
        ui.label(f"{len(prediction.passes):,} pass(es).").classes(
            "text-caption text-grey mt-2"
        )


def _next_pass_summary(prediction: _Prediction) -> str:
    now = datetime.now(UTC)
    tz = zoneinfo.ZoneInfo(prediction.station.timezone)
    for p in prediction.passes:
        if p.aos <= now < p.los:
            return (
                f"Pass in progress over {prediction.station.name}: LOS in "
                f"{_duration_str(p.los - now)} (max elevation "
                f"{p.max_elevation_deg:.1f}°)."
            )
        if now < p.aos:
            return (
                f"Next pass over {prediction.station.name}: AOS in "
                f"{_duration_str(p.aos - now)}, at "
                f"{p.aos.astimezone(tz):%a %H:%M:%S %Z} (max elevation "
                f"{p.max_elevation_deg:.1f}°, {_duration_str(p.duration)} long)."
            )
    return "No upcoming passes in the selected window."


def _tle_card(prediction: _Prediction) -> None:
    tle = prediction.tle
    now = datetime.now(UTC)
    with ui.card().classes("w-full"):
        ui.label("TLE (from SatNOGS DB)").classes("text-lg font-bold")
        ui.label(
            f"{tle.name} -- epoch {tle.epoch:%Y-%m-%d %H:%M:%S} UTC "
            f"({_duration_str(now - tle.epoch)} old), source: {tle.source}, "
            f"last updated in SatNOGS DB {tle.updated_at:%Y-%m-%d %H:%M:%S} UTC."
        ).classes("text-caption text-grey")
        ui.code(f"{tle.line1}\n{tle.line2}", language="text").classes("w-full")


def _fetch_and_predict(
    norad_id: str,
    station: overpasses.GroundStation,
    start: datetime,
    end: datetime,
    *,
    min_elevation_deg: float,
) -> _Prediction:
    """Blocking -- run it via `run.io_bound`."""
    tle = overpasses.fetch_satnogs_tle(norad_id)
    passes = overpasses.compute_overpasses(
        tle, station, start, end, min_elevation_deg=min_elevation_deg
    )
    return _Prediction(tle=tle, station=station, passes=passes)


def build_overpasses_page() -> None:  # noqa: PLR0915
    default = overpasses.RAO
    state: dict[str, _Prediction | None] = {"prediction": None}

    with page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Overpasses").classes("text-2xl font-bold")
            refresh_button = ui.button("Refresh", icon="refresh")

        with ui.card().classes("w-full"):
            ui.label("Prediction settings").classes("text-lg font-bold")
            with ui.row().classes("gap-4 flex-wrap items-end mt-2"):
                norad_id = ui.input(
                    "Satellite NORAD ID", value=overpasses.DEFAULT_NORAD_ID
                ).classes("w-36")
                station_name = ui.input("Station", value=default.name).classes("w-28")
                latitude = ui.number(
                    "Latitude (°N)",
                    value=default.latitude_deg,
                    min=-90,
                    max=90,
                    format="%.4f",
                ).classes("w-32")
                longitude = ui.number(
                    "Longitude (°E)",
                    value=default.longitude_deg,
                    min=-180,
                    max=180,
                    format="%.4f",
                ).classes("w-32")
                altitude = ui.number(
                    "Altitude (m)", value=default.altitude_m, format="%.0f"
                ).classes("w-28")
                timezone = ui.select(
                    sorted(zoneinfo.available_timezones()),
                    label="Local time zone",
                    value=default.timezone,
                    with_input=True,
                ).classes("w-56")
                days_ahead = ui.number(
                    "Days ahead", value=DEFAULT_DAYS_AHEAD, min=0.1, max=14, step=1
                ).classes("w-28")
                min_elevation = ui.number(
                    "Min elevation (°)", value=0.0, min=0, max=90, step=1
                ).classes("w-32")

            def _reset_station() -> None:
                station_name.value = default.name
                latitude.value = default.latitude_deg
                longitude.value = default.longitude_deg
                altitude.value = default.altitude_m
                timezone.value = default.timezone

            with ui.row().classes("gap-2 mt-2"):
                predict_button = ui.button("Predict", icon="satellite_alt")
                ui.button(
                    f"Reset to {default.name}",
                    icon="restart_alt",
                    on_click=_reset_station,
                ).props("flat")

        status_label = ui.label("").classes("text-body1 font-bold")
        error_label = ui.label("").classes("text-negative")
        results = ui.column().classes("w-full gap-4")

    def _update_status() -> None:
        prediction = state["prediction"]
        if prediction is not None:
            status_label.text = _next_pass_summary(prediction)

    async def _predict() -> None:
        error_label.text = ""
        sat_id = (norad_id.value or "").strip()
        if (
            not sat_id
            or not timezone.value
            or latitude.value is None
            or longitude.value is None
            or altitude.value is None
        ):
            error_label.text = "Fill in every prediction setting first."
            return
        station = overpasses.GroundStation(
            name=(station_name.value or "").strip() or "Station",
            latitude_deg=float(latitude.value),
            longitude_deg=float(longitude.value),
            altitude_m=float(altitude.value),
            timezone=timezone.value,
        )
        start = datetime.now(UTC)
        end = start + timedelta(days=float(days_ahead.value or DEFAULT_DAYS_AHEAD))

        results.clear()
        with results, ui.row().classes("items-center gap-2 p-2"):
            ui.spinner(size="md")
            ui.label("Fetching TLE from SatNOGS DB and predicting passes...")
        predict_button.disable()
        refresh_button.disable()
        try:
            prediction = await run.io_bound(
                _fetch_and_predict,
                sat_id,
                station,
                start,
                end,
                min_elevation_deg=float(min_elevation.value or 0.0),
            )
        except (requests.RequestException, LookupError, ValueError, RuntimeError) as e:
            logger.exception("Overpass prediction failed")
            results.clear()
            error_label.text = f"Prediction failed: {e}"
            return
        finally:
            predict_button.enable()
            refresh_button.enable()
        if prediction is None:  # Only when the server's shutting down.
            return

        state["prediction"] = prediction
        results.clear()
        with results:
            _passes_table(prediction)
            _tle_card(prediction)
        _update_status()

    predict_button.on_click(_predict)
    refresh_button.on_click(_predict)
    ui.timer(COUNTDOWN_REFRESH_INTERVAL_SEC, _update_status)
    ui.timer(0.0, _predict, once=True)
