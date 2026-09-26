"""The "Overpasses" page: upcoming passes of the satellite over the uplink
ground station (RAO by default), predicted from SatNOGS DB's latest TLE --
see `overpasses` for the data layer.

Each pass can be ticked to get a browser notification shortly before (and
at) its AOS. Notifications are sent from this page, so it has to stay open
(a background tab is fine) for them to arrive.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["build_overpasses_page"]

import json
import zoneinfo
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from loguru import logger
from nicegui import app, events, run, ui

from . import overpasses
from .layout import page_shell

DEFAULT_DAYS_AHEAD = 2.0
COUNTDOWN_REFRESH_INTERVAL_SEC = 1.0

DEFAULT_NOTIFY_LEAD_MIN = 5.0

# Remembered per-browser (in `app.storage.user`) so the ticked passes survive
# page reloads and later visits.
_NOTIFY_AOS_STORAGE_KEY = "overpass_notify_aos"
_NOTIFY_LEAD_STORAGE_KEY = "overpass_notify_lead_min"
# A ticked pass is matched to freshly-predicted passes by AOS: a newer TLE
# shifts a pass's AOS by seconds, while consecutive passes are ~90 min apart.
_NOTIFY_MATCH_TOLERANCE = timedelta(minutes=5)
# A notification is only sent if the page notices it's due within this long
# of its trigger time, so reopening the page later doesn't send stale ones.
_NOTIFY_FIRE_WINDOW = timedelta(minutes=1)
# Ticked passes this long past their AOS are forgotten.
_NOTIFY_FORGET_AFTER = timedelta(days=1)

_DATETIME_FORMAT = "%a %Y-%m-%d %H:%M:%S"

# Evaluated in the browser; yields "granted"/"denied"/"default", or
# "insecure"/"unsupported" if the browser won't do notifications here at all.
_NOTIFY_PERMISSION_JS = """
    !window.isSecureContext ? 'insecure'
    : !('Notification' in window) ? 'unsupported'
    : Notification.permission
"""
# `js_handler` for the "Send test notification" button: runs in the browser,
# within the click, since browsers only prompt for permission on a user gesture.
_TEST_NOTIFICATION_JS_HANDLER = """
    async () => {
        let permission = PERMISSION_JS;
        if (permission === 'default') {
            permission = await Notification.requestPermission();
        }
        if (permission === 'granted') {
            new Notification('CTS-SAT-1 test notification', {
                body: 'Browser notifications are working.',
            });
        }
        emit(permission);
    }
""".replace("PERMISSION_JS", _NOTIFY_PERMISSION_JS.strip())
# `js_handler` for a pass's checkbox: asks for permission (within the click)
# the first time a pass is ticked, then passes the event on to the server.
_NOTIFY_TOGGLE_JS_HANDLER = """
    (e) => {
        if (e.value && window.isSecureContext && 'Notification' in window
                && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        emit(e);
    }
"""
_SEND_NOTIFICATION_JS = """
    if (window.isSecureContext && 'Notification' in window
            && Notification.permission === 'granted') {
        new Notification(%s, {body: %s, tag: %s, requireInteraction: true});
    }
"""

_PERMISSION_DESCRIPTIONS = {
    "granted": "Browser notifications are allowed.",
    "denied": (
        "Browser notifications are blocked -- re-allow them in this site's "
        "settings in your browser."
    ),
    "default": (
        "Browser notifications aren't allowed yet -- click "
        '"Send test notification" to allow them.'
    ),
    "insecure": (
        "Browser notifications need the page served over HTTPS (or from localhost)."
    ),
    "unsupported": "This browser doesn't support notifications.",
}


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


def _matching_notify_aos(aos: datetime, notify_aos: list[datetime]) -> datetime | None:
    """The ticked AOS (from `notify_aos`) that `aos` is the same pass as, if
    any.
    """
    for ticked in notify_aos:
        if abs(aos - ticked) <= _NOTIFY_MATCH_TOLERANCE:
            return ticked
    return None


def _due_notifications(
    passes: list[overpasses.Overpass],
    notify_aos: list[datetime],
    lead: timedelta,
    now: datetime,
    already_sent: set[tuple[datetime, str]],
) -> list[tuple[tuple[datetime, str], overpasses.Overpass, str]]:
    """The notifications due at `now`, as `(key, pass, kind)`, where `kind` is
    "lead" (`lead` before AOS) or "aos" (at AOS), and `key` (for
    `already_sent`) stays the same when a re-prediction shifts the pass.
    """
    due: list[tuple[tuple[datetime, str], overpasses.Overpass, str]] = []
    for p in passes:
        ticked = _matching_notify_aos(p.aos, notify_aos)
        if ticked is None:
            continue
        triggers = [("aos", p.aos)]
        if lead > timedelta(0):
            triggers.insert(0, ("lead", p.aos - lead))
        for kind, trigger_at in triggers:
            key = (ticked, kind)
            if key not in already_sent and (
                trigger_at <= now < trigger_at + _NOTIFY_FIRE_WINDOW
            ):
                due.append((key, p, kind))
    return due


def _load_notify_aos() -> list[datetime]:
    return [
        datetime.fromisoformat(s)
        for s in app.storage.user.get(_NOTIFY_AOS_STORAGE_KEY, [])
    ]


def _save_notify_aos(notify_aos: list[datetime]) -> None:
    cutoff = datetime.now(UTC) - _NOTIFY_FORGET_AFTER
    app.storage.user[_NOTIFY_AOS_STORAGE_KEY] = sorted(
        a.isoformat() for a in notify_aos if a >= cutoff
    )


def _pass_rows(
    prediction: _Prediction,
    tz: zoneinfo.ZoneInfo,
    now: datetime,
    notify_aos: list[datetime],
) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "notify": _matching_notify_aos(p.aos, notify_aos) is not None,
            "aos_iso": p.aos.isoformat(),
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


def _passes_table(
    prediction: _Prediction,
    on_notify_toggle: events.Handler[events.GenericEventArguments],
) -> ui.table | None:
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
            return None

        columns = [
            {"name": "index", "label": "#", "field": "index", "align": "right"},
            {"name": "notify", "label": "Notify", "field": "notify"},
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
            rows=_pass_rows(prediction, tz, now, _load_notify_aos()),
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
        table.add_slot(
            "body-cell-notify",
            r"""
                <q-td :props="props" auto-width>
                    <q-checkbox
                        dense
                        :model-value="props.value"
                        :disable="props.row.status === 'Done'"
                        @update:model-value="value => $parent.$emit(
                            'notify_toggle', {aos: props.row.aos_iso, value}
                        )"
                    />
                </q-td>
            """,
        )
        table.on(
            "notify_toggle", on_notify_toggle, js_handler=_NOTIFY_TOGGLE_JS_HANDLER
        )
        ui.label(f"{len(prediction.passes):,} pass(es).").classes(
            "text-caption text-grey mt-2"
        )
    return table


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


def _send_notification(
    prediction: _Prediction, p: overpasses.Overpass, kind: str
) -> None:
    tz = zoneinfo.ZoneInfo(prediction.station.timezone)
    now = datetime.now(UTC)
    where = f"{prediction.tle.name} over {prediction.station.name}"
    title = (
        f"{where}: AOS now"
        if kind == "aos"
        else f"{where}: AOS in {_duration_str(p.aos - now)}"
    )
    body = (
        f"AOS {p.aos.astimezone(tz):%H:%M:%S %Z}, "
        f"LOS {p.los.astimezone(tz):%H:%M:%S %Z} "
        f"({_duration_str(p.duration)}), max elevation "
        f"{p.max_elevation_deg:.1f}°, "
        f"{overpasses.compass_direction(p.aos_azimuth_deg)} → "
        f"{overpasses.compass_direction(p.los_azimuth_deg)}."
    )
    ui.run_javascript(
        _SEND_NOTIFICATION_JS
        % (json.dumps(title), json.dumps(body), json.dumps(f"overpass-{kind}"))
    )
    # Also shown in the page, in case browser notifications aren't allowed.
    ui.notify(f"{title}. {body}", type="info", timeout=0, close_button=True)


def build_overpasses_page() -> None:  # noqa: C901, PLR0915
    default = overpasses.RAO
    state: dict[str, _Prediction | None] = {"prediction": None}
    tables: dict[str, ui.table | None] = {"passes": None}
    sent_notifications: set[tuple[datetime, str]] = set()
    app.storage.user.setdefault(_NOTIFY_LEAD_STORAGE_KEY, DEFAULT_NOTIFY_LEAD_MIN)

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

        with ui.card().classes("w-full"):
            ui.label("Notifications").classes("text-lg font-bold")
            ui.label(
                "Tick a pass's Notify box to get a browser notification before "
                "its AOS, and again at AOS. Keep this page open (a background "
                "tab is fine) for notifications to arrive."
            ).classes("text-caption text-grey")
            with ui.row().classes("gap-4 flex-wrap items-center mt-2"):
                ui.number(
                    "Minutes before AOS", min=0, max=120, step=1, format="%.0f"
                ).bind_value(app.storage.user, _NOTIFY_LEAD_STORAGE_KEY).classes("w-40")
                test_button = ui.button(
                    "Send test notification", icon="notifications"
                ).props("outline")
                permission_label = ui.label("").classes("text-caption")

        status_label = ui.label("").classes("text-body1 font-bold")
        error_label = ui.label("").classes("text-negative")
        results = ui.column().classes("w-full gap-4")

    def _show_permission(permission: str) -> None:
        permission_label.text = _PERMISSION_DESCRIPTIONS.get(
            permission, f"Browser notification permission: {permission}."
        )
        permission_label.classes(
            replace="text-caption "
            + ("text-positive" if permission == "granted" else "text-warning")
        )

    async def _refresh_permission() -> None:
        _show_permission(str(await ui.run_javascript(_NOTIFY_PERMISSION_JS)))

    def _on_test_notification(e: events.GenericEventArguments) -> None:
        permission = str(e.args)
        _show_permission(permission)
        if permission != "granted":
            ui.notify(
                _PERMISSION_DESCRIPTIONS.get(permission, permission), type="warning"
            )

    def _on_notify_toggle(e: events.GenericEventArguments) -> None:
        aos = datetime.fromisoformat(e.args["aos"])
        notify_aos = [
            a for a in _load_notify_aos() if abs(a - aos) > _NOTIFY_MATCH_TOLERANCE
        ]
        if e.args["value"]:
            notify_aos.append(aos)
        _save_notify_aos(notify_aos)
        table = tables["passes"]
        if table is not None:
            for row in table.rows:
                if row["aos_iso"] == e.args["aos"]:
                    row["notify"] = bool(e.args["value"])
            table.update()
        ui.timer(0.0, _refresh_permission, once=True)

    def _check_notifications() -> None:
        prediction = state["prediction"]
        if prediction is None:
            return
        lead_min = app.storage.user.get(_NOTIFY_LEAD_STORAGE_KEY)
        lead = timedelta(minutes=float(lead_min or 0.0))
        for key, p, kind in _due_notifications(
            prediction.passes,
            _load_notify_aos(),
            lead,
            datetime.now(UTC),
            sent_notifications,
        ):
            sent_notifications.add(key)
            _send_notification(prediction, p, kind)

    def _update_status() -> None:
        prediction = state["prediction"]
        if prediction is not None:
            status_label.text = _next_pass_summary(prediction)
        _check_notifications()

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
            tables["passes"] = _passes_table(prediction, _on_notify_toggle)
            _tle_card(prediction)
        _update_status()

    test_button.on(
        "click", _on_test_notification, js_handler=_TEST_NOTIFICATION_JS_HANDLER
    )
    predict_button.on_click(_predict)
    refresh_button.on_click(_predict)
    ui.timer(COUNTDOWN_REFRESH_INTERVAL_SEC, _update_status)
    ui.timer(0.0, _predict, once=True)
    ui.timer(0.0, _refresh_permission, once=True)
