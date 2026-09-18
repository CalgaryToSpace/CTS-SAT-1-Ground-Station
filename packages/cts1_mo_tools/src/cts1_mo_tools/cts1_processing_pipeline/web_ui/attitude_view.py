"""3D view of the satellite's estimated attitude and body rates.

A 3U CubeSat (10 x 10 x 34 cm) drawn in the orbit reference frame, rotated
by the ADCS-estimated roll/pitch/yaw, with the estimated body angular rates
drawn as an arrow (the angular velocity vector) plus per-axis readouts.
`AttitudePlayer` wraps that view with playback controls, stepping through
every extended beacon in a time window.

Frames:
    - Orbit frame: +X = ram (along-track velocity), +Z = nadir, +Y = orbit
      anti-normal (right-handed). The scene is z-up, so orbit coordinates
      are mapped into the scene with `_ORBIT_TO_SCENE` = diag(1, -1, -1):
      ram stays +x, nadir becomes down, and orbit normal (-Y orbit, the
      orbital angular momentum direction r x v) becomes +y.
    - Body frame: at zero attitude, it coincides with the orbit frame. The
      3U long axis is body +X, so the +X 1U face (the one with the slit)
      points into ram. The slit runs along body Y, so it's horizontal
      (perpendicular to nadir) whenever roll = 0.
    - Euler angles are applied as a 3-2-1 (yaw, then pitch, then roll)
      intrinsic sequence: R_orbit_from_body = Rz(yaw) @ Ry(pitch) @ Rx(roll).
      That's the sequence under which "roll = 0" keeps body Y horizontal
      regardless of pitch/yaw.

Scene units are decimetres (1 unit = 10 cm = 1U), which keeps the default
camera near-plane comfortably out of the way.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["AttitudePlayer", "AttitudeView"]

import bisect
import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from nicegui import ui

from . import data as beacon_data

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from nicegui import events
    from nicegui.elements.scene.scene_object3d import Object3D

Matrix = list[list[float]]
Vector = tuple[float, float, float]

_ORBIT_TO_SCENE: Matrix = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]

_HALF_LENGTH = 1.7  # 34 cm / 2, in decimetres
_HALF_WIDTH = 0.5  # 10 cm / 2

_BODY_COLOR = "#9aa4ad"
_FRONT_FACE_COLOR = "#d5dade"
_SLIT_COLOR = "#0b0d10"
_PANEL_COLOR = "#1f3a68"
_AXIS_COLORS = {"x": "#ef4444", "y": "#22c55e", "z": "#3b82f6"}
_RATE_COLOR = "#facc15"
_ORBIT_AXIS_COLOR = "#94a3b8"

_LABEL_STYLE = (
    "color: #f8fafc; font-size: 11px; background: rgba(15, 23, 42, 0.7); "
    "padding: 1px 5px; border-radius: 3px; white-space: nowrap;"
)


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def _rot_x(angle_rad: float) -> Matrix:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(angle_rad: float) -> Matrix:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle_rad: float) -> Matrix:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def scene_from_body(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Matrix:
    """Rotation taking body-frame vectors into scene coordinates (3-2-1
    Euler sequence -- see the module docstring).
    """
    orbit_from_body = _matmul(
        _rot_z(math.radians(yaw_deg)),
        _matmul(_rot_y(math.radians(pitch_deg)), _rot_x(math.radians(roll_deg))),
    )
    return _matmul(_ORBIT_TO_SCENE, orbit_from_body)


def _align_y_to(direction: Vector) -> Matrix:
    """A rotation taking local +Y onto `direction` (three.js cylinders are
    built along +Y, so this is what points an arrow).
    """
    norm = math.sqrt(sum(c * c for c in direction))
    y = [c / norm for c in direction]
    # Any reference not parallel to `y` works for building the basis.
    ref = [1.0, 0.0, 0.0] if abs(y[0]) < 0.9 else [0.0, 0.0, 1.0]  # noqa: PLR2004
    dot = sum(r * c for r, c in zip(ref, y, strict=True))
    x = [r - dot * c for r, c in zip(ref, y, strict=True)]
    x_norm = math.sqrt(sum(c * c for c in x))
    x = [c / x_norm for c in x]
    z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ]
    return [[x[i], y[i], z[i]] for i in range(3)]


def _arrow(
    scene: ui.scene,
    direction: Vector,
    length: float,
    color: str,
    *,
    radius: float = 0.03,
) -> Object3D:
    """An arrow from the current parent's origin along `direction`."""
    head_length = min(0.3, length * 0.3)
    shaft_length = length - head_length
    with scene.group().rotate_R(_align_y_to(direction)) as arrow:
        scene.cylinder(radius, radius, shaft_length, radial_segments=12).move(
            0, shaft_length / 2, 0
        ).material(color)
        scene.cylinder(0, radius * 3, head_length, radial_segments=16).move(
            0, shaft_length + head_length / 2, 0
        ).material(color)
    return arrow


def _rate_arrow_length(rate_norm_deg_per_sec: float) -> float:
    """Log-scaled so both a slow drift (~0.1 deg/s) and a tumble (~tens of
    deg/s) read sensibly without the arrow leaving the view. The floor is
    past the body's furthest corner (~1.8), so the arrow always pokes out
    of the satellite whatever direction it points.
    """
    return min(3.6, 2.2 + 0.9 * math.log10(1.0 + rate_norm_deg_per_sec))


def _fmt(value: float | None, unit: str) -> str:
    return "?" if value is None else f"{value:+.2f}{unit}"


class AttitudeView:
    """A card holding the 3D scene plus a numeric readout.

    Built once per page; `update(row)` re-poses it in place, so a periodic
    refresh doesn't reset whatever camera angle the user has orbited to.
    """

    def __init__(self) -> None:
        with ui.card().classes("w-full") as self.card:
            ui.label("Attitude & Body Rates").classes("text-lg font-bold")
            self._caption = ui.label().classes("text-caption text-grey")
            with ui.row().classes("w-full gap-6 items-start flex-wrap"):
                with ui.scene(
                    width=640,
                    height=440,
                    grid=False,
                    camera=ui.scene.perspective_camera(fov=45),
                    background_color="#0f172a",
                ) as self._scene:
                    self._build_static()
                with ui.column().classes("gap-2"):
                    self._no_attitude = ui.label(
                        "No attitude estimate -- shown at zero attitude."
                    ).classes("text-warning")
                    self._readout: dict[str, ui.label] = {}
                    for name in ("Roll", "Pitch", "Yaw", "ωx", "ωy", "ωz"):
                        with ui.row().classes("gap-3 items-baseline"):
                            ui.label(name).classes("text-caption text-grey w-12")
                            self._readout[name] = ui.label().classes(
                                "text-base font-medium font-mono"
                            )
                    ui.label(
                        "Orbit frame: +X ram, +Z nadir, -Y orbit normal. "
                        "Euler 3-2-1 "
                        "(yaw, pitch, roll). Body rates drawn in body axes; "
                        "yellow arrow = angular velocity vector (log-scaled). "
                        "Drag to orbit the camera."
                    ).classes("text-caption text-grey max-w-xs")
        self._scene.move_camera(3.4, -4.6, 2.6, 0.3, 0, -0.2, duration=0)
        self._dynamic: Object3D | None = None

    def _build_static(self) -> None:
        scene = self._scene

        # Orbit reference axes, fixed in the scene.
        _arrow(scene, (1, 0, 0), 3.2, _ORBIT_AXIS_COLOR, radius=0.015)
        scene.text("Ram", _LABEL_STYLE).move(3.45, 0, 0)
        _arrow(scene, (0, 0, -1), 2.4, _ORBIT_AXIS_COLOR, radius=0.015)
        scene.text("Nadir", _LABEL_STYLE).move(0, 0, -2.65)
        _arrow(scene, (0, 1, 0), 2.4, _ORBIT_AXIS_COLOR, radius=0.015)
        scene.text("Orbit normal", _LABEL_STYLE).move(0, 2.65, 0)

        # The satellite body, in body coordinates.
        with scene.group() as self._body:
            scene.box(2 * _HALF_LENGTH, 2 * _HALF_WIDTH, 2 * _HALF_WIDTH).material(
                _BODY_COLOR
            )
            # One solar panel per 1U segment on each long face.
            for x_center in (-1.0, 0.0, 1.0):
                for sign in (-1, 1):
                    offset = sign * (_HALF_WIDTH + 0.01)
                    scene.box(0.92, 0.02, 0.8).move(x_center, offset, 0).material(
                        _PANEL_COLOR
                    )
                    scene.box(0.92, 0.8, 0.02).move(x_center, 0, offset).material(
                        _PANEL_COLOR
                    )
            # Ram-facing 1U face, with the horizontal (body-Y) slit.
            scene.box(0.02, 0.96, 0.96).move(_HALF_LENGTH + 0.01, 0, 0).material(
                _FRONT_FACE_COLOR
            )
            scene.box(0.03, 0.7, 0.08).move(_HALF_LENGTH + 0.025, 0, 0).material(
                _SLIT_COLOR
            )

            # Body axes.
            for axis, direction, length in (
                ("x", (1, 0, 0), 2.6),
                ("y", (0, 1, 0), 1.4),
                ("z", (0, 0, 1), 1.4),
            ):
                _arrow(scene, direction, length, _AXIS_COLORS[axis], radius=0.02)

    def update(self, row: dict[str, Any] | None, caption: str) -> None:
        """Re-pose the model from an extended-beacon `row` (None or missing
        attitude fields leaves it at zero attitude, with a note).
        """
        self._caption.set_text(caption)

        roll = row.get("adcs_estimated_roll_angle_deg") if row else None
        pitch = row.get("adcs_estimated_pitch_angle_deg") if row else None
        yaw = row.get("adcs_estimated_yaw_angle_deg") if row else None
        rates: Sequence[float | None] = (
            [row.get(f"adcs_estimated_rate_{a}_deg_per_sec") for a in "xyz"]
            if row
            else [None, None, None]
        )
        has_attitude = None not in (roll, pitch, yaw)

        self._body.rotate_R(
            scene_from_body(roll or 0.0, pitch or 0.0, yaw or 0.0)
            if has_attitude
            else scene_from_body(0.0, 0.0, 0.0)
        )

        # Build the new rate arrow/labels before removing the old ones, so
        # playback doesn't flicker between frames.
        previous = self._dynamic
        with self._scene, self._body, self._scene.group() as self._dynamic:
            tips = {"x": (2.8, 0, 0), "y": (0, 1.6, 0), "z": (0, 0, 1.6)}
            for (axis, tip), rate in zip(tips.items(), rates, strict=True):
                self._scene.text(
                    f"{axis.upper()}  ω{axis} {_fmt(rate, '°/s')}", _LABEL_STYLE
                ).move(*tip)
            if None not in rates:
                omega: Vector = (rates[0], rates[1], rates[2])  # type: ignore[assignment]
                norm = math.sqrt(sum(c * c for c in omega))
                if norm > 0.01:  # noqa: PLR2004
                    length = _rate_arrow_length(norm)
                    _arrow(self._scene, omega, length, _RATE_COLOR, radius=0.04)
                    direction = [c / norm * (length + 0.25) for c in omega]
                    self._scene.text(f"ω {norm:.2f}°/s", _LABEL_STYLE).move(*direction)

        if previous is not None:
            previous.delete()

        self._no_attitude.set_visibility(not has_attitude)
        for name, value in (
            ("Roll", _fmt(roll, "°")),
            ("Pitch", _fmt(pitch, "°")),
            ("Yaw", _fmt(yaw, "°")),
            ("ωx", _fmt(rates[0], " °/s")),
            ("ωy", _fmt(rates[1], " °/s")),
            ("ωz", _fmt(rates[2], " °/s")),
        ):
            self._readout[name].set_text(value)


_SPEED_CHOICES: dict[float, str] = {
    1.0: "1 frame/s",
    2.0: "2 frames/s",
    5.0: "5 frames/s",
    10.0: "10 frames/s",
}


def _gap_str(delta: timedelta) -> str:
    total_sec = int(delta.total_seconds())
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m {total_sec % 60}s"
    return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"


class AttitudePlayer:
    """`AttitudeView` plus playback: a scrubber over every extended beacon
    in a selectable window, play/pause, stepping, and speed.

    Frames are the beacons themselves -- no interpolation between them.
    Extended beacons arrive in bursts during passes with hours-long gaps in
    between, and slerping across a gap would invent an attitude history
    that was never measured. The caption shows each frame's gap to the
    previous one instead, so a jump across a gap is obvious.

    `reload()` (called on the page's periodic refresh) re-queries the
    window, keeping the current frame by timestamp -- or, if the scrubber
    is parked on the newest frame, following the newest one as new
    beacons arrive.
    """

    def __init__(
        self,
        path: Path,
        window_choices: dict[str, float],
        default_window: str,
        format_time: Callable[[datetime], str],
    ) -> None:
        self._path = path
        self._window_choices = window_choices
        self._hours = window_choices[default_window]
        self._format_time = format_time
        self._frames: list[dict[str, Any]] = []
        self._times: list[datetime] = []
        self._index = 0
        self._showing_fallback = False

        self._view = AttitudeView()
        with self._view.card:
            with ui.row().classes("w-full items-center gap-1 flex-wrap"):
                ui.select(
                    list(window_choices),
                    value=default_window,
                    label="Playback window",
                    on_change=lambda e: self._set_window(e.value),
                ).classes("w-40 mr-2")
                ui.button(icon="first_page", on_click=lambda: self._seek(0)).props(
                    "flat round"
                ).tooltip("Oldest")
                ui.button(icon="skip_previous", on_click=lambda: self._step(-1)).props(
                    "flat round"
                ).tooltip("Previous beacon")
                self._play_button = (
                    ui.button(icon="play_arrow", on_click=self._toggle_play)
                    .props("round")
                    .tooltip("Play / pause")
                )
                ui.button(icon="skip_next", on_click=lambda: self._step(1)).props(
                    "flat round"
                ).tooltip("Next beacon")
                ui.button(
                    icon="last_page", on_click=lambda: self._seek(len(self._frames) - 1)
                ).props("flat round").tooltip("Latest")
                ui.select(
                    _SPEED_CHOICES,
                    value=2.0,
                    label="Speed",
                    on_change=self._set_speed,
                ).classes("w-32 ml-2")
                self._position = ui.label().classes("text-caption text-grey ml-2")
            self._slider = ui.slider(
                min=0, max=1, value=0, on_change=self._on_slider
            ).classes("w-full px-2")
        self._timer = ui.timer(1.0 / 2.0, self._tick, active=False)
        self.reload()

    # -- data ---------------------------------------------------------------

    def reload(self) -> None:
        """Re-query the window, keeping the current frame (by timestamp)
        unless parked on the newest frame, in which case follow the newest.
        """
        at_latest = not self._frames or self._index >= len(self._frames) - 1
        current_time = None if at_latest else self._times[self._index]

        since = datetime.now(UTC) - timedelta(hours=self._hours)
        self._frames = beacon_data.load_attitude_window(
            self._path, since=since
        ).to_dicts()
        # Nothing in the window: still show the newest extended beacon ever,
        # the same way the summary cards ignore the chart window.
        self._showing_fallback = not self._frames
        if self._showing_fallback:
            self._frames = beacon_data.latest_beacons(
                self._path, n=1, packet_types=("BEACON_EXTENDED",)
            ).to_dicts()
        self._times = [f["received_at"] for f in self._frames]

        last = max(len(self._frames) - 1, 0)
        self._slider.props(f"max={max(last, 1)}")
        self._slider.set_enabled(len(self._frames) > 1)
        if current_time is None:
            index = last
        else:
            index = min(bisect.bisect_left(self._times, current_time), last)
        self._seek(index, force=True)

    def _set_window(self, label: str) -> None:
        self._hours = self._window_choices[label]
        self.reload()

    # -- playback -----------------------------------------------------------

    def _seek(self, index: int, *, force: bool = False) -> None:
        """Move to frame `index`. Every path goes through the slider's value
        so it always reflects the frame shown; `_on_slider` does the render.
        """
        index = max(0, min(index, len(self._frames) - 1))
        if self._slider.value == index:
            if force or index != self._index:
                self._index = index
                self._show()
        else:
            self._slider.set_value(index)

    def _step(self, delta: int) -> None:
        self._pause()
        self._seek(self._index + delta)

    def _on_slider(self, e: events.ValueChangeEventArguments) -> None:
        if e.value is None or not self._frames:
            return
        self._index = max(0, min(int(e.value), len(self._frames) - 1))
        self._show()

    def _toggle_play(self) -> None:
        if self._timer.active:
            self._pause()
            return
        if len(self._frames) <= 1:
            return
        if self._index >= len(self._frames) - 1:
            self._seek(0)  # at the end: play again from the start
        self._timer.activate()
        self._play_button.props("icon=pause")

    def _pause(self) -> None:
        self._timer.deactivate()
        self._play_button.props("icon=play_arrow")

    def _tick(self) -> None:
        if self._index >= len(self._frames) - 1:
            self._pause()
            return
        self._seek(self._index + 1)

    def _set_speed(self, e: events.ValueChangeEventArguments) -> None:
        self._timer.interval = 1.0 / float(e.value)

    # -- rendering ----------------------------------------------------------

    def _show(self) -> None:
        if not self._frames:
            self._view.update(None, "No extended beacon packets decoded yet.")
            self._position.set_text("")
            return

        row = self._frames[self._index]
        received_at = row["received_at"]
        caption = f"From extended beacon at {self._format_time(received_at)}"
        if self._showing_fallback:
            caption += " -- none in the playback window, showing the latest"
        elif self._index > 0:
            gap = received_at - self._times[self._index - 1]
            caption += f" -- {_gap_str(gap)} after the previous frame"
        self._view.update(row, caption)
        self._position.set_text(f"Frame {self._index + 1} of {len(self._frames)}")
