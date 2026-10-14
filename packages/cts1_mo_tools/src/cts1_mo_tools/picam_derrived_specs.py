"""Derived specs for the SkyFox piCAM (60 deg diagonal FOV, 640x480).

Datasheet: https://www.skyfoxlabs.com/pdf/piCAM_Datasheet_rev_A_2025.pdf

Selected stats are displayed on the Wiki: https://github.com/CalgaryToSpace/CTS-SAT-1-Wiki/wiki/Camera
"""

import math
from collections.abc import Callable

# --- Inputs ---------------------------------------------------------------
W_PX, H_PX = 640, 480  # image size [px]
FOV_DIAG_DEG = 60.0  # diagonal field of view [deg]
ALT_KM = 500.0  # orbit altitude [km]
ALT_TOL_KM = 10.0  # altitude uncertainty [km]
AURORA_ALT_KM = 200.0  # altitude of auroral emission layer [km]
R_EARTH_KM = 6371.0  # mean Earth radius [km]

# --- Camera geometry ------------------------------------------------------
diag_px = math.hypot(W_PX, H_PX)  # 800 px
half_diag = math.radians(FOV_DIAG_DEG / 2)
f_px = (diag_px / 2) / math.tan(half_diag)  # focal length [px]
ifov = 1 / f_px  # instantaneous FOV [rad/px]
half_h = math.atan((W_PX / 2) / f_px)  # half horizontal FOV [rad]
half_v = math.atan((H_PX / 2) / f_px)  # half vertical FOV [rad]


def flat_swath(h: float, half_angle: float) -> float:
    """Full swath on a flat Earth [km]."""
    return 2 * h * math.tan(half_angle)


def sphere_swath(h: float, half_angle: float) -> float:
    """Full swath on a spherical Earth [km] (arc length along the surface)."""
    central = (
        math.asin((R_EARTH_KM + h) / R_EARTH_KM * math.sin(half_angle)) - half_angle
    )
    return 2 * R_EARTH_KM * central


def gsd_m(range_km: float) -> float:
    """Sample distance at a given slant range [m/px]."""
    return range_km * ifov * 1000


def nominal_pm(fn: Callable[[float], float]) -> tuple[float, float]:
    """Return (nominal, +/- half-range) of fn over altitude ALT_KM +/- ALT_TOL_KM."""
    lo, mid, hi = fn(ALT_KM - ALT_TOL_KM), fn(ALT_KM), fn(ALT_KM + ALT_TOL_KM)
    return mid, (max(lo, hi) - min(lo, hi)) / 2


def emit(key: str, unit: str, val: float, tol: float | None = None) -> None:
    if tol is None:
        print(f"  {key}: {val:.1f} {unit}")  # noqa: T201
    else:
        print(f"  {key}: {val:.1f} ± {tol:.1f} {unit}")  # noqa: T201


# --- Derived stats --------------------------------------------------------
results = {
    ("GSD, nadir", "m/px"): nominal_pm(gsd_m),
    ("GSD, corner cross-track", "m/px"): nominal_pm(
        lambda h: gsd_m(h / math.cos(half_diag))
    ),
    ("GSD, corner radial", "m/px"): nominal_pm(
        lambda h: gsd_m(h / math.cos(half_diag) ** 2)
    ),
    ("Swath diagonal, flat", "km"): nominal_pm(lambda h: flat_swath(h, half_diag)),
    ("Swath diagonal, sphere", "km"): nominal_pm(lambda h: sphere_swath(h, half_diag)),
    ("Swath horizontal, flat", "km"): nominal_pm(lambda h: flat_swath(h, half_h)),
    ("Swath horizontal, sphere", "km"): nominal_pm(lambda h: sphere_swath(h, half_h)),
    ("Swath vertical, flat", "km"): nominal_pm(lambda h: flat_swath(h, half_v)),
    ("Swath vertical, sphere", "km"): nominal_pm(lambda h: sphere_swath(h, half_v)),
    ("Auroral sample dist.", "m/px"): nominal_pm(lambda h: gsd_m(h - AURORA_ALT_KM)),
}

# Swath for any frame rotation spans short side .. diagonal
sw_min = flat_swath(ALT_KM, half_v)
sw_max = flat_swath(ALT_KM, half_diag)


# --- Output ---------------------------------------------------------------
def main() -> None:
    """Print the input values, then the camera geometry and derived orbit stats."""
    print("=== Input values ===")  # noqa: T201
    emit("Image width", "px", W_PX)
    emit("Image height", "px", H_PX)
    emit("Diagonal FOV", "deg", FOV_DIAG_DEG)
    emit("Altitude", "km", ALT_KM, ALT_TOL_KM)
    emit("Auroral emission altitude", "km", AURORA_ALT_KM)
    emit("Earth radius", "km", R_EARTH_KM)

    print("\n=== Output values ===")  # noqa: T201
    emit("Diagonal", "px", diag_px)
    emit("Focal length", "px", f_px)
    emit("Instantaneous FOV", "mrad/px", ifov * 1e3)
    emit("Instantaneous FOV", "mdeg/px", math.degrees(ifov) * 1e3)
    emit("Horizontal FOV", "deg", math.degrees(2 * half_h))
    emit("Vertical FOV", "deg", math.degrees(2 * half_v))
    for (name, unit), (val, tol) in results.items():
        emit(name, unit, val, tol)
    emit("Swath, any orientation, min (flat)", "km", sw_min)
    emit("Swath, any orientation, max (flat)", "km", sw_max)


if __name__ == "__main__":
    main()
