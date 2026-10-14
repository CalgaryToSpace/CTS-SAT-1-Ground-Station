"""Derived specs for the SkyFox piCAM (60 deg diagonal FOV, 640x480).

Datasheet: https://www.skyfoxlabs.com/pdf/piCAM_Datasheet_rev_A_2025.pdf

Selected stats are displayed on the Wiki: https://github.com/CalgaryToSpace/CTS-SAT-1-Wiki/wiki/Camera

Note: the datasheet gives no focal length, sensor part number, pixel pitch or
aperture. Only the focal length *in pixels* follows directly from the datasheet
(FOV + resolution). Pixel pitch and f-number are inferred; see the "Optics &
sensor (inferred)" section for the reasoning and the ranges carried through.
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


def emit(val: str) -> None:
    print(val)  # noqa: T201


def emit_val(key: str, unit: str, val: float, tol: float | None = None) -> None:
    if tol is None:
        emit(f"  {key}: {val:.1f} {unit}")
    else:
        emit(f"  {key}: {val:.1f} ± {tol:.1f} {unit}")


def emit_range(
    key: str, unit: str, nom_lo_hi: tuple[float, float, float], fmt: str = ".2f"
) -> None:
    """Print a nominal value with an asymmetric plausible range."""
    nom, lo, hi = nom_lo_hi
    emit(f"  {key}: {nom:{fmt}} {unit}  (range {lo:{fmt}} - {hi:{fmt}})")


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


# --- Optics & sensor (inferred) ---------------------------------------------
# Derivation chain:
#   f_px (datasheet FOV + resolution)  x  pixel pitch p  ->  focal length f [mm]
#   f [mm] / f-number N                                  ->  entrance pupil D [mm]
# Every value below is (nominal, low, high). Since each output is monotonic in its
# inputs, the range is found by evaluating the extreme corners of the inputs.

# 1) Diagonal FOV tolerance.
# The datasheet says "~60 deg" (pp. 2, 10). M12 lens FOV specs are typically
# quoted to about +/-5 %, so take +/-3 deg. Lens distortion (barrel, at 60 deg
# typically a few %) makes the paraxial focal length differ slightly from the
# value implied by the corner angle; that is absorbed in this tolerance.
FOV_DIAG_TOL_DEG = 3.0


def f_px_for_fov(fov_diag_deg: float) -> float:
    """Focal length [px] implied by a diagonal FOV for this image size."""
    return (diag_px / 2) / math.tan(math.radians(fov_diag_deg / 2))


f_px_lo = f_px_for_fov(FOV_DIAG_DEG + FOV_DIAG_TOL_DEG)  # wider FOV -> shorter f
f_px_hi = f_px_for_fov(FOV_DIAG_DEG - FOV_DIAG_TOL_DEG)

# 2) Effective pixel pitch [um] = sensor active-area width / 640 output px.
# This is the pitch of an *output* pixel projected onto the sensor, which is what
# the focal length and GSD depend on. If the sensor scales or bins down to VGA,
# the native photosite pitch is smaller than this.
#
# The datasheet does not name the sensor. What it does give:
#   - An S-mount (M12) lens holder (p. 10; Fig. 2 shows a 17 mm lens barrel),
#     which implies a small board-camera sensor, roughly 1/6" to 1/3" format.
#   - On-chip JPEG at VGA, typical of small CMOS "SoC" camera sensors.
#   - p. 15: "640x480x4 image cells, i.e. less than 2 megapixels". This is
#     ambiguous. Read literally, it is 1.23 M photosites, i.e. a 1280x960 Bayer
#     array reduced 2x to VGA. It may instead just mean one RGGB quad per output
#     pixel, or loosely refer to a 2 MP-class sensor scaled down.
# Candidate sensor classes. Part numbers are only examples of each pitch class;
# none is confirmed to be inside the piCAM.
PITCH_CANDIDATES_UM = {
    "1/6in native VGA (e.g. OV7670, 3.6 um)": 3.6,
    "1/5in native VGA (e.g. OV7740, 4.2 um)": 4.2,
    "1/4in 2 MP scaled to VGA (e.g. OV2640, 1600 px x 2.2 um / 640)": 5.5,
    "1/4in native VGA (e.g. OV7725, 6.0 um)": 6.0,
    "1/3in 1.2 MP binned 2x2 (e.g. 1280x960 x 3.75 um / 640)": 7.5,
}
# Nominal: 1/4in class. It is the most common format for VGA JPEG camera modules
# with M12 lenses, and it covers both the "native VGA" and the "2 MP scaled"
# readings of p. 15. The plausible range spans all candidate classes.
PITCH_UM = (
    5.5,
    min(PITCH_CANDIDATES_UM.values()),
    max(PITCH_CANDIDATES_UM.values()),
)
# Weak cross-check (does not discriminate): the 21 deg lens option shown in
# Fig. 5 would be ~11.9 mm at 5.5 um (stock 12 mm M12 lens) or ~7.8 mm at 3.6 um
# (stock 8 mm). Both land on stock focal lengths, so the FOV options alone can't
# pin the sensor format. Fig. 2's "2.7" callout is not used: its referent is
# unclear from the drawing.

# 3) f-number.
# Nothing in the datasheet states the aperture, and the lens has no iris.
# Prior: fixed-iris M12 board lenses of 2.5-6 mm focal length are almost all
# F/1.6 - F/2.8, with F/2.0 the most common. The two checks below are only
# soft bounds; they confirm the prior is consistent with the datasheet but do
# not narrow it.
F_NUM = (2.0, 1.6, 2.8)
LAMBDA_UM = 0.55  # mid-visible wavelength (IR-cut filter fitted) [um]
BLUR_PX = 2.0  # acceptable blur diameter at VGA after JPEG [px]
NEAR_SHARP_MM = 1000.0  # Fig. 8: foil at 1 m reproduced acceptably sharp [mm]


def focal_mm(fpx: float, pitch_um: float) -> float:
    """Focal length [mm] from focal length [px] and effective pixel pitch [um]."""
    return fpx * pitch_um / 1000


def map3(
    fn: Callable[[float], float], t: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply fn to each element of a (nominal, lo, hi) tuple, keeping its length."""
    return fn(t[0]), fn(t[1]), fn(t[2])


focal = (
    focal_mm(f_px, PITCH_UM[0]),
    focal_mm(f_px_lo, PITCH_UM[1]),
    focal_mm(f_px_hi, PITCH_UM[2]),
)
sensor_w = map3(lambda p: W_PX * p / 1000, PITCH_UM)  # active width [mm]
sensor_h = map3(lambda p: H_PX * p / 1000, PITCH_UM)  # active height [mm]
sensor_d = map3(lambda p: diag_px * p / 1000, PITCH_UM)  # active diagonal [mm]
pupil = (  # entrance pupil diameter D = f / N [mm]
    focal[0] / F_NUM[0],
    focal[1] / F_NUM[2],
    focal[2] / F_NUM[1],
)


def n_max_diffraction(pitch_um: float) -> float:
    """Largest N whose Airy disk (2.44 * lambda * N) fits within BLUR_PX pixels.

    On-orbit samples (Fig. 7) resolve fine cloud structure, so the lens is not
    badly diffraction-limited at VGA.
    """
    return BLUR_PX * pitch_um / (2.44 * LAMBDA_UM)


def n_min_depth_of_field(fpx: float, pitch_um: float) -> float:
    """Smallest N for which an object at NEAR_SHARP_MM is acceptably sharp.

    Focus is fixed at infinity (p. 10). An object at distance s then blurs to
    b ~= f^2 / (N * s) on the sensor; requiring b <= BLUR_PX * p gives
    N >= f^2 / (BLUR_PX * p * s). Caveat: Fig. 8 may have used a custom focus,
    and "acceptably sharp" is judged by eye on a VGA JPEG.
    """
    f = focal_mm(fpx, pitch_um)
    return f**2 / (BLUR_PX * pitch_um / 1000 * NEAR_SHARP_MM)


n_diff = map3(n_max_diffraction, PITCH_UM)
n_dof = (
    n_min_depth_of_field(f_px, PITCH_UM[0]),
    n_min_depth_of_field(f_px_lo, PITCH_UM[1]),
    n_min_depth_of_field(f_px_hi, PITCH_UM[2]),
)


# --- Output ---------------------------------------------------------------
def main() -> None:
    """Print the input values, then the camera geometry and derived orbit stats."""
    emit("=== Input values ===")
    emit_val("Image width", "px", W_PX)
    emit_val("Image height", "px", H_PX)
    emit_val("Diagonal FOV", "deg", FOV_DIAG_DEG)
    emit_val("Altitude", "km", ALT_KM, ALT_TOL_KM)
    emit_val("Auroral emission altitude", "km", AURORA_ALT_KM)
    emit_val("Earth radius", "km", R_EARTH_KM)

    emit("\n=== Output values ===")
    emit_val("Diagonal", "px", diag_px)
    emit_val("Focal length", "px", f_px)
    emit_val("Instantaneous FOV", "mrad/px", ifov * 1e3)
    emit_val("Instantaneous FOV", "mdeg/px", math.degrees(ifov) * 1e3)
    emit_val("Horizontal FOV", "deg", math.degrees(2 * half_h))
    emit_val("Vertical FOV", "deg", math.degrees(2 * half_v))
    for (name, unit), (val, tol) in results.items():
        emit_val(name, unit, val, tol)
    emit_val("Swath, any orientation, min (flat)", "km", sw_min)
    emit_val("Swath, any orientation, max (flat)", "km", sw_max)

    emit("\n=== Optics & sensor: input assumptions ===")
    emit_val("Diagonal FOV tolerance", "deg", FOV_DIAG_TOL_DEG)
    emit("  Pixel pitch candidates (effective, at 640x480 output):")
    for name, p in PITCH_CANDIDATES_UM.items():
        emit(f"    {p:.1f} um  [{name}]")
    emit_range("Effective pixel pitch", "um", PITCH_UM, ".1f")
    emit_range("f-number (prior)", "", F_NUM, ".1f")
    emit_val("Wavelength (mid-visible)", "nm", LAMBDA_UM * 1000)
    emit_val("Acceptable blur diameter", "px", BLUR_PX)
    emit_val("Near object sharp at (Fig. 8)", "mm", NEAR_SHARP_MM)

    emit("\n=== Optics & sensor: derived output ===")
    emit_range("Focal length", "px", (f_px, f_px_lo, f_px_hi), ".1f")
    emit("  Focal length per pitch candidate (at nominal f_px):")
    for name, p in PITCH_CANDIDATES_UM.items():
        emit(f"    {focal_mm(f_px, p):.2f} mm  [{name}]")
    emit_range("Sensor active width", "mm", sensor_w)
    emit_range("Sensor active height", "mm", sensor_h)
    emit_range("Sensor active diagonal", "mm", sensor_d)
    emit_range("Focal length", "mm", focal)
    emit_range("Entrance pupil diameter", "mm", pupil)
    emit_range("Check: max N before diffraction blur", "", n_diff, ".1f")
    emit_range("Check: min N for 1 m sharp at inf. focus", "", n_dof, ".1f")


if __name__ == "__main__":
    main()
