"""Step 9's MPI frame cleaning, ported from simple_sat_ops' MPI viewer
(`utils/mpi_viewer.c`, `rebuild_clean` -- its "b" key, all three steps, with
its defaults: a 5-image window and the median estimator). See that file's
header for the measurements behind each step; in brief:

The MPI subtracts a background of its own in the FPGA. Every so often (every
256 frames -- 16 sweeps -- by default) it keeps the frame it takes at the 0 V
setpoint as its no-ion-signal background, and takes those counts off every
frame that follows until the next estimate, landing them on a fixed ~2000 DN
offset (`PEDESTAL`). So what arrives is `raw - background + 2000`, except on
the frame it kept, which arrives as measured -- ~10x its neighbours, the
bright vertical stripes in the strip.

1. Undo that: add the range's background frame back onto every frame after
   it and take the 2000 DN off, leaving raw counts.
2. Detrend every frame across its own strip: a least-squares line through its
   first two and last two pixels comes off it, and 2000 DN goes back on. This
   takes out the block shifts left by a range whose estimate had gone stale.
3. Subtract a local background instead: the first frame of each sweep (image)
   is the 0 V dwell read back, so holds no ion signal. The pixel-by-pixel
   median of those over the `window` images centred on each image comes off
   every frame of it, and 2000 DN goes back on.

The result reads on the scale the frames arrived on (`raw - background +
2000`), but with a background measured a few sweeps away rather than one up
to 255 frames stale.

One difference from the C viewer: it only uses frames whose CRC checks out,
are complete and have no JSON marker spliced in as background samples. Step 6
only keeps such frames, so every frame here qualifies.

Pure numpy: see `pipeline.py` for reading and writing the frames.
"""

from __future__ import annotations

__all__ = ["BACKGROUND_WINDOW_IMAGES", "PEDESTAL", "CleanedPixels", "clean_pixels"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# The fixed offset the instrument's own background subtraction lands on, DN.
PEDESTAL = 2000.0
# Images (sweeps) in step 3's sliding background window (odd).
BACKGROUND_WINDOW_IMAGES = 5
# A background-estimate frame stands at least `_REF_RATIO` times above the
# level of the `_REF_FRAMES` frames after it.
_REF_FRAMES = 32
_REF_RATIO = 2.0
# The pixels (of each frame's strip) step 2's line is fitted through.
_EDGE_PIXELS = (0, 1, -2, -1)
_COUNTER_MOD = 0x10000
# Frames on a 16-step-only setpoint it takes to call a sweep 16 steps.
_MIN_SIXTEEN_ONLY_FRAMES = 4


def _upper_median(values: npt.NDArray[np.float64], axis: int | None = None) -> Any:
    """The median, but the upper of the two middle values for an even count
    rather than their mean -- as the C viewer does, so a window straddling a
    change of level returns one of the levels rather than a blend of them.
    """
    ordered = np.sort(values, axis=axis)
    n = ordered.size if axis is None else ordered.shape[axis]
    return np.take(ordered, n // 2, axis=axis)


def _setpoint_index(
    target_voltage: npt.NDArray[np.int64], n: int
) -> npt.NDArray[np.int64]:
    """Each inner dome target voltage's setpoint index (0..n-1) on the n-step
    grid (setpoint m is `m * 65536/n - 1`, m = n being 0 V), or -1 where it's
    off the grid -- a corrupt value.
    """
    step = 65536 // n
    m = (target_voltage + 1 + step // 2) // step
    nearest = m * step - 1
    valid = (m >= 1) & (m <= n) & (np.abs(target_voltage - nearest) <= step // 8)
    return np.where(valid, m - 1, -1)


def _scan_period(target_voltage: npt.NDArray[np.int64]) -> int:
    """Frames per sweep: 16 if the setpoints only a 16-step sweep uses (the
    odd ones of the 16-step grid) show up, else 8.
    """
    rows = _setpoint_index(target_voltage, 16)
    sixteen_only = int(np.count_nonzero((rows >= 0) & (rows % 2 == 0)))
    return 16 if sixteen_only >= _MIN_SIXTEEN_ONLY_FRAMES else 8


def _sweep_columns(
    counter: npt.NDArray[np.int64], target_voltage: npt.NDArray[np.int64], fpi: int
) -> npt.NDArray[np.int64]:
    """Each frame's column (0..fpi-1) within its sweep, or -1 if it can't be
    told. A frame's pixels were integrated during the previous frame's slot,
    so its column comes from the previous frame's target voltage -- which
    must be the previous frame by counter. Column 0 is the raw 0 V
    background (whose own voltage has already moved on), fpi-1 the
    subtracted 0 V dwell (still at 0 V), and the never-commanded setpoint
    fpi-2 is rejected.
    """
    n = len(counter)
    previous_ok = np.zeros(n, dtype=bool)
    previous_ok[1:] = (counter[:-1] + 1) % _COUNTER_MOD == counter[1:]
    previous_row = np.full(n, -1, dtype=np.int64)
    previous_row[1:] = _setpoint_index(target_voltage[:-1], fpi)
    own_row = _setpoint_index(target_voltage, fpi)
    columns = np.where(
        previous_row == fpi - 1,
        np.where(own_row == fpi - 1, fpi - 1, 0),
        fpi - 2 - previous_row,
    )
    columns[~previous_ok | (previous_row < 0) | (previous_row == fpi - 2)] = -1
    return columns


def _image_ids(
    counter: npt.NDArray[np.int64], columns: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], int]:
    """Each frame's image (sweep) index, -1 where its column is unknown, and
    the image count. A frame's counter less its column names its sweep; a
    counter stepping back (an instrument restart, or a re-sent counter)
    starts a new image too.
    """
    ids = np.full(len(counter), -1, dtype=np.int64)
    n_images, previous_counter, current_first = 0, 0, 0
    for k, (ctr, col) in enumerate(
        zip(counter.tolist(), columns.tolist(), strict=True)
    ):
        if col < 0:
            continue
        first = (ctr - col) % _COUNTER_MOD
        step = ((ctr - previous_counter + 0x8000) % _COUNTER_MOD) - 0x8000
        if n_images == 0 or step <= 0 or first != current_first:
            n_images += 1
            current_first = first
        previous_counter = ctr
        ids[k] = n_images - 1
    return ids, n_images


def _background_frames(
    levels: npt.NDArray[np.float64],
    columns: npt.NDArray[np.int64],
    counter: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """The frames the instrument kept as background estimates: 0 V frames
    (column 0) standing well above the frames after them -- keeping, since
    the estimates come at a fixed cadence, only those on the beat most of
    them share (a bright event or corrupt frame can pass the level test).
    """
    valid = np.flatnonzero(~np.isnan(levels))
    keys: list[int] = []
    for i, k in enumerate(valid.tolist()):
        if columns[k] != 0:
            continue
        ref = levels[valid[i + 1 : i + 1 + _REF_FRAMES]]
        if len(ref) and levels[k] >= _REF_RATIO * float(_upper_median(ref)):
            keys.append(k)
    if len(keys) < 3:  # noqa: PLR2004 -- the C viewer's thresholds.
        return np.array(keys, dtype=np.int64)
    key_counters = counter[keys]
    step = int(_upper_median(np.diff(key_counters) % _COUNTER_MOD))
    if step < 2:  # noqa: PLR2004
        return np.array(keys, dtype=np.int64)
    phases = key_counters % step
    values, counts = np.unique(phases, return_counts=True)
    # The C viewer takes the first key's phase among those tied for most.
    tied = values[counts == counts.max()]
    phase = next(p for p in phases.tolist() if p in set(tied.tolist()))
    return np.array(keys, dtype=np.int64)[phases == phase]


def _undo_instrument_subtraction(
    pixels: npt.NDArray[np.float64],
    counter: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Step 1: add each frame's background estimate (the last one at or
    before it) back on and take the pedestal off, leaving raw counts. The
    estimate frames themselves were sent raw, so are left alone.

    Also returns which frames come before the first estimate: those have
    nothing to add back, so are still on the instrument's own scale.
    """
    n_frames = len(counter)
    levels = np.full(n_frames, np.nan)
    for k in np.flatnonzero(~np.isnan(pixels).all(axis=1)).tolist():
        row = pixels[k]
        levels[k] = float(_upper_median(row[~np.isnan(row)]))
    keys = _background_frames(levels, columns, counter)
    cleaned = pixels.copy()
    if not len(keys):
        return cleaned, np.ones(n_frames, dtype=bool)
    latest = np.searchsorted(keys, np.arange(n_frames), side="right") - 1
    owner = np.where(latest >= 0, keys[np.maximum(latest, 0)], -1)
    corrected = (owner >= 0) & (owner != np.arange(n_frames))
    cleaned[corrected] += pixels[owner[corrected]] - PEDESTAL
    return cleaned, owner < 0


def _detrend(pixels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Step 2: take a least-squares line through each frame's edge pixels
    off it, putting the pedestal back on.
    """
    x = np.array(_EDGE_PIXELS, dtype=np.float64) % pixels.shape[1]
    edges = pixels[:, x.astype(np.intp)]
    dx = x - x.mean()
    mean_edge = edges.mean(axis=1, keepdims=True)
    slope = ((edges - mean_edge) * dx).sum(axis=1, keepdims=True) / (dx**2).sum()
    along = np.arange(pixels.shape[1], dtype=np.float64) - x.mean()
    fit: npt.NDArray[np.float64] = mean_edge + slope * along
    return pixels - fit + PEDESTAL


def _image_backgrounds(
    pixels: npt.NDArray[np.float64],
    first_frames: npt.NDArray[np.int64],
    window: int,
) -> npt.NDArray[np.float64]:
    """Step 3's background per image: the pixel-by-pixel median of the first
    frames (`first_frames`, -1 where an image has none) of the `window`
    images centred on it -- NaN where none of those has one.
    """
    half = (window - 1) // 2
    background = np.full((len(first_frames), pixels.shape[1]), np.nan)
    for i in range(len(first_frames)):
        samples = first_frames[max(0, i - half) : i + half + 1]
        samples = samples[samples >= 0]
        if len(samples):
            background[i] = _upper_median(pixels[samples], axis=0)
    return background


@dataclass(frozen=True, slots=True)
class CleanedPixels:
    """Frames' pixels after each cleaning step (see the module docstring),
    each step building on the one before -- one row per frame, NaN where a
    frame has no value at that step.
    """

    # Instrument's background subtraction undone: raw counts. NaN before a
    # recording's first background estimate, as those frames are still on
    # the instrument's scale with nothing to add back.
    step_1: npt.NDArray[np.float64]
    # ...and detrended across each frame's strip.
    step_2: npt.NDArray[np.float64]
    # ...and a local, sliding background subtracted: fully cleaned.
    step_3: npt.NDArray[np.float64]


def _clean_recording(
    pixels: npt.NDArray[np.float64],
    counter: npt.NDArray[np.int64],
    target_voltage: npt.NDArray[np.int64],
    window: int,
) -> CleanedPixels:
    """`clean_pixels` for one recording (frame collection)."""
    nothing = np.full_like(pixels, np.nan)
    if len(counter) == 0:
        return CleanedPixels(nothing, nothing, nothing)
    columns = _sweep_columns(counter, target_voltage, _scan_period(target_voltage))
    image_ids, n_images = _image_ids(counter, columns)
    if n_images == 0:
        return CleanedPixels(nothing, nothing, nothing)

    step_1, before_first_estimate = _undo_instrument_subtraction(
        pixels, counter, columns
    )
    step_2 = _detrend(step_1)
    ok = ~np.isnan(pixels).all(axis=1) & ~np.isnan(
        step_2[:, np.array(_EDGE_PIXELS)]
    ).any(axis=1)

    # Each image's first frame: the 0 V dwell, so background alone.
    first_frames = np.full(n_images, -1, dtype=np.int64)
    for k in np.flatnonzero(ok & (columns == 0) & (image_ids >= 0)).tolist()[::-1]:
        first_frames[image_ids[k]] = k
    background = _image_backgrounds(step_2, first_frames, window)

    # A frame with no column of its own (its predecessor's missing, or its
    # voltage is off the grid) uses the background of the image before it.
    known = image_ids >= 0
    last_known = np.maximum.accumulate(np.where(known, np.arange(len(counter)), -1))
    use = np.where(last_known >= 0, image_ids[np.maximum(last_known, 0)], 0)
    step_3 = nothing.copy()
    step_3[ok] = step_2[ok] - background[use[ok]] + PEDESTAL

    step_1[before_first_estimate] = np.nan
    step_2[~ok] = np.nan
    return CleanedPixels(step_1, step_2, step_3)


def clean_pixels(
    pixels: npt.NDArray[np.float64],
    counter: npt.NDArray[np.int64],
    target_voltage: npt.NDArray[np.int64],
    recording: npt.NDArray[np.int64],
    *,
    window: int = BACKGROUND_WINDOW_IMAGES,
) -> CleanedPixels:
    """`pixels` (one row per frame, sorted by `recording` then frame
    `counter`) after each cleaning step, as described in the module
    docstring -- NaN for a frame without a value at a step (see
    `CleanedPixels`), e.g. with no background sample in its window, or
    missing an edge pixel.

    `target_voltage` is each frame's inner dome voltage setting (-1 where
    unknown); `recording` labels which frame collection each frame is from,
    as the frame counter restarts with each, so each is cleaned on its own.
    """
    steps = [np.full_like(pixels, np.nan) for _ in range(3)]
    starts = np.flatnonzero(np.diff(recording, prepend=recording[:1] - 1) != 0)
    for start, end in zip(starts, [*starts[1:], len(recording)], strict=True):
        cleaned = _clean_recording(
            pixels[start:end], counter[start:end], target_voltage[start:end], window
        )
        for out, step in zip(
            steps, (cleaned.step_1, cleaned.step_2, cleaned.step_3), strict=True
        ):
            out[start:end] = step
    return CleanedPixels(*steps)
