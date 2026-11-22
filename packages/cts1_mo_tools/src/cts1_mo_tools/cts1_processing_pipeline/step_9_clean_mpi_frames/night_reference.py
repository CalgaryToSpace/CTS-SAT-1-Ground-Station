"""Step 9's other MPI cleaning: against a background measured in the dark.
Ported from simple_sat_ops' MPI viewer (`utils/mpi_viewer.c` as of 9629e12,
its "k" key). See "the night reference background" in that file for the
measurements behind it; in brief:

The inner dome doesn't sweep in flight, so a steady ion signal is in the
instrument's background estimates and in `cleaning`'s local background alike,
and both take it away. This method replaces `cleaning`'s steps 2 and 3:

1. Undo the instrument's subtraction. A range whose estimate never came down
   (ahead of the first one, or lost further on) gets it back by matching the
   frames either side of where it fell.
2. Take off the dark background for the frame's integration period: a pattern
   across the strip that's a straight line in temperature, pixel by pixel,
   plus a drift of the whole strip with the frame counter.
3. Match each range's level to the one before it (the level jumps at every
   estimate), and put the recording's median edge level on the pedestal.

The backgrounds are fitted on every run, from whichever frames of
`DARK_WINDOWS` are in the input: with none, nothing is cleaned.

Differences from the C viewer: a recording here is one of step 6's frame
collections, in counter order, so a counter never goes backwards within one;
and a frame the instrument kept is found the new way here only -- `cleaning`
still passes over one whose column can't be told, as it always has.

Pure numpy: see `pipeline.py` for reading and writing the frames.
"""

# The sweep geometry is shared with `cleaning`, which is otherwise left alone.
# pyright: reportPrivateUsage=false

from __future__ import annotations

__all__ = [
    "DARK_WINDOWS",
    "FIRST_PIXEL",
    "PIXEL_COUNT",
    "ReferenceFrames",
    "clean_with_night_reference",
]

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from .cleaning import PEDESTAL, _scan_period, _sweep_columns, _upper_median

if TYPE_CHECKING:
    import numpy.typing as npt

# The only pixel range there are backgrounds for: 48..112.
FIRST_PIXEL = 48
PIXEL_COUNT = 65

# When the reference recordings were dark, as (integration period, from, to).
# At 1000 that's in the Earth's shadow with the MPI face at least 80 degrees
# off ram; at 260, in shadow at whatever attitude. Computed with the C viewer's
# own SGP4 and ADCS log code, sampled every 50 ms. Most run past both ends of
# their recording: only 10-22's and 11-05's starts fall inside one.
# shortcut: fixed windows, recompute them if upstream's `g_ref_recordings` or
# its darkness rule changes, or more ADCS logs of those recordings come down.
DARK_WINDOWS = tuple(
    (period, datetime.fromisoformat(start), datetime.fromisoformat(end))
    for period, start, end in (
        (1000, "2026-11-05T19:57:34.081Z", "2026-11-05T20:05:39Z"),
        (1000, "2026-11-06T17:56:31Z", "2026-11-06T18:09:21Z"),
        (260, "2026-08-09T06:28:10Z", "2026-08-09T06:43:10Z"),
        (260, "2026-09-17T07:11:45Z", "2026-09-17T07:19:30Z"),
        (260, "2026-10-05T10:21:26Z", "2026-10-05T10:30:37Z"),
        (260, "2026-10-05T11:56:02Z", "2026-10-05T12:04:08Z"),
        (260, "2026-10-05T13:30:37Z", "2026-10-05T13:39:04Z"),
        (260, "2026-10-16T19:05:25Z", "2026-10-16T19:14:10Z"),
        (260, "2026-10-18T10:30:02Z", "2026-10-18T10:39:26Z"),
        (260, "2026-10-20T19:16:00Z", "2026-10-20T19:26:57Z"),
        (260, "2026-10-22T16:57:54.121Z", "2026-10-22T17:07:10Z"),
        (260, "2026-10-29T17:54:00Z", "2026-10-29T18:04:41Z"),
    )
)

# Frames either side of an estimate that levels are matched over.
_MATCH_FRAMES = 10
# Frames between estimates, where a recording has too few to measure it.
_DEFAULT_KEY_STEP = 256
# Pixels at each end of the strip a recording's level is set on.
_EDGE_PIXEL_COUNT = 10
# A background estimate stands this many times above the frames after it.
_REF_RATIO = 2.0
_REF_FRAMES = 32


@dataclass(frozen=True, slots=True)
class ReferenceFrames:
    """Frames covering exactly the reference's pixels, one row each, sorted
    by `recording` then `counter`.
    """

    pixels: npt.NDArray[np.float64]
    counter: npt.NDArray[np.int64]
    target_voltage: npt.NDArray[np.int64]
    recording: npt.NDArray[np.int64]
    collected_at_us: npt.NDArray[np.int64]  # Unix time.
    temperature: npt.NDArray[np.float64]  # Signed raw counts.
    integration_period: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _Background:
    """The dark background at one integration period."""

    pattern: npt.NDArray[np.float64]  # Less its own mean, at `temperature`.
    per_count: npt.NDArray[np.float64]  # Its change per count of temperature.
    temperature: float
    level_per_frame: float = 0.0


@dataclass(frozen=True, slots=True)
class _Recording:
    rows: slice  # Where it is in the `ReferenceFrames`.
    pixels: npt.NDArray[np.float64]
    counter: npt.NDArray[np.int64]
    temperature: npt.NDArray[np.float64]
    period: npt.NDArray[np.int64]
    keys: npt.NDArray[np.int64]  # The instrument's background estimates.
    key_step: int  # Frames between them, 0 if there are too few to tell.
    ranges: npt.NDArray[np.int64]  # Estimates at or before each frame.


def _find_estimates(
    pixels: npt.NDArray[np.float64],
    counter: npt.NDArray[np.int64],
    columns: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], int]:
    """The frames kept as background estimates, and the step between them.
    Unlike `cleaning`'s, a frame whose column can't be told is a candidate,
    or a whole range is lost with the frame before its estimate.
    """
    levels = _upper_median(pixels, axis=1)
    found = [
        k
        for k in np.flatnonzero(columns <= 0).tolist()
        if k + 1 < len(levels)
        and levels[k] >= _REF_RATIO * _upper_median(levels[k + 1 : k + 1 + _REF_FRAMES])
    ]
    keys = np.array(found, dtype=np.int64)
    if len(keys) < 2:  # noqa: PLR2004 -- the C viewer's thresholds.
        return keys, 0
    # Measured over every candidate, before the off-beat ones are dropped.
    step = int(_upper_median(np.diff(counter[keys])))
    if step < 2:  # noqa: PLR2004
        return keys, 0
    if len(keys) < 3:  # noqa: PLR2004
        return keys, step
    phases = (counter[keys] % step).tolist()
    # `max` takes the first of those tied, as the C viewer does.
    phase = max(phases, key=phases.count)
    return keys[np.array(phases) == phase], step


def _recordings(frames: ReferenceFrames) -> list[_Recording]:
    labels = frames.recording
    starts = np.flatnonzero(np.diff(labels, prepend=labels[:1] - 1)).tolist()
    recordings: list[_Recording] = []
    for start, end in zip(starts, [*starts[1:], len(labels)], strict=True):
        rows = slice(start, end)
        pixels, counter = frames.pixels[rows], frames.counter[rows]
        voltage = frames.target_voltage[rows]
        columns = _sweep_columns(counter, voltage, _scan_period(voltage))
        if (columns < 0).all():
            continue  # No sweeps at all, as in `cleaning`.
        keys, step = _find_estimates(pixels, counter, columns)
        if not len(keys):
            continue  # Nothing to undo the instrument's subtraction with.
        recordings.append(
            _Recording(
                rows=rows,
                pixels=pixels,
                counter=counter,
                temperature=frames.temperature[rows],
                period=frames.integration_period[rows],
                keys=keys,
                key_step=step,
                ranges=np.searchsorted(keys, np.arange(len(counter)), side="right"),
            )
        )
    return recordings


def _raw_counts(
    rec: _Recording, *, own_estimate_only: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """The instrument's subtraction undone, and which frames that worked for.

    A range whose own estimate is missing is put back by matching across
    where it fell -- or, with `own_estimate_only` (for measuring the
    backgrounds themselves), left out.
    """
    sent, counter, keys = rec.pixels, rec.counter, rec.keys
    counts = sent.copy()
    ok = np.ones(len(counts), dtype=bool)
    owned = rec.ranges > 0
    owned[keys] = False  # The estimates themselves were sent raw.
    counts[owned] += sent[keys[rec.ranges[owned] - 1]] - PEDESTAL

    # Ahead of the first estimate: one range back can be matched across it.
    first = int(keys[0])
    span = rec.key_step or _DEFAULT_KEY_STEP
    back = counter[first] - counter[:first]
    reachable = (back > 0) & (back <= span)
    on = counter[first + 1 :] - counter[first]
    before = np.flatnonzero(reachable)[-_MATCH_FRAMES:]
    after = first + 1 + np.flatnonzero((on > 0) & (on < span))[:_MATCH_FRAMES]
    if own_estimate_only or not len(before) or not len(after):
        ok[:first] = False
    else:
        change = sent[after].mean(axis=0) - sent[before].mean(axis=0)
        counts[:first] += sent[first] + change - PEDESTAL
        ok[:first] = reachable

    if not rec.key_step:
        return counts, ok

    # Estimates lost further on: where each fell is known from the counter,
    # and each is the one before it plus the change across that spot.
    for key, end in zip(keys.tolist(), [*keys[1:].tolist(), len(counts)], strict=True):
        members: npt.NDArray[np.int64] = np.arange(key + 1, end, dtype=np.int64)
        since = counter[members] - counter[key]
        lost, offset = since // rec.key_step, since % rec.key_step
        correction = np.zeros(sent.shape[1])
        broken = own_estimate_only
        for m in range(1, int(lost.max(initial=0)) + 1):
            here = members[lost == m]
            if not broken:
                before = members[(lost == m - 1) & (offset != 0)][-_MATCH_FRAMES:]
                after = members[(lost == m) & (offset != 0)][:_MATCH_FRAMES]
                broken = not len(before) or not len(after)
                if not broken:
                    correction += sent[before].mean(axis=0) - sent[after].mean(axis=0)
            if broken:
                ok[here] = False
            else:
                counts[here] += correction
                # A frame on the very spot is the lost estimate itself.
                ok[members[(lost == m) & (offset == 0)]] = False
    return counts, ok


def _subtract_background(
    rec: _Recording,
    counts: npt.NDArray[np.float64],
    ok: npt.NDArray[np.bool_],
    backgrounds: dict[int, _Background],
) -> npt.NDArray[np.float64]:
    """`counts` less their backgrounds and levelled -- NaN for a frame
    that isn't `ok` or has no background.
    """
    out = np.full_like(counts, np.nan)
    for period, bg in backgrounds.items():
        use = ok & (rec.period == period)
        warmer = rec.temperature[use, None] - bg.temperature
        drift = bg.level_per_frame * rec.counter[use, None]
        out[use] = counts[use] - bg.pattern - bg.per_count * warmer - drift
    means = out.mean(axis=1)
    usable = ~np.isnan(means)
    if not usable.any():
        return out

    # Each range's level matched to the one before it, across its estimate.
    shifts = np.zeros(len(rec.keys) + 1)
    for r, key in enumerate(rec.keys.tolist(), start=1):
        in_before = usable[:key] & (rec.ranges[:key] == r - 1)
        in_after = usable[key + 1 :] & (rec.ranges[key + 1 :] == r)
        before = means[:key][in_before][-_MATCH_FRAMES:]
        after = means[key + 1 :][in_after][:_MATCH_FRAMES]
        jump = after.mean() - before.mean() if len(before) and len(after) else 0.0
        shifts[r] = shifts[r - 1] + jump
    out -= shifts[rec.ranges, None]

    edges = np.r_[:_EDGE_PIXEL_COUNT, -_EDGE_PIXEL_COUNT:0]
    anchor = _upper_median(out[usable][:, edges].mean(axis=1))
    return out - anchor + PEDESTAL


def _dark_frames(frames: ReferenceFrames) -> npt.NDArray[np.bool_]:
    dark = np.zeros(len(frames.counter), dtype=bool)
    for period, start, end in DARK_WINDOWS:
        dark |= (
            (frames.integration_period == period)
            & (frames.collected_at_us >= start.timestamp() * 1e6)
            & (frames.collected_at_us < end.timestamp() * 1e6)
        )
    return dark


def _fit_backgrounds(
    frames: ReferenceFrames, recordings: list[_Recording]
) -> dict[int, _Background]:
    """A background for each integration period with dark frames."""
    in_window = _dark_frames(frames)
    measured: list[
        tuple[_Recording, npt.NDArray[np.float64], npt.NDArray[np.bool_]]
    ] = []
    dark = np.zeros(len(in_window), dtype=bool)
    patterns = np.zeros_like(frames.pixels)
    for rec in recordings:
        if not in_window[rec.rows].any():
            continue
        # Only on frames undone with their own range's estimate.
        counts, ok = _raw_counts(rec, own_estimate_only=True)
        measured.append((rec, counts, ok))
        sample = in_window[rec.rows] & ok
        sample[rec.keys] = False
        dark[rec.rows] = sample
        patterns[rec.rows] = counts - counts.mean(axis=1, keepdims=True)

    fitted: dict[int, _Background] = {}
    for period in np.unique(frames.integration_period[dark]).tolist():
        of_period = dark & (frames.integration_period == period)
        if of_period.sum() < 2:  # noqa: PLR2004 -- a line needs two.
            continue
        temperature, pattern = frames.temperature[of_period], patterns[of_period]
        centred: npt.NDArray[np.float64] = temperature - temperature.mean()
        spread = float(centred @ centred)
        per_count = np.zeros(PIXEL_COUNT, dtype=np.float64)
        if spread:
            per_count[:] = centred @ pattern / spread
        fitted[period] = _Background(
            pattern=pattern.mean(axis=0),
            per_count=per_count,
            temperature=float(temperature.mean()),
        )

    # The drift: the slope of the dark frames' level against the counter with
    # the patterns off, pooled, each recording about its own mean.
    xy, xx = dict.fromkeys(fitted, 0.0), dict.fromkeys(fitted, 0.0)
    for rec, counts, ok in measured:
        levels = _subtract_background(rec, counts, ok, fitted).mean(axis=1)
        for period in fitted:
            use = dark[rec.rows] & (rec.period == period)
            if use.any():
                x = rec.counter[use] - rec.counter[use].mean()
                xy[period] += float(x @ (levels[use] - levels[use].mean()))
                xx[period] += float(x @ x)
    return {
        period: _Background(
            bg.pattern,
            bg.per_count,
            bg.temperature,
            xy[period] / xx[period] if xx[period] else 0.0,
        )
        for period, bg in fitted.items()
    }


def clean_with_night_reference(frames: ReferenceFrames) -> npt.NDArray[np.float64]:
    """`frames`' pixels cleaned as in the module docstring -- NaN for a frame
    with no background for its integration period, in a recording with no
    estimate, or in a range that can't be put back.
    """
    cleaned = np.full_like(frames.pixels, np.nan)
    recordings = _recordings(frames)
    backgrounds = _fit_backgrounds(frames, recordings)
    for rec in recordings:
        counts, ok = _raw_counts(rec, own_estimate_only=False)
        cleaned[rec.rows] = _subtract_background(rec, counts, ok, backgrounds)
    return cleaned
