import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_9_clean_mpi_frames import (
    cleaning,
)
from cts1_mo_tools.cts1_processing_pipeline.step_9_clean_mpi_frames import (
    pipeline as step_9_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui import mpi_strip

_FPI = 16
_NPIX = 65
# The instrument keeps a background estimate every 256 frames (16 sweeps).
_KEY_EVERY = 256
_T0 = datetime(2026, 10, 20, 19, 21, tzinfo=UTC)


def _setpoint(row: int) -> int:
    return (row + 1) * (65536 // _FPI) - 1


def _target_voltages(n: int, first_counter: int = 0) -> np.ndarray:
    """Each frame's inner dome target voltage for a recording of back-to-back
    16-step sweeps -- frame k's aux carries the setpoint frame k+1's column
    comes from (see `cleaning._sweep_columns`).
    """

    def row_for_column(col: int) -> int:
        return _FPI - 1 if col in (0, _FPI - 1) else _FPI - 2 - col

    return np.array(
        [_setpoint(row_for_column((first_counter + k + 1) % _FPI)) for k in range(n)]
    )


def _recording(
    n: int, bump_frame: int, *, first_counter: int = 0, bump: float = 500.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frames as the instrument sends them, counters from `first_counter`: a
    drifting, tilted background with a `bump` of ion signal on pixels 30-34
    of `bump_frame`, every frame sent as `raw - background(last estimate) +
    2000` except the estimate frames (counter a multiple of 256), sent raw.
    """
    counter = np.arange(first_counter, first_counter + n)
    t = counter.astype(float)[:, None]
    j = np.arange(_NPIX, dtype=float)[None, :]
    raw = 18_000 + 0.5 * t + 3.0 * j + np.zeros((n, _NPIX))
    raw[bump_frame, 30:35] += bump
    sent = raw.copy()
    for k, ctr in enumerate(counter.tolist()):
        key = ctr - ctr % _KEY_EVERY - first_counter
        if key >= 0 and k != key:
            sent[k] = raw[k] - raw[key] + cleaning.PEDESTAL
        elif key < 0:
            # Before the first estimate in the recording: against one we
            # never received.
            sent[k] = raw[k] - 17_000 + cleaning.PEDESTAL
    return sent, counter, _target_voltages(n, first_counter)


def test_cleaning_flattens_background_and_keeps_signal() -> None:
    n, bump_frame = 4 * _KEY_EVERY, 600
    sent, counter, tv = _recording(n, bump_frame)
    cleaned = cleaning.clean_pixels(sent, counter, tv, np.zeros(n, dtype=int))

    # Step 1 is raw counts: the stripes are gone, but not the drift. Frame 0
    # has no frame before it to place it in its sweep, so isn't taken as the
    # first estimate (as in simple_sat_ops): step 1 starts at the second.
    assert np.isnan(cleaned.step_1[:_KEY_EVERY]).all()
    after = np.arange(_KEY_EVERY + 1, _KEY_EVERY + 5)
    np.testing.assert_allclose(cleaned.step_1[after, 0], 18_000 + 0.5 * after)
    # The instrument's own stripes, and its stale-estimate drift, are gone.
    assert np.nanmax(sent) > 15_000
    without_bump = np.delete(cleaned.step_3, bump_frame, axis=0)
    assert np.isfinite(without_bump[1:]).all()
    assert np.nanmax(np.abs(without_bump - cleaning.PEDESTAL)) < 20
    # The ion signal survives at its own size.
    np.testing.assert_allclose(
        cleaned.step_3[bump_frame, 30:35], cleaning.PEDESTAL + 500, atol=20
    )


def test_step_1_leaves_out_frames_before_the_first_estimate() -> None:
    # Starts mid-range: the first 56 frames' estimate was never received.
    n = 3 * _KEY_EVERY
    sent, counter, tv = _recording(n, bump_frame=400, first_counter=200)
    cleaned = cleaning.clean_pixels(sent, counter, tv, np.zeros(n, dtype=int))
    first_key = _KEY_EVERY - 200
    assert np.isnan(cleaned.step_1[:first_key]).all()
    assert np.isfinite(cleaned.step_1[first_key:]).all()
    # Step 2 puts them on the same footing as everything else.
    assert np.isfinite(cleaned.step_2[1:]).all()


def test_each_recording_is_cleaned_on_its_own() -> None:
    n = 3 * _KEY_EVERY
    a, counter, tv = _recording(n, bump_frame=100)
    b = a + 1_000  # A second recording on a different level entirely.
    cleaned = cleaning.clean_pixels(
        np.concatenate([a, b]),
        np.concatenate([counter, counter]),
        np.concatenate([tv, tv]),
        np.repeat([0, 1], n),
    )
    alone = cleaning.clean_pixels(a, counter, tv, np.zeros(n, dtype=int))
    np.testing.assert_allclose(cleaned.step_3[:n], alone.step_3, equal_nan=True)


def test_unknown_voltages_leave_nothing_to_clean() -> None:
    sent, counter, _ = _recording(64, bump_frame=10)
    cleaned = cleaning.clean_pixels(
        sent, counter, np.full(64, -1), np.zeros(64, dtype=int)
    )
    for step in (cleaned.step_1, cleaned.step_2, cleaned.step_3):
        assert np.isnan(step).all()


def _frames_df(n: int, bump_frame: int, *, first_pixel_index: int = 0) -> pl.DataFrame:
    sent, counter, tv = _recording(n, bump_frame)
    return pl.DataFrame(
        {
            "collection_started_at": [_T0] * n,
            "collected_at": [
                _T0 + timedelta(milliseconds=31 * int(c)) for c in counter
            ],
            "frame_counter": counter,
            "first_pixel_index": [first_pixel_index] * n,
            "pixels": [json.dumps([int(v) for v in row]) for row in sent],
            "inner_dome_voltage_setting": tv,
        }
    )


def test_run_writes_frames_with_each_steps_pixels(tmp_path: Path) -> None:
    n = 3 * _KEY_EVERY
    frames = _frames_df(n, bump_frame=300, first_pixel_index=10)
    # Out of order on disk: step 9 sorts by recording and counter.
    frames.reverse().write_parquet(tmp_path / step_6_pipeline.OUTPUT_FILENAME)

    step_9_pipeline.run(data_dir=tmp_path)

    out = pl.read_parquet(tmp_path / step_9_pipeline.OUTPUT_FILENAME)
    assert out.columns == [*frames.columns, *step_9_pipeline.CLEANED_PIXEL_COLUMNS]
    assert out["frame_counter"].to_list() == list(range(n))
    assert out["pixels"].to_list() == frames["pixels"].to_list()
    step_3 = json.loads(out["pixels_clean_step_3"][300])
    assert len(step_3) == _NPIX  # On the frame's own pixel indices.
    assert abs(step_3[32] - (cleaning.PEDESTAL + 500)) < 20
    # Step 1 is raw counts, from the first estimate frame the counters can
    # place (frame 0 has no frame before it).
    assert out["pixels_clean_step_1"][0] is None
    assert (
        json.loads(out["pixels_clean_step_1"][257])[0] == 18_128
    )  # Whole DN, as the frames are.


def test_run_needs_step_6(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="step_6"):
        step_9_pipeline.run(data_dir=tmp_path)


def test_mpi_vis_shows_each_cleaning_step() -> None:
    n = 3 * _KEY_EVERY
    frames = step_9_pipeline.add_cleaned_pixels(_frames_df(n, bump_frame=300))
    raw = mpi_strip.render_strip(frames, mode="raw")
    assert raw is not None
    step_3 = mpi_strip.render_strip(frames, mode="clean_step_3")
    assert step_3 is not None
    assert step_3.has_pixel_values
    assert step_3.intensity_high - step_3.intensity_low < 100
    value = step_3.pixel_value(300, 32)
    assert value is not None
    assert abs(value - (cleaning.PEDESTAL + 500)) < 20
    step_1 = mpi_strip.render_strip(frames, mode="clean_step_1")
    assert step_1 is not None
    assert step_1.pixel_value(257, 0) == 18_128  # Whole DN, as the frames are.
    # Raw is untouched by any of it.
    assert raw.pixel_value(1, 0) == cleaning.PEDESTAL


def test_mpi_vis_without_step_9_has_no_cleaned_values() -> None:
    strip = mpi_strip.render_strip(_frames_df(64, bump_frame=10), mode="clean_step_2")
    assert strip is not None
    assert not strip.has_pixel_values
