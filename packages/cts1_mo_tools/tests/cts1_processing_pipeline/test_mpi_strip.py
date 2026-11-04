import io
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections.pipeline import (  # noqa: E501
    frame_collections_in,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui import mpi_strip
from PIL import Image

_T0 = datetime(2026, 10, 20, 19, 21, tzinfo=UTC)
_T1 = _T0 + timedelta(minutes=30)


def _frames(
    counters: list[int], *, started_at: datetime = _T0, pixel: int = 2000
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collection_started_at": [started_at] * len(counters),
            "collected_at": [
                started_at + timedelta(milliseconds=31 * c) for c in counters
            ],
            "frame_counter": counters,
            "first_pixel_index": [48] * len(counters),
            "pixels": [json.dumps([pixel, pixel + c, pixel]) for c in counters],
            "board_temperature_c": [10.0 + c for c in counters],
            "inner_dome_voltage_v": [-12.5] * len(counters),
        }
    )


def test_strip_has_a_column_per_frame_and_missing_frame() -> None:
    frames = pl.concat([_frames([0, 1, 4]), _frames([0, 1], started_at=_T1)])
    strip = mpi_strip.render_strip(frames, zoom=3)
    assert strip is not None
    kinds = [c.kind for c in strip.columns]
    assert kinds == ["frame", "frame", "missing", "missing", "frame"] + [
        "separator"
    ] * 8 + ["frame", "frame"]
    assert [c.frame_counter for c in strip.columns[2:4]] == [2, 3]
    image = Image.open(io.BytesIO(strip.png))
    assert image.size == (strip.width, strip.height) == (len(kinds) * 3, strip.height)

    # The mouse over the 5th column (frame 4), in the detector's 2nd pixel.
    column = strip.column_at(4 * 3 + 1)
    assert column is not None
    assert column.frame_index == 2
    assert strip.pixel_index_at(5) == 49
    assert strip.pixel_value(2, 49) == 2004


def test_strip_bands_span_their_values() -> None:
    strip = mpi_strip.render_strip(_frames([0, 1, 2]))
    assert strip is not None
    by_column = {b.band.column: b for b in strip.bands}
    assert by_column["board_temperature_c"].range_text() == "10.0 to 12.0 °C"
    assert by_column["inner_dome_voltage_v"].range_text() == "-12.50 V throughout"
    # Not in these frames at all.
    assert by_column["integration_period"].range_text() == "no values"
    band = strip.band_at(by_column["inner_dome_voltage_v"].top + 1)
    assert band is not None
    assert band.band.column == "inner_dome_voltage_v"


def test_typical_scale_leaves_out_warm_up_frames() -> None:
    warm_up = _frames(list(range(5)), pixel=7500)
    rest = _frames(list(range(5, 100)), pixel=2000)
    frames = pl.concat([warm_up, rest])
    typical = mpi_strip.render_strip(frames, intensity_range="typical")
    full = mpi_strip.render_strip(frames, intensity_range="full")
    assert typical is not None
    assert full is not None
    assert typical.intensity_high < 7500
    assert full.intensity_high == 7504


def test_minus_median_takes_out_each_pixels_baseline() -> None:
    strip = mpi_strip.render_strip(_frames([0, 1, 2]), mode="minus_median")
    assert strip is not None
    assert strip.pixel_value(0, 48) == 0
    assert strip.pixel_value(0, 49) == -1


def test_no_frames_no_strip() -> None:
    assert mpi_strip.render_strip(pl.DataFrame()) is None


@pytest.mark.parametrize(
    ("collection_index", "expected"),
    [(0, [_T0]), (1, [_T1, _T1 + timedelta(minutes=2)])],
)
def test_frame_collections_in(collection_index: int, expected: list[datetime]) -> None:
    analysis = {
        "analysis_started_at": _T1,
        "analysis_ended_at": _T1 + timedelta(minutes=5),
        "analysis_file": "mpi_data/a.mpi",
        "analysis_received_at": _T1 + timedelta(days=1),
    }
    collections = pl.DataFrame(
        [
            {"collection_started_at": _T0, "collection_time_source": "frames"}
            | dict.fromkeys(analysis),
            {"collection_started_at": _T1, "collection_time_source": "analysis"}
            | analysis,
        ]
    )
    frame_collections = pl.DataFrame(
        {
            "collection_started_at": [_T0, _T1, _T1 + timedelta(minutes=2)],
            "collection_ended_at": [
                _T0 + timedelta(minutes=1),
                _T1 + timedelta(minutes=1),
                _T1 + timedelta(minutes=3),
            ],
        }
    )
    collection = collections.to_dicts()[collection_index]
    assert frame_collections_in(collection, collections, frame_collections) == expected


def test_prepared_pixels_line_up_by_pixel_index() -> None:
    # Frames covering different (and differently long) pixel ranges, one
    # with none at all.
    frames = _frames([0, 1, 2]).with_columns(
        first_pixel_index=pl.Series([50, 49, 48]),
        pixels=pl.Series(["[1, null]", None, "[3, 4, 5, 6]"]),
    )
    prepared = mpi_strip.prepare_frames(frames)
    assert prepared is not None
    assert prepared.first_pixel_index == 48
    assert prepared.pixel_count == 4
    np.testing.assert_array_equal(
        prepared.raw_pixels,
        [[np.nan, np.nan, 1, np.nan], [np.nan] * 4, [3, 4, 5, 6]],
    )
    # Bands with no column at all are all NaN.
    assert np.isnan(prepared.band_values[-1]).all()


def test_tiles_split_the_strip_and_fit_together() -> None:
    frames = _frames(list(range(10)))
    whole = mpi_strip.render_strip(frames, zoom=2)
    assert whole is not None
    prepared = mpi_strip.prepare_frames(frames)
    assert prepared is not None
    tiled = mpi_strip.render_prepared(prepared, zoom=2, tile_width=8)
    assert tiled.tile_widths == (8, 8, 4)
    images = [Image.open(io.BytesIO(png)) for png in tiled.tiles]
    assert [image.size[0] for image in images] == [8, 8, 4]
    joined = np.concatenate([np.asarray(image) for image in images], axis=1)
    np.testing.assert_array_equal(joined, np.asarray(Image.open(io.BytesIO(whole.png))))


def test_max_width_averages_columns_and_maps_back() -> None:
    # 10 frames, then a gap of 2 missing frames, then 3 more: 15 columns.
    prepared = mpi_strip.prepare_frames(_frames([*range(10), 12, 13, 14]))
    assert prepared is not None
    strip = mpi_strip.render_prepared(prepared, zoom=4, max_width=5)
    assert (strip.width, strip.zoom, strip.bin_columns) == (5, 1, 3)
    full = np.asarray(
        Image.open(io.BytesIO(mpi_strip.render_prepared(prepared).png)), dtype=float
    )
    shrunk = np.asarray(Image.open(io.BytesIO(strip.png)), dtype=float)
    # Opaque frames average plainly.
    np.testing.assert_allclose(shrunk[:, 0], full[:, 0:3].mean(axis=1), atol=0.5)
    # Image pixel 3 covers columns 9..11; just past its middle is column 10.
    column = strip.column_at(3.5)
    assert column is not None
    assert (column.kind, column.frame_counter) == ("missing", 10)
