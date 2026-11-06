from datetime import UTC, datetime

import orjson
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui import mpi_strip
from cts1_mo_tools.cts1_processing_pipeline.web_ui.mpi_strip_view import (
    _format_decimals,  # pyright: ignore[reportPrivateUsage]
    _next_zoom,  # pyright: ignore[reportPrivateUsage]
    _readout_frames_json,  # pyright: ignore[reportPrivateUsage]
    collection_key,
)


def test_collection_key_is_millisecond_iso_utc() -> None:
    started_at = datetime(2026, 7, 21, 17, 23, 0, 208_000, tzinfo=UTC)
    assert collection_key(started_at) == "2026-07-21T17:23:00.208Z"
    # Zero-padded, and sub-millisecond digits dropped.
    started_at = datetime(2026, 7, 21, 17, 23, 0, 5_999, tzinfo=UTC)
    assert collection_key(started_at) == "2026-07-21T17:23:00.005Z"


@pytest.mark.parametrize(
    ("current", "direction", "expected"),
    [
        # A long collection: fit-width (0.5 px/frame) is below 1x.
        ("fit", 1, 1),
        (1, -1, "fit"),
        (2, 1, 4),
        (4, -1, 2),
        (8, 1, None),
        ("fit", -1, None),
    ],
)
def test_next_zoom_long_collection(
    current: int | str, direction: int, expected: int | str | None
) -> None:
    got = _next_zoom(current, direction=direction, client_width=1000, column_count=2000)
    assert got == expected


@pytest.mark.parametrize(
    ("current", "direction", "expected"),
    [
        # A short collection: fit-width (3.33 px/frame) sits between 2x and 4x.
        ("fit", 1, 4),
        ("fit", -1, 2),
        (2, 1, "fit"),
        (4, -1, "fit"),
        (1, -1, None),
    ],
)
def test_next_zoom_short_collection(
    current: int | str, direction: int, expected: int | str | None
) -> None:
    got = _next_zoom(current, direction=direction, client_width=1000, column_count=300)
    assert got == expected


def test_format_decimals_reads_every_band_format() -> None:
    assert _format_decimals(".2f") == 2
    for band in mpi_strip.BANDS:  # The browser's readout needs each of them.
        _format_decimals(band.format)
    with pytest.raises(ValueError, match="fixed-point"):
        _format_decimals(",.0%")


def test_readout_frames_json_encodes_columns_and_fields() -> None:
    columns = [
        mpi_strip.StripColumn("frame", frame_index=0, frame_counter=7),
        mpi_strip.StripColumn("missing", frame_counter=8),
        mpi_strip.StripColumn("separator"),
        mpi_strip.StripColumn("frame", frame_index=1, frame_counter=0),
    ]
    at = datetime(2026, 8, 9, 6, 33, 10, 114_500, tzinfo=UTC)
    frames = [
        {
            "collected_at": at,
            "timing_method": "interpolated",
            "time_sync_source": "E",
            "board_temperature_c": 18.25,
        },
        # No board temperature (NaN or null), and no other band column at all.
        {
            "collected_at": at,
            "timing_method": "extrapolated",
            "time_sync_source": "G",
            "board_temperature_c": float("nan"),
        },
    ]
    got = orjson.loads(_readout_frames_json(columns, frames))
    assert got["columnFrames"] == [0, -1, -2, 1]
    assert got["columnCounters"] == [7, 8, None, 0]
    assert got["times"] == ["2026-08-09 06:33:10.114 UTC"] * 2
    assert got["timing"] == ["interpolated (sync E)", "extrapolated (sync G)"]
    assert [b["label"] for b in got["bands"]] == [b.label for b in mpi_strip.BANDS]
    temperature = got["bands"][0]
    assert temperature["unit"] == "°C"
    assert temperature["decimals"] == 1
    assert temperature["values"] == [18.25, None]
    assert got["bands"][1]["values"] == [None, None]
