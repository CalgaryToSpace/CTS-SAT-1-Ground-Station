"""One MPI collection's frames as a long strip image, for the MPI Data page:
one column per frame (in frame counter order, a blank column for each
frame missing by its counter), the detector's pixels down it coloured by
intensity, and a band under them for each of the frames' housekeeping
fields (board temperature, inner dome voltage, ...), coloured by value.

Pure data layer: no NiceGUI/rendering concerns live here, just polars,
numpy and Pillow -- see `mpi_collections.py` for the collections table's.
"""

from __future__ import annotations

__all__ = [
    "BANDS",
    "BAND_HEIGHT",
    "INTENSITY_GRADIENT_CSS",
    "VALUE_GRADIENT_CSS",
    "Band",
    "BandRange",
    "IntensityMode",
    "IntensityRange",
    "MpiStrip",
    "PreparedFrames",
    "StripColumn",
    "column_count",
    "load_collection_frames",
    "prepare_frames",
    "render_prepared",
    "render_strip",
]

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from PIL import Image

from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import numpy.typing as npt

# What each pixel's colour shows: its raw value, or how far it is above (or
# below) that pixel's median over the collection -- which takes out each
# pixel's own baseline, so a signal stands out from the background.
IntensityMode = Literal["raw", "minus_median"]
# The values mapped onto the colour scale's ends: the 1st to 99.5th
# percentile of the typical frames' pixels, or the full range. A frame isn't
# typical when its mean is more than `_TYPICAL_FRAME_MADS` median absolute
# deviations from the median frame's: the detector warming up at the start
# of each recording (its first ~78 frames read ~4x the background), or a
# strong signal. Those then show at the top of the scale, rather than
# washing every other frame out to the bottom of it.
IntensityRange = Literal["typical", "full"]
_PERCENTILE_RANGE = (1.0, 99.5)
_TYPICAL_FRAME_MADS = 10.0


@dataclass(frozen=True, slots=True)
class Band:
    """A housekeeping field of `mpi_frames.parquet`, shown as a band of colour
    under the detector's pixels.
    """

    column: str
    label: str
    unit: str
    format: str


BANDS: tuple[Band, ...] = (
    Band("board_temperature_c", "Board temperature", "°C", ".1f"),
    Band("inner_dome_voltage_v", "Inner dome voltage", "V", ".2f"),
    Band("inner_dome_voltage_setting", "Inner dome setting", "", ".0f"),
    Band("inner_dome_scan_index", "Inner dome scan index", "", ".0f"),
    Band("integration_period", "Integration period", "", ".0f"),
    Band("times_received", "Times received", "", ".0f"),
)

# Layout, in image pixels (at 1 image pixel per frame across).
_PIXEL_ROW_HEIGHT = 4
_DETECTOR_GAP = 10
BAND_HEIGHT = 16
_BAND_GAP = 4
# Blank columns between two frame collections merged into one collection --
# two recordings in one file, whose counters each start again from 0.
_SEPARATOR_COLUMNS = 8
# Browsers struggle with images much wider than this.
_MAX_WIDTH = 32_767

_MISSING_RGBA = (128, 128, 128, 80)
_NO_VALUE_RGBA = (128, 128, 128, 40)


def _gradient(stops: tuple[str, ...]) -> npt.NDArray[np.uint8]:
    """A 256-entry RGB lookup table, evenly interpolated between `stops`."""
    rgb = np.array(
        [[int(stop[i : i + 2], 16) for i in (1, 3, 5)] for stop in stops], dtype=float
    )
    positions = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    return (
        np.stack(
            [np.interp(t, positions, rgb[:, channel]) for channel in range(3)], axis=1
        )
        .round()
        .astype(np.uint8)
    )


# Approximations of matplotlib's "inferno" (dark to bright, for intensity)
# and "viridis" (for the housekeeping values, so they don't read as more
# detector data).
_INFERNO_STOPS = (
    "#000004",
    "#1f0c48",
    "#550f6d",
    "#88226a",
    "#ba3655",
    "#e35933",
    "#f98e09",
    "#f8c932",
    "#fcffa4",
)
_VIRIDIS_STOPS = (
    "#440154",
    "#472d7b",
    "#3b528b",
    "#2c728e",
    "#21918c",
    "#28ae80",
    "#5ec962",
    "#addc30",
    "#fde725",
)
_INTENSITY_LUT = _gradient(_INFERNO_STOPS)
_VALUE_LUT = _gradient(_VIRIDIS_STOPS)
# The same scales as CSS, for their legends.
INTENSITY_GRADIENT_CSS = f"linear-gradient(to right, {', '.join(_INFERNO_STOPS)})"
VALUE_GRADIENT_CSS = f"linear-gradient(to right, {', '.join(_VIRIDIS_STOPS)})"


def load_collection_frames(
    data_dir: Path, collection: dict[str, Any], collections: pl.DataFrame
) -> pl.DataFrame:
    """Every frame (from `mpi_frames.parquet`) in `collection` -- one row of
    `mpi_collections.parquet`, `collections` being all of them -- oldest
    frame collection first, then by frame counter. Empty if step 6 hasn't
    run yet.
    """
    path = data_dir / step_6_pipeline.OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    lf = pl.scan_parquet(path)
    frame_collections = (
        lf.group_by("collection_started_at")
        .agg(collection_ended_at=pl.col("collected_at").max())
        .collect()
    )
    starts = step_8_pipeline.frame_collections_in(
        collection, collections, frame_collections
    )
    return (
        lf.filter(
            pl.col("collection_started_at").is_in(
                pl.Series(
                    starts, dtype=frame_collections["collection_started_at"].dtype
                ).implode()
            )
        )
        .sort("collection_started_at", "frame_counter", "collected_at")
        .collect()
    )


@dataclass(frozen=True, slots=True)
class StripColumn:
    """One column of the strip: a frame, a frame missing by its counter, or a
    gap between two frame collections.
    """

    kind: Literal["frame", "missing", "separator"]
    # The frame's row in the frames the strip was rendered from.
    frame_index: int | None = None
    # The frame's counter, or the one the missing frame would have had.
    frame_counter: int | None = None


@dataclass(frozen=True, slots=True)
class BandRange:
    band: Band
    top: int  # In image pixels.
    low: float | None  # None when no frame has a value.
    high: float | None

    def range_text(self) -> str:
        if self.low is None or self.high is None:
            return "no values"
        fmt, unit = self.band.format, f" {self.band.unit}" if self.band.unit else ""
        if self.low == self.high:
            return f"{self.low:{fmt}}{unit} throughout"
        return f"{self.low:{fmt}} to {self.high:{fmt}}{unit}"


@dataclass(frozen=True, slots=True)
class MpiStrip:
    png: bytes
    width: int  # In image pixels, `zoom` per column.
    height: int
    zoom: int
    columns: list[StripColumn]
    first_pixel_index: int
    pixel_count: int
    bands: list[BandRange]
    # Each frame's pixel values as coloured (per `IntensityMode`), one row
    # per frame, from `first_pixel_index` -- NaN where a frame has none.
    pixel_values: npt.NDArray[np.float64]
    # The values at the ends of the intensity colour scale.
    intensity_low: float
    intensity_high: float

    @property
    def detector_height(self) -> int:
        return self.pixel_count * _PIXEL_ROW_HEIGHT

    def column_at(self, image_x: float) -> StripColumn | None:
        i = int(image_x // self.zoom)
        return self.columns[i] if 0 <= i < len(self.columns) else None

    def pixel_index_at(self, image_y: float) -> int | None:
        """The detector pixel at `image_y`, if it's in the detector's rows."""
        if not 0 <= image_y < self.detector_height:
            return None
        return self.first_pixel_index + int(image_y // _PIXEL_ROW_HEIGHT)

    def pixel_value(self, frame_index: int, pixel_index: int) -> float | None:
        value = float(
            self.pixel_values[frame_index, pixel_index - self.first_pixel_index]
        )
        return None if np.isnan(value) else value

    def band_at(self, image_y: float) -> BandRange | None:
        """The housekeeping band at `image_y`, if it's in one."""
        return next(
            (b for b in self.bands if b.top <= image_y < b.top + BAND_HEIGHT), None
        )


def _columns(frames: pl.DataFrame) -> list[StripColumn]:
    """`frames` (sorted as `load_collection_frames` returns them) laid out as
    columns: a blank column for every counter skipped within a frame
    collection, and `_SEPARATOR_COLUMNS` between frame collections.
    """
    columns: list[StripColumn] = []
    previous: tuple[datetime, int] | None = None
    for i, (started_at, counter) in enumerate(
        zip(frames["collection_started_at"], frames["frame_counter"], strict=True)
    ):
        if previous is not None and previous[0] != started_at:
            columns += [StripColumn("separator")] * _SEPARATOR_COLUMNS
        elif previous is not None:
            columns += [
                StripColumn("missing", frame_counter=missing)
                for missing in range(previous[1] + 1, counter)
            ]
        columns.append(StripColumn("frame", frame_index=i, frame_counter=counter))
        previous = (started_at, counter)
    return columns


def _scaled(values: npt.NDArray[np.float64], low: float, high: float) -> Any:
    """`values` mapped onto 0..255 (NaN to 0) between `low` and `high` -- the
    middle of the scale when they're equal.
    """
    if high <= low:
        scaled = np.full(values.shape, 128.0)
    else:
        scaled = (values - low) / (high - low) * 255
    return np.nan_to_num(np.clip(scaled, 0, 255)).astype(np.uint8)


def _rgba(
    values: npt.NDArray[np.float64], lut: npt.NDArray[np.uint8], low: float, high: float
) -> npt.NDArray[np.uint8]:
    """`values` coloured through `lut`, transparent grey where they're NaN."""
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = lut[_scaled(values, low, high)]
    rgba[..., 3] = 255
    rgba[np.isnan(values)] = _NO_VALUE_RGBA
    return rgba


def _typical_frames(by_frame: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Which frames (rows of `by_frame`) are typical -- see `IntensityRange`."""
    means = np.nanmean(by_frame, axis=1)
    deviations = np.abs(means - np.median(means))
    return deviations <= _TYPICAL_FRAME_MADS * np.median(deviations)


def column_count(frames: pl.DataFrame) -> int:
    """How many columns (frames, missing frames and separators) `frames`'
    strip has -- its width at zoom 1.
    """
    return len(_columns(frames)) if not frames.is_empty() else 0


@dataclass(frozen=True, slots=True)
class PreparedFrames:
    """A collection's frames (from `load_collection_frames`), decoded and
    laid out for `render_prepared` -- the slow, per-collection part of
    rendering a strip, done once however many times it's then re-rendered
    (e.g. as the view's settings change).
    """

    columns: list[StripColumn]
    first_pixel_index: int
    pixel_count: int
    # Each frame's raw pixel values, one row per frame, from
    # `first_pixel_index` -- NaN where a frame has none.
    raw_pixels: npt.NDArray[np.float64]
    # Each of `BANDS`' values per frame -- NaN where a frame has none.
    band_values: tuple[npt.NDArray[np.float64], ...]


def _pixel_matrix(frames: pl.DataFrame) -> tuple[int, npt.NDArray[np.float64]]:
    """`frames`' pixels as one row per frame, from the lowest pixel index any
    frame starts at -- vectorised, rather than a Python loop over frames, as
    a long collection has thousands.
    """
    # A frame with no pixels (null) as an empty list, so each frame's values
    # in the exploded series are exactly its list's length.
    pixels = (
        frames["pixels"]
        .str.json_decode(pl.List(pl.Int64))
        .fill_null(pl.Series([[]], dtype=pl.List(pl.Int64)))
    )
    lengths = pixels.list.len().to_numpy().astype(np.int64)
    firsts = frames["first_pixel_index"].to_numpy().astype(np.int64)
    first_pixel = int(firsts.min())
    pixel_count = max(1, int((firsts + lengths).max()) - first_pixel)
    matrix = np.full((frames.height, pixel_count), np.nan)
    values = (
        pixels.explode(empty_as_null=False)
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy()
    )
    if len(values):
        rows = np.repeat(np.arange(frames.height), lengths)
        # Each value's position within its own frame's list.
        offsets = np.arange(len(values)) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        matrix[rows, np.repeat(firsts - first_pixel, lengths) + offsets] = values
    return first_pixel, matrix


def prepare_frames(frames: pl.DataFrame) -> PreparedFrames | None:
    """`frames` (from `load_collection_frames`) ready to render -- see
    `PreparedFrames`. None when there are no frames.
    """
    if frames.is_empty():
        return None
    first_pixel, raw_pixels = _pixel_matrix(frames)
    return PreparedFrames(
        columns=_columns(frames),
        first_pixel_index=first_pixel,
        pixel_count=raw_pixels.shape[1],
        raw_pixels=raw_pixels,
        band_values=tuple(
            frames[band.column].cast(pl.Float64).fill_null(np.nan).to_numpy()
            if band.column in frames.columns
            else np.full(frames.height, np.nan)
            for band in BANDS
        ),
    )


def render_strip(
    frames: pl.DataFrame,
    *,
    zoom: int = 1,
    mode: IntensityMode = "raw",
    intensity_range: IntensityRange = "typical",
) -> MpiStrip | None:
    """`frames` (from `load_collection_frames`) as a strip image -- see
    `render_prepared`. None when there are no frames.
    """
    prepared = prepare_frames(frames)
    if prepared is None:
        return None
    return render_prepared(
        prepared, zoom=zoom, mode=mode, intensity_range=intensity_range
    )


def render_prepared(
    prepared: PreparedFrames,
    *,
    zoom: int = 1,
    mode: IntensityMode = "raw",
    intensity_range: IntensityRange = "typical",
) -> MpiStrip:
    """`prepared`'s frames as a strip image, `zoom` image pixels across per
    frame (less, if that would make it too wide to show).
    """
    columns = prepared.columns
    zoom = max(1, min(zoom, _MAX_WIDTH // len(columns)))
    frame_columns = np.array(
        [c.frame_index if c.kind == "frame" else -1 for c in columns]
    )
    is_frame = frame_columns >= 0
    rows = frame_columns[is_frame]

    first_pixel, pixel_count = prepared.first_pixel_index, prepared.pixel_count
    by_frame = prepared.raw_pixels
    if mode == "minus_median":
        by_frame = by_frame - np.nanmedian(by_frame, axis=0)
    if intensity_range == "typical":
        typical = by_frame[_typical_frames(by_frame)]
        low, high = (float(v) for v in np.nanpercentile(typical, _PERCENTILE_RANGE))
    else:
        low, high = float(np.nanmin(by_frame)), float(np.nanmax(by_frame))

    detector_height = pixel_count * _PIXEL_ROW_HEIGHT
    bands_top = detector_height + _DETECTOR_GAP
    height = bands_top + len(BANDS) * (BAND_HEIGHT + _BAND_GAP) - _BAND_GAP
    image = np.zeros((height, len(columns), 4), dtype=np.uint8)  # Transparent.
    image[:, np.array([c.kind == "missing" for c in columns])] = _MISSING_RGBA

    detector = _rgba(by_frame[rows].T, _INTENSITY_LUT, low, high)
    image[:detector_height, is_frame] = np.repeat(detector, _PIXEL_ROW_HEIGHT, axis=0)

    band_ranges: list[BandRange] = []
    for i, (band, values) in enumerate(zip(BANDS, prepared.band_values, strict=True)):
        top = bands_top + i * (BAND_HEIGHT + _BAND_GAP)
        has_values = not np.isnan(values).all()
        band_low = float(np.nanmin(values)) if has_values else None
        band_high = float(np.nanmax(values)) if has_values else None
        colours = _rgba(values[rows], _VALUE_LUT, band_low or 0.0, band_high or 0.0)
        image[top : top + BAND_HEIGHT, is_frame] = colours
        band_ranges.append(BandRange(band, top, band_low, band_high))

    if zoom > 1:
        image = np.repeat(image, zoom, axis=1)
    buf = io.BytesIO()
    Image.fromarray(image, "RGBA").save(buf, format="PNG")
    return MpiStrip(
        png=buf.getvalue(),
        width=image.shape[1],
        height=height,
        zoom=zoom,
        columns=columns,
        first_pixel_index=first_pixel,
        pixel_count=pixel_count,
        bands=band_ranges,
        pixel_values=by_frame,
        intensity_low=low,
        intensity_high=high,
    )
