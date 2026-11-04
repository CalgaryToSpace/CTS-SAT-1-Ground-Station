"""Step 9: clean MPI frames' pixels, as simple_sat_ops' MPI viewer does.

Reads step 6's `mpi_frames.parquet` and writes `mpi_frames_cleaned.parquet`:
the same rows and columns, plus the frame's pixels after each of the
cleaning steps (see `cleaning.py` for what each does), as JSON arrays like
`pixels` and on the same pixel indices:

* `pixels_clean_step_1`: the instrument's own background subtraction
  undone -- raw counts. Null before a recording's first background estimate.
* `pixels_clean_step_2`: ...and detrended across the frame's strip.
* `pixels_clean_step_3`: ...and a local, sliding background subtracted --
  fully cleaned.

Each is null for a frame that has no value at that step. A frame's cleaned
values depend on its neighbours (back to its range's background estimate, up
to 256 frames before it, and the sweeps either side), so they can change as
more of a recording is downlinked: the whole file is recomputed every run,
like step 6's.

Each frame collection (`collection_started_at`, from step 6) is cleaned on
its own, as the frame counter restarts with every recording.
"""

from __future__ import annotations

__all__ = [
    "CLEANED_PIXEL_COLUMNS",
    "DEFAULT_DATA_DIR",
    "OUTPUT_FILENAME",
    "add_cleaned_pixels",
    "pixel_matrix",
    "run",
]

from typing import TYPE_CHECKING

import numpy as np
import orjson
import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)

from . import cleaning

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

DEFAULT_DATA_DIR = step_6_pipeline.DEFAULT_DATA_DIR
OUTPUT_FILENAME = "mpi_frames_cleaned.parquet"

# The pixel columns this step adds, one per cleaning step, in order.
CLEANED_PIXEL_COLUMNS = (
    "pixels_clean_step_1",
    "pixels_clean_step_2",
    "pixels_clean_step_3",
)
# Cleaned values are written to this many decimal places (raw ones are whole
# DN, so this is well under their precision).
_DECIMALS = 1
# The order frames are cleaned in: by recording, then frame counter.
_SORT_COLUMNS = ("collection_started_at", "frame_counter", "collected_at")


def _pixel_lists(frames: pl.DataFrame, column: str) -> pl.Series:
    """`frames[column]` (JSON arrays of pixel values) decoded, a frame with
    none (null) as an empty list -- so each frame's values in the exploded
    series are exactly its list's length.
    """
    return (
        frames[column]
        .str.json_decode(pl.List(pl.Float64))
        .fill_null(pl.Series([[]], dtype=pl.List(pl.Float64)))
    )


def pixel_matrix(
    frames: pl.DataFrame, column: str = "pixels"
) -> tuple[int, npt.NDArray[np.float64]]:
    """`frames[column]` (JSON arrays of pixel values, each starting at its
    frame's `first_pixel_index`) as one row per frame, from the lowest pixel
    index any frame starts at -- NaN where a frame has no value. Vectorised,
    rather than a Python loop over frames, as a long collection has
    thousands.
    """
    pixels = _pixel_lists(frames, column)
    lengths = pixels.list.len().to_numpy().astype(np.int64)
    firsts = frames["first_pixel_index"].to_numpy().astype(np.int64)
    first_pixel = int(firsts.min())
    pixel_count = max(1, int((firsts + lengths).max()) - first_pixel)
    matrix = np.full((frames.height, pixel_count), np.nan)
    values = (pixels.explode(empty_as_null=False).fill_null(np.nan).to_numpy()).astype(
        np.float64
    )
    if len(values):
        rows = np.repeat(np.arange(frames.height), lengths)
        # Each value's position within its own frame's list.
        offsets = np.arange(len(values)) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        matrix[rows, np.repeat(firsts - first_pixel, lengths) + offsets] = values
    return first_pixel, matrix


def _to_json(
    matrix: npt.NDArray[np.float64], frames: pl.DataFrame, first_pixel: int
) -> list[str | None]:
    """Each row of `matrix` (laid out as `pixel_matrix` returns) as a JSON
    array over that frame's own pixels (as in its `pixels`) -- null for a
    frame without a value, and within the array for a pixel without one.
    """
    lengths = _pixel_lists(frames, "pixels").list.len().to_list()
    starts = (frames["first_pixel_index"] - first_pixel).to_list()
    rounded = np.round(matrix, _DECIMALS)
    out: list[str | None] = []
    for row, start, length in zip(rounded, starts, lengths, strict=True):
        values = row[start : start + length]
        if not length or np.isnan(values).all():
            out.append(None)
        elif np.array_equal(values, np.trunc(values)):
            # Whole DN (as step 1's usually are): written as integers, which
            # is about a third smaller.
            out.append(orjson.dumps(values.astype(np.int64).tolist()).decode())
        else:
            # orjson writes NaN as null.
            out.append(orjson.dumps(values.tolist()).decode())
    return out


def add_cleaned_pixels(frames: pl.DataFrame) -> pl.DataFrame:
    """`frames` (`mpi_frames.parquet`'s rows) sorted by recording then frame
    counter, with `CLEANED_PIXEL_COLUMNS` added -- see the module docstring.
    """
    frames = frames.sort(*_SORT_COLUMNS)
    if frames.is_empty():
        return frames.with_columns(
            pl.lit(None, dtype=pl.String).alias(c) for c in CLEANED_PIXEL_COLUMNS
        )
    first_pixel, pixels = pixel_matrix(frames)
    cleaned = cleaning.clean_pixels(
        pixels,
        counter=frames["frame_counter"].to_numpy().astype(np.int64),
        target_voltage=(
            frames["inner_dome_voltage_setting"].fill_null(-1).to_numpy()
        ).astype(np.int64),
        recording=(
            frames["collection_started_at"].rank("dense").to_numpy().astype(np.int64)
        ),
    )
    return frames.with_columns(
        pl.Series(name, _to_json(values, frames, first_pixel), dtype=pl.String)
        for name, values in zip(
            CLEANED_PIXEL_COLUMNS,
            (cleaned.step_1, cleaned.step_2, cleaned.step_3),
            strict=True,
        )
    )


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename, so a crash mid-write
    can't leave a truncated/corrupt parquet file at `path`.
    """
    tmp_path = path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def run(*, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Recompute `mpi_frames_cleaned.parquet` from step 6's output.

    Reads `mpi_frames.parquet` from `data_dir` (where step 6 writes it) and
    writes the result back into the same directory -- parquet-in-parquet-out,
    no database involved.
    """
    frames_path = data_dir / step_6_pipeline.OUTPUT_FILENAME
    if not frames_path.exists():
        msg = f"{frames_path} not found -- run step_6 first."
        raise FileNotFoundError(msg)
    logger.info(f"Reading {frames_path}")
    result = add_cleaned_pixels(pl.read_parquet(frames_path))

    out_path = data_dir / OUTPUT_FILENAME
    _write_parquet_atomic(result, out_path)
    counts = ", ".join(
        f"step {i}: {result[c].is_not_null().sum():,}"
        for i, c in enumerate(CLEANED_PIXEL_COLUMNS, start=1)
    )
    logger.info(
        f"Done. {result.height:,} MPI frame(s) written to {out_path}, with "
        f"cleaned pixels for {counts}."
    )
