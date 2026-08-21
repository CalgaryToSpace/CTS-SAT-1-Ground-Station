"""Loading `satellite_events_from_beacons.parquet` (step 4's output) for the
web UI's Satellite Events page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars -- see
`pipeline_status.py` for the analogous loader over step 1/2/3's outputs.
"""

from __future__ import annotations

__all__ = ["FILENAME", "load_events"]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_4_detect_satellite_events import (
    pipeline as step_4_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

FILENAME = step_4_pipeline.OUTPUT_FILENAME


def load_events(output_dir: Path) -> pl.DataFrame:
    """Every detected event, newest first -- empty if step 4 hasn't run yet."""
    path = output_dir / FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)
