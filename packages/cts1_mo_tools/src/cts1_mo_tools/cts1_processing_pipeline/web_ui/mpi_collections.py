"""Loading `mpi_collections.parquet` (step 6's per-collection summary) for
the web UI's MPI Data page.

Pure data layer: no NiceGUI/rendering concerns live here, just polars -- see
`satellite_events.py` for the analogous loader over step 4's output.
"""

from __future__ import annotations

__all__ = ["load_collections"]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames import (
    pipeline as step_6_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_collections(data_dir: Path) -> pl.DataFrame:
    """Every MPI collection, newest first -- empty if step 6 hasn't run yet."""
    path = data_dir / step_6_pipeline.COLLECTIONS_OUTPUT_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path).sort("collection_started_at", descending=True)
