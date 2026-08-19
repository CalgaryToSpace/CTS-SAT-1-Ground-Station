"""Loading + shaping `everything_decoded.parquet` for the web UI.

Pure data layer: no NiceGUI/rendering concerns live here, just polars. Every
loader here scans the parquet file lazily and pushes its `packet_type`/
`received_at` filters down into that scan, so a query only materializes the
(small) slice of rows it actually needs instead of the whole file -- the
default 24h chart window is typically a few hundred rows out of a file that
only grows over the mission's lifetime.
"""

from __future__ import annotations

__all__ = [
    "BEACON_PACKET_TYPES",
    "DEFAULT_PARQUET_PATH",
    "latest_beacons",
    "load_beacon_window",
    "load_packet_window",
]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

DEFAULT_PARQUET_PATH = (
    step_3_pipeline.DEFAULT_OUTPUT_DIR / step_3_pipeline.OUTPUT_FILENAME
)

BEACON_PACKET_TYPES = ("BEACON_BASIC", "BEACON_EXTENDED")


def _scan(path: Path) -> pl.LazyFrame | None:
    """Lazily open `path`, or None if the pipeline hasn't produced it yet."""
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def load_packet_window(
    path: Path = DEFAULT_PARQUET_PATH, *, since: datetime | None = None
) -> pl.DataFrame:
    """Every decoded packet (any type) received at/after `since`, oldest first.

    `since=None` means no time filter (the whole file). The `received_at`
    comparison is applied on the lazy frame so polars can push it down into
    the parquet scan rather than filtering after loading every row.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    if since is not None:
        lf = lf.filter(pl.col("received_at") >= since)
    return lf.sort("received_at").collect()


def load_beacon_window(
    path: Path = DEFAULT_PARQUET_PATH, *, since: datetime | None = None
) -> pl.DataFrame:
    """Beacon packets (BASIC or EXTENDED) received at/after `since`, oldest first.

    Both the packet-type and time filters are applied on the lazy frame
    before `.collect()`, so only beacon rows in the requested window are ever
    materialized.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type").is_in(BEACON_PACKET_TYPES))
    if since is not None:
        lf = lf.filter(pl.col("received_at") >= since)
    return lf.sort("received_at").collect()


def latest_beacons(path: Path = DEFAULT_PARQUET_PATH, n: int = 10) -> pl.DataFrame:
    """The `n` most recently received beacon packets, newest first.

    Ignores any time window -- this is "whatever the most recent beacon(s)
    are", even if that's older than the chart window -- so the UI can still
    show *something* when nothing has come down in the last 24h.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    return (
        lf.filter(pl.col("packet_type").is_in(BEACON_PACKET_TYPES))
        .sort("received_at", descending=True)
        .head(n)
        .collect()
    )
