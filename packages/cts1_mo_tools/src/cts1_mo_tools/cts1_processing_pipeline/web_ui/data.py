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
    "load_bulk_file_downlink_packets",
    "load_packet_window",
    "load_tcmd_response_packets",
]

from typing import TYPE_CHECKING

import polars as pl

from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
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


def _filter_to_ranges(
    lf: pl.LazyFrame, ranges: Sequence[tuple[datetime, datetime]]
) -> pl.LazyFrame:
    """Restrict `lf` to rows whose `received_at` falls in any of `ranges`
    (each an inclusive [start, end] window). An empty `ranges` means no time
    filter at all -- every row.
    """
    if not ranges:
        return lf
    in_any_range = pl.any_horizontal(
        [
            pl.col("received_at").is_between(start, end, closed="both")
            for start, end in ranges
        ]
    )
    return lf.filter(in_any_range)


def load_bulk_file_downlink_packets(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    ranges: Sequence[tuple[datetime, datetime]] = (),
) -> pl.DataFrame:
    """`BULK_FILE_DOWNLINK` packets restricted to the union of `ranges`,
    ordered by `bulk_file_offset` (the order file-reassembly cares about,
    not receipt order).

    `ranges=()` (the default) means no time filter -- every such packet ever
    decoded. The web UI's File Reassembler page is what narrows this down to
    one or more `received_at` windows isolating a single download.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type") == "BULK_FILE_DOWNLINK")
    lf = _filter_to_ranges(lf, ranges)
    return lf.sort("bulk_file_offset").collect()


def load_tcmd_response_packets(
    path: Path = DEFAULT_PARQUET_PATH,
    *,
    ranges: Sequence[tuple[datetime, datetime]] = (),
) -> pl.DataFrame:
    """`TCMD_RESPONSE` packets restricted to the union of `ranges`, oldest
    first -- see `load_bulk_file_downlink_packets` for the `ranges` contract.
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame()
    lf = lf.filter(pl.col("packet_type") == "TCMD_RESPONSE")
    lf = _filter_to_ranges(lf, ranges)
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
