"""Data layer for the "Browse Packets" page: server-side filtered, paged
access to every column of `everything_decoded.parquet`, plus CSV/Excel
exports of the filtered set.

Every filter (packet type, time range, message substring) is pushed into
the lazy scan before `.collect()`, and a page is a `.slice()` on that same
lazy frame -- so browsing a file with a lot of history only ever
materializes the one page (plus one column-pruned null-count pass) actually
needed for the current filter, never the whole table.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_EXPORT_ROWS",
    "CsvExportResult",
    "ExcelExportResult",
    "PacketBrowserFilters",
    "PacketPage",
    "export_filtered_csv",
    "export_filtered_excel",
    "load_page",
    "packet_type_options",
]

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import xlsxwriter  # pyright: ignore[reportMissingTypeStubs]

from cts1_mo_tools.cts1_processing_pipeline.common import drop_timezones_for_excel

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

DEFAULT_PAGE_SIZE = 200

# A "no filters at all" export could otherwise try to materialize the
# entire mission's history in one HTTP response -- cap it (same for every
# export format) and tell the user to narrow the filters instead of
# silently truncating without a word.
MAX_EXPORT_ROWS = 10_000


@dataclass(frozen=True, slots=True)
class PacketBrowserFilters:
    """Every server-side filter the Browse Packets page applies."""

    packet_types: tuple[str, ...] | None = None  # None/empty = every type
    start: datetime | None = None
    end: datetime | None = None
    message_substring: str | None = None
    case_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class PacketPage:
    rows: pl.DataFrame
    total_rows: int
    columns: list[str]  # schema order, minus columns all-null in the filtered set


@dataclass(frozen=True, slots=True)
class CsvExportResult:
    csv_bytes: bytes
    row_count: int
    truncated: bool  # True if the filtered set exceeded MAX_EXPORT_ROWS


@dataclass(frozen=True, slots=True)
class ExcelExportResult:
    xlsx_bytes: bytes
    row_count: int
    truncated: bool  # True if the filtered set exceeded MAX_EXPORT_ROWS


def _scan(path: Path) -> pl.LazyFrame | None:
    if not path.exists():
        return None
    return pl.scan_parquet(path)


def _apply_filters(lf: pl.LazyFrame, filters: PacketBrowserFilters) -> pl.LazyFrame:
    if filters.packet_types:
        lf = lf.filter(pl.col("packet_type").is_in(filters.packet_types))
    if filters.start is not None:
        lf = lf.filter(pl.col("received_at") >= filters.start)
    if filters.end is not None:
        lf = lf.filter(pl.col("received_at") <= filters.end)
    if filters.message_substring:
        message = pl.col("general_message")
        needle = filters.message_substring
        if not filters.case_sensitive:
            message = message.str.to_lowercase()
            needle = needle.lower()
        lf = lf.filter(message.str.contains(needle, literal=True))
    return lf


def packet_type_options(path: Path) -> list[str]:
    """Every distinct `packet_type` in the file, sorted -- for the filter
    dropdown's choices.
    """
    lf = _scan(path)
    if lf is None:
        return []
    return sorted(
        lf.select(pl.col("packet_type").unique()).collect()["packet_type"].to_list()
    )


def _total_and_non_null_columns(
    filtered: pl.LazyFrame, column_names: list[str]
) -> tuple[int, list[str]]:
    """One pass over the filtered set: its row count, plus which columns
    have at least one non-null value in it -- computed together so a page
    load or export only scans the filtered rows once for this, not twice.
    """
    agg = filtered.select(
        pl.len().alias("__total__"),
        *[pl.col(c).null_count().alias(c) for c in column_names],
    ).collect()
    total = int(agg.item(0, "__total__"))
    if total == 0:
        return 0, []
    row = agg.row(0, named=True)
    non_null_columns = [c for c in column_names if row[c] < total]
    return total, non_null_columns


def load_page(
    path: Path,
    filters: PacketBrowserFilters,
    *,
    offset: int,
    limit: int = DEFAULT_PAGE_SIZE,
) -> PacketPage:
    """One page of rows matching `filters`, newest first, restricted to
    columns that aren't all-null across the *entire* filtered set (not just
    this page).
    """
    lf = _scan(path)
    if lf is None:
        return PacketPage(rows=pl.DataFrame(), total_rows=0, columns=[])

    column_names = lf.collect_schema().names()
    filtered = _apply_filters(lf, filters)
    total, non_null_columns = _total_and_non_null_columns(filtered, column_names)
    if total == 0:
        return PacketPage(rows=pl.DataFrame(), total_rows=0, columns=[])

    rows = (
        filtered.sort("received_at", descending=True)
        .slice(offset, limit)
        .select(non_null_columns)
        .collect()
    )
    return PacketPage(rows=rows, total_rows=total, columns=non_null_columns)


def _filtered_export_df(
    path: Path, filters: PacketBrowserFilters, *, row_cap: int
) -> tuple[pl.DataFrame, bool]:
    """The rows matching `filters` (capped at `row_cap`, newest first),
    restricted to columns non-null somewhere in that set -- the same shape
    the grid itself shows, just without pagination. Shared by every export
    format; returns (df, truncated).
    """
    lf = _scan(path)
    if lf is None:
        return pl.DataFrame(), False

    column_names = lf.collect_schema().names()
    filtered = _apply_filters(lf, filters)
    total, non_null_columns = _total_and_non_null_columns(filtered, column_names)
    if total == 0:
        return pl.DataFrame(), False

    df = (
        filtered.sort("received_at", descending=True)
        .head(row_cap)
        .select(non_null_columns)
        .collect()
    )
    return df, total > row_cap


def export_filtered_csv(path: Path, filters: PacketBrowserFilters) -> CsvExportResult:
    """CSV of every row matching `filters`, capped at `MAX_EXPORT_ROWS`."""
    df, truncated = _filtered_export_df(path, filters, row_cap=MAX_EXPORT_ROWS)
    return CsvExportResult(
        csv_bytes=df.write_csv().encode("utf-8"),
        row_count=df.height,
        truncated=truncated,
    )


def export_filtered_excel(
    path: Path, filters: PacketBrowserFilters
) -> ExcelExportResult:
    """Excel (.xlsx) of every row matching `filters`, capped at
    `MAX_EXPORT_ROWS`.
    """
    df, truncated = _filtered_export_df(path, filters, row_cap=MAX_EXPORT_ROWS)
    df = drop_timezones_for_excel(df)
    buf = io.BytesIO()
    workbook = xlsxwriter.Workbook(buf, {"in_memory": True})
    df.write_excel(workbook=workbook, worksheet="packets")
    workbook.close()
    return ExcelExportResult(
        xlsx_bytes=buf.getvalue(), row_count=df.height, truncated=truncated
    )
