"""Data layer for the "Browse Packets" page: server-side filtered, paged
access to every column of `everything_decoded.parquet`, plus CSV/Excel
exports of the filtered set.

Every filter (packet type, time range, message substring) is pushed into
the lazy scan before `.collect()`, and a page is a `.slice()` on that same
lazy frame -- so browsing a file with a lot of history only ever
materializes the one page (plus one column-pruned null-count pass) actually
needed for the current filter, never the whole table.

With `PacketBrowserFilters.reassemble_tcmd_responses` set, every raw
`TCMD_RESPONSE` packet is swapped out for step 5's reassembled responses
(`reassembled_tcmd_responses.parquet`, one row per response, however many
packets it spanned) -- see `_reassembled_tcmd_rows` for how those rows are
fit into `everything_decoded`'s columns, so every filter/sort/export below
works on either view unchanged.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_EXPORT_ROWS",
    "CsvExportResult",
    "ExcelExportResult",
    "PacketBrowserFilters",
    "PacketPage",
    "PacketSort",
    "export_filtered_csv",
    "export_filtered_excel",
    "load_page",
    "packet_type_options",
    "reassembled_tcmd_responses_available",
]

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import xlsxwriter  # pyright: ignore[reportMissingTypeStubs]

from cts1_mo_tools.cts1_processing_pipeline.common import drop_timezones_for_excel
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)

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
    # Show one row per (possibly multi-packet) telecommand response instead
    # of one row per raw TCMD_RESPONSE packet -- see the module docstring.
    reassemble_tcmd_responses: bool = False


@dataclass(frozen=True, slots=True)
class PacketSort:
    """Which column the whole filtered set is ordered by (server-side, so a
    sort spans every page rather than just the one currently loaded).
    """

    column: str = "received_at"
    descending: bool = True


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


def _reassembled_tcmd_path(decoded_path: Path) -> Path:
    """Step 5's output, which lives next to step 3's `everything_decoded`."""
    return decoded_path.parent / step_5_pipeline.OUTPUT_FILENAME


def reassembled_tcmd_responses_available(decoded_path: Path) -> bool:
    """Whether step 5 has produced anything to show in the reassembled view."""
    return _reassembled_tcmd_path(decoded_path).exists()


# Step 5's per-response columns that have no per-packet counterpart in
# `everything_decoded` -- slotted in among the existing tcmd_* header
# columns so they sit together in the grid.
_REASSEMBLY_ONLY_COLUMNS = (
    "tcmd_part_count",
    "tcmd_received_part_count",
    "tcmd_missing_seq_nums",
    "tcmd_is_complete",
    "tcmd_last_received_at",
    "packet_ids",
)


def _reassembled_tcmd_rows(path: Path) -> pl.LazyFrame | None:
    """Step 5's reassembled responses, reshaped to line up with
    `everything_decoded`'s columns: `received_at` is when the response's
    first packet arrived, and `general_message`/`tcmd_response_text` hold
    the whole joined-together text. Every per-packet column (CRC, RSSI,
    decoders, ...) is left null, since a row may now span several packets --
    `packet_ids` lists them instead.
    """
    lf = _scan(path)
    if lf is None:
        return None
    return lf.select(
        received_at=pl.col("first_received_at"),
        packet_type=pl.lit("TCMD_RESPONSE"),
        general_message=pl.col("tcmd_response_text"),
        tcmd_ts_sent=pl.col("tcmd_ts_sent"),
        tcmd_response_code=pl.col("tcmd_response_code"),
        tcmd_duration_ms=pl.col("tcmd_duration_ms"),
        tcmd_part_count=pl.col("part_count"),
        tcmd_received_part_count=pl.col("received_part_count"),
        tcmd_missing_seq_nums=pl.col("missing_seq_nums"),
        tcmd_is_complete=pl.col("is_complete"),
        tcmd_last_received_at=pl.col("last_received_at"),
        packet_ids=pl.col("packet_ids"),
        tcmd_response_text=pl.col("tcmd_response_text"),
    )


def _with_reassembled_tcmd(decoded: pl.LazyFrame, path: Path) -> pl.LazyFrame:
    """`decoded` with its raw TCMD_RESPONSE packets replaced by step 5's
    reassembled responses. Falls back to `decoded` as-is if step 5 hasn't
    run yet, rather than silently hiding every telecommand response.
    """
    reassembled = _reassembled_tcmd_rows(_reassembled_tcmd_path(path))
    if reassembled is None:
        return decoded

    base_columns = decoded.collect_schema().names()
    # The per-packet sequence numbers mean nothing once the packets are
    # joined -- `tcmd_part_count` etc. replace them.
    dropped = {"tcmd_response_seq_num", "tcmd_response_max_seq_num"}
    # The new columns take the dropped ones' place, among the tcmd_* header
    # columns (step 3 keeps tcmd_response_text far to the right).
    insert_at = next(
        (i for i, c in enumerate(base_columns) if c in dropped), len(base_columns)
    )
    kept_before = [c for c in base_columns[:insert_at] if c not in dropped]
    kept_after = [c for c in base_columns[insert_at:] if c not in dropped]
    ordered = [*kept_before, *_REASSEMBLY_ONLY_COLUMNS, *kept_after]

    combined = pl.concat(
        [
            decoded.filter(pl.col("packet_type") != "TCMD_RESPONSE").drop(
                dropped, strict=False
            ),
            reassembled,
        ],
        how="diagonal_relaxed",
    )
    return combined.select(ordered)


def _source(path: Path, filters: PacketBrowserFilters) -> pl.LazyFrame | None:
    """The table the page browses: `everything_decoded` as-is, or with its
    TCMD_RESPONSE packets reassembled, per `filters`.
    """
    lf = _scan(path)
    if lf is None or not filters.reassemble_tcmd_responses:
        return lf
    return _with_reassembled_tcmd(lf, path)


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
    sort: PacketSort | None = None,
) -> PacketPage:
    """One page of rows matching `filters`, ordered by `sort` (newest first
    by default), restricted to columns that aren't all-null across the
    *entire* filtered set (not just this page).

    A `sort.column` not in the file (e.g. left over from before a schema
    change) falls back to the default order rather than erroring.
    """
    sort = sort or PacketSort()
    lf = _source(path, filters)
    if lf is None:
        return PacketPage(rows=pl.DataFrame(), total_rows=0, columns=[])

    column_names = lf.collect_schema().names()
    filtered = _apply_filters(lf, filters)
    total, non_null_columns = _total_and_non_null_columns(filtered, column_names)
    if total == 0:
        return PacketPage(rows=pl.DataFrame(), total_rows=0, columns=[])

    if sort.column not in column_names:
        sort = PacketSort()
    # received_at as a tiebreaker keeps paging stable when sorting by a
    # low-cardinality column (packet_type, a boolean flag, ...).
    by, descending = [sort.column], [sort.descending]
    if sort.column != "received_at":
        by.append("received_at")
        descending.append(True)
    rows = (
        filtered.sort(by, descending=descending, nulls_last=True)
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
    lf = _source(path, filters)
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
