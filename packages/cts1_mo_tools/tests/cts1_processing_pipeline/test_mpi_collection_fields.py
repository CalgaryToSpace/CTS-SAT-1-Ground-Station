import csv
import io
from datetime import UTC, datetime

import openpyxl
import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections import (
    pipeline as step_8_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui import mpi_collection_fields

_SCHEMA = step_8_pipeline._COLLECTIONS_SCHEMA  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_every_collection_field_is_in_a_group() -> None:
    row = dict.fromkeys(_SCHEMA, 0)
    assert mpi_collection_fields.other_fields(row) == {}


def test_every_grouped_field_is_a_collection_field() -> None:
    fields = {
        name
        for group in mpi_collection_fields.COLUMN_GROUPS
        for column in (*group.columns, *group.details)
        for name in (column.sources or (column.field,))
    }
    assert fields - set(_SCHEMA) <= {
        "_downlinked_percent",
        "_time_sync_age",
    }


_COLLECTIONS = pl.DataFrame(
    {
        "collection_started_at": [datetime(2026, 10, 21, 20, 32, 6, tzinfo=UTC)],
        "collection_duration_ms": [117_758],
        "frames_present": [0],
        "analysis_file": ["mpi_data/2026-10-21_203206Z.mpi"],
        "analysis_valid_frame_count": [667],
        "some_new_field": ["x"],  # In no group.
    }
)


def _group_of(groups: list[str], i: int) -> str:
    """The group over column `i`: each group's label is over its first."""
    return next(g for g in reversed(groups[: i + 1]) if g)


def test_csv_export_is_grouped_and_formatted_as_shown() -> None:
    groups, labels, row = csv.reader(
        io.StringIO(mpi_collection_fields.collections_csv_bytes(_COLLECTIONS).decode())
    )
    assert groups[0] == "Collection"
    assert row[labels.index("Started (UTC)")] == "2026-10-21 20:32:06.000"
    assert row[labels.index("Duration")] == "1m 57s"
    valid, new = labels.index("Valid Frames"), labels.index("some_new_field")
    assert row[valid] == "667"
    assert _group_of(groups, valid) == "On-Board Analysis (analyze_mpi_data)"
    assert (row[new], _group_of(groups, new)) == ("x", "Other")
    assert row[labels.index("Stop Reason")] == ""  # Not in this row.


def test_excel_export_has_grouped_sheet_and_raw_table() -> None:
    workbook = openpyxl.load_workbook(
        io.BytesIO(mpi_collection_fields.collections_xlsx_bytes(_COLLECTIONS))
    )
    assert workbook.sheetnames == ["As Shown", "mpi_collections"]
    shown = workbook["As Shown"]
    groups, labels, row = shown.iter_rows(values_only=True)
    assert groups[0] == "Collection"
    assert row[list(labels).index("Valid Frames")] == 667
    # Each group's label is merged across its columns.
    assert "A1:E1" in {str(r) for r in shown.merged_cells.ranges}  # Collection
    header, raw = workbook["mpi_collections"].iter_rows(values_only=True)
    assert list(header) == _COLLECTIONS.columns
    assert raw[1] == 117_758
