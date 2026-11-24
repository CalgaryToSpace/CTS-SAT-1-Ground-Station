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
