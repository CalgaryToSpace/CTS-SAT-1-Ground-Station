import io
from datetime import UTC, datetime

import openpyxl
import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.mpi_collections import (
    analyses_of,
    collections_csv_bytes,
    collections_xlsx_bytes,
    strong_signal_counts_text,
    strong_signals_downlinked_text,
    time_sync_sources_text,
)

_COLLECTIONS = pl.DataFrame(
    {
        "collection_started_at": [datetime(2026, 10, 21, 20, 32, 6, tzinfo=UTC)],
        "frames_present": [0],
        "analysis_file": ["mpi_data/2026-10-21_203206Z.mpi"],
        "analysis_valid_frame_count": [667],
    }
)


def test_csv_export_has_every_column() -> None:
    exported = pl.read_csv(io.BytesIO(collections_csv_bytes(_COLLECTIONS)))
    assert exported.columns == _COLLECTIONS.columns
    assert exported["analysis_valid_frame_count"].to_list() == [667]


def test_excel_export_has_every_column() -> None:
    workbook = openpyxl.load_workbook(io.BytesIO(collections_xlsx_bytes(_COLLECTIONS)))
    header, row = workbook["mpi_collections"].iter_rows(values_only=True)
    assert list(header) == _COLLECTIONS.columns
    assert row[2] == "mpi_data/2026-10-21_203206Z.mpi"


def test_strong_signal_counts_text() -> None:
    assert strong_signal_counts_text('{"100":1,"200":0}') == "≥100 DN: 1 · ≥200 DN: 0"
    assert strong_signal_counts_text(None) == ""


def test_analyses_of_a_collection_puts_the_used_one_first() -> None:
    received_at = datetime(2026, 10, 22, tzinfo=UTC)
    analyses = pl.DataFrame(
        {
            "analysis_file": ["a.mpi", "a.mpi", "b.mpi"],
            "analysis_version": [1, 2, 1],
            "analysis_file_size": [100, 50, 100],
            "analysis_received_at": [received_at] * 3,
        }
    )
    matching = analyses_of(
        analyses, {"analysis_file": "a.mpi", "analysis_action": "analyze_mpi_data_v2"}
    )
    assert [a["analysis_version"] for a in matching] == [2, 1]
    assert analyses_of(analyses, {"analysis_file": None, "analysis_action": None}) == []


def test_strong_signals_downlinked_text() -> None:
    signals = [{"frames": [5, 6]}, {"frames": [9, 11]}, {"frames": [50, 50]}, {}]
    assert strong_signals_downlinked_text(signals, "[2,1,0,3]") == [
        "yes (2/2 frames)",
        "partly (1/3 frames)",
        "no (0/1 frames)",
        "yes (3 frames)",
    ]
    assert strong_signals_downlinked_text(signals[:1], None) == ["unknown"]


def test_time_sync_sources_text() -> None:
    assert time_sync_sources_text('["E","G"]') == "EPS_RTC (E), GNSS_UART (G)"
    assert time_sync_sources_text('["X"]') == "UNKNOWN (X)"
    assert time_sync_sources_text("[]") == ""
    assert time_sync_sources_text(None) == ""
