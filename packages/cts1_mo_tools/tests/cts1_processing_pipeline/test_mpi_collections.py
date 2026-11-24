from datetime import UTC, datetime

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.mpi_collections import (
    analyses_of,
    downlinked_percent_text,
    strong_signal_counts_text,
    strong_signals_downlinked_text,
    time_sync_sources_text,
)


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


def test_downlinked_percent_text() -> None:
    assert (
        downlinked_percent_text(
            {
                "frames_present": 200,
                "frames_expected": 250,
                "analysis_valid_frame_count": 400,
            }
        )
        == "50.0%"
    )
    assert (
        downlinked_percent_text({"frames_present": 200, "frames_expected": 250})
        == "80.0%"
    )
    assert (
        downlinked_percent_text(
            {"frames_present": 0, "analysis_valid_frame_count": 535}
        )
        == "0.0%"
    )
    assert downlinked_percent_text({"frames_present": 0, "frames_expected": None}) == ""
