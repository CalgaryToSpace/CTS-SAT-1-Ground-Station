from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses import (
    pipeline as step_5_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui import packet_browser
from cts1_mo_tools.cts1_processing_pipeline.web_ui.packet_browser import (
    PacketBrowserFilters,
    PacketSort,
    export_filtered_csv,
    load_page,
    reassembled_tcmd_responses_available,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_BASE_EPOCH_MS = int(_BASE.timestamp() * 1000)

_RAW = PacketBrowserFilters()
_REASSEMBLED = PacketBrowserFilters(reassemble_tcmd_responses=True)


def _row(  # noqa: PLR0913
    *,
    packet_id: str,
    seconds: float,
    packet_type: str = "TCMD_RESPONSE",
    text: str | None = None,
    seq_num: int | None = None,
    max_seq_num: int | None = None,
    ts_sent: int | None = None,
) -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "csp_crc_valid": True,
        "received_at": _BASE + timedelta(seconds=seconds),
        "packet_type": packet_type,
        "general_message": text,
        "tcmd_ts_sent": ts_sent,
        "tcmd_response_code": 0 if packet_type == "TCMD_RESPONSE" else None,
        "tcmd_duration_ms": 500 if packet_type == "TCMD_RESPONSE" else None,
        "tcmd_response_seq_num": seq_num,
        "tcmd_response_max_seq_num": max_seq_num,
        "log_message": text if packet_type == "LOG_MESSAGE" else None,
        "tcmd_response_text": text if packet_type == "TCMD_RESPONSE" else None,
    }


def _write_data_dir(tmp_path: Path, *, run_step_5: bool = True) -> Path:
    """A 2-part response (received out of order), a 1-part response, and a
    log message, written as step 3's output (plus step 5's, if asked).
    """
    pl.DataFrame(
        [
            _row(packet_id="log", seconds=1, packet_type="LOG_MESSAGE", text="hi"),
            _row(
                packet_id="a2",
                seconds=10,
                text="-part-two",
                seq_num=2,
                max_seq_num=2,
                ts_sent=_BASE_EPOCH_MS,
            ),
            _row(
                packet_id="a1",
                seconds=11,
                text="part-one",
                seq_num=1,
                max_seq_num=2,
                ts_sent=_BASE_EPOCH_MS,
            ),
            _row(
                packet_id="b1",
                seconds=20,
                text="single",
                seq_num=1,
                max_seq_num=1,
                ts_sent=_BASE_EPOCH_MS + 1,
            ),
        ]
    ).write_parquet(tmp_path / "everything_decoded.parquet")
    if run_step_5:
        step_5_pipeline.run(data_dir=tmp_path)
    return tmp_path / "everything_decoded.parquet"


def test_raw_view_unchanged(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path)
    page = load_page(path, _RAW, offset=0)
    assert page.total_rows == 4
    assert "tcmd_response_seq_num" in page.columns
    assert "tcmd_part_count" not in page.columns


def test_reassembled_view_joins_multi_packet_responses(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path)
    page = load_page(path, _REASSEMBLED, offset=0)
    assert page.total_rows == 3  # log + 2 responses

    tcmd = page.rows.filter(pl.col("packet_type") == "TCMD_RESPONSE").sort(
        "received_at"
    )
    assert tcmd["general_message"].to_list() == ["part-one-part-two", "single"]
    assert tcmd["tcmd_response_text"].to_list() == ["part-one-part-two", "single"]
    assert tcmd["tcmd_part_count"].to_list() == [2, 1]
    assert tcmd["packet_ids"].to_list() == ['["a1","a2"]', '["b1"]']
    # received_at is the response's *first* packet's arrival.
    assert tcmd["received_at"].to_list()[0] == _BASE + timedelta(seconds=10)

    # Per-packet sequence numbers are meaningless once joined.
    assert "tcmd_response_seq_num" not in page.columns
    assert "tcmd_response_max_seq_num" not in page.columns


def test_reassembled_view_filters_and_sorts(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path)
    filters = PacketBrowserFilters(
        message_substring="PART-TWO", reassemble_tcmd_responses=True
    )
    page = load_page(path, filters, offset=0)
    assert page.rows["general_message"].to_list() == ["part-one-part-two"]

    page = load_page(
        path,
        _REASSEMBLED,
        offset=0,
        sort=PacketSort("tcmd_part_count", descending=True),
    )
    assert page.rows["tcmd_part_count"].to_list()[0] == 2


def test_reassembled_view_export(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path)
    result = export_filtered_csv(path, _REASSEMBLED)
    assert result.row_count == 3
    assert b"part-one-part-two" in result.csv_bytes


def test_reassembled_view_falls_back_to_raw_without_step_5(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path, run_step_5=False)
    assert not reassembled_tcmd_responses_available(path)
    page = load_page(path, _REASSEMBLED, offset=0)
    assert page.total_rows == 4


def test_availability_reflects_step_5_output(tmp_path: Path) -> None:
    path = _write_data_dir(tmp_path)
    assert reassembled_tcmd_responses_available(path)
    assert packet_browser.packet_type_options(path) == [
        "LOG_MESSAGE",
        "TCMD_RESPONSE",
    ]
