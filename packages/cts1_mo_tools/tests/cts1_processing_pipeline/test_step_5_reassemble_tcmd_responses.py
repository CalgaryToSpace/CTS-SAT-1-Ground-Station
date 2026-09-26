from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_5_reassemble_tcmd_responses.pipeline import (  # noqa: E501
    MISSING_PART_FILL,
    compute_reassembled_tcmd_responses,
    run,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_BASE_EPOCH_MS = int(_BASE.timestamp() * 1000)


def _at(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


_packet_counter = 0


def _tcmd(  # noqa: PLR0913
    *,
    text: str,
    seq_num: int = 1,
    max_seq_num: int = 1,
    ts_sent: int = _BASE_EPOCH_MS,
    seconds: float = 10.0,
    response_code: int = 0,
    duration_ms: int = 500,
    csp_crc_valid: bool = True,
    packet_id: str | None = None,
) -> dict[str, Any]:
    global _packet_counter  # noqa: PLW0603
    _packet_counter += 1
    return {
        "packet_id": packet_id or f"pkt{_packet_counter:04d}",
        "packet_type": "TCMD_RESPONSE",
        "csp_crc_valid": csp_crc_valid,
        "received_at": _at(seconds),
        "tcmd_ts_sent": ts_sent,
        "tcmd_response_code": response_code,
        "tcmd_duration_ms": duration_ms,
        "tcmd_response_seq_num": seq_num,
        "tcmd_response_max_seq_num": max_seq_num,
        "tcmd_response_text": text,
    }


def _df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_empty_input_returns_empty_with_expected_schema() -> None:
    result = compute_reassembled_tcmd_responses(pl.DataFrame())
    assert result.height == 0
    assert result.schema["tcmd_response_text"] == pl.String
    assert result.schema["first_received_at"] == pl.Datetime("us", "UTC")
    assert result.schema["is_complete"] == pl.Boolean


def test_no_tcmd_responses_returns_empty_with_expected_schema() -> None:
    row = _tcmd(text="ignored")
    row["packet_type"] = "LOG_MESSAGE"
    result = compute_reassembled_tcmd_responses(_df([row]))
    assert result.height == 0
    assert result.schema == compute_reassembled_tcmd_responses(pl.DataFrame()).schema


def test_single_packet_response_passes_through() -> None:
    result = compute_reassembled_tcmd_responses(_df([_tcmd(text="SUCCESS: ok")]))
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["tcmd_response_text"] == "SUCCESS: ok"
    assert row["part_count"] == 1
    assert row["received_part_count"] == 1
    assert row["is_complete"] is True
    assert row["missing_seq_nums"] == "[]"
    assert row["tcmd_sent_at"] == _BASE


def test_multi_packet_response_joined_in_seq_order() -> None:
    # Received out of order: the joined text must follow seq_num, not
    # received_at.
    df = _df(
        [
            _tcmd(text="C", seq_num=3, max_seq_num=3, seconds=1, packet_id="p3"),
            _tcmd(text="A", seq_num=1, max_seq_num=3, seconds=3, packet_id="p1"),
            _tcmd(text="B", seq_num=2, max_seq_num=3, seconds=2, packet_id="p2"),
        ]
    )
    result = compute_reassembled_tcmd_responses(df)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["tcmd_response_text"] == "ABC"
    assert row["is_complete"] is True
    assert row["received_part_count"] == 3
    assert row["packet_ids"] == '["p1","p2","p3"]'
    assert row["first_received_at"] == _at(1)
    assert row["last_received_at"] == _at(3)


def test_missing_parts_filled_with_question_marks() -> None:
    df = _df(
        [
            _tcmd(text="A", seq_num=1, max_seq_num=4),
            _tcmd(text="C", seq_num=3, max_seq_num=4),
        ]
    )
    row = compute_reassembled_tcmd_responses(df).row(0, named=True)
    assert (
        row["tcmd_response_text"] == "A" + MISSING_PART_FILL + "C" + MISSING_PART_FILL
    )
    assert set(MISSING_PART_FILL) == {"?"}
    assert row["is_complete"] is False
    assert row["received_part_count"] == 2
    assert row["missing_seq_nums"] == "[2,4]"


def test_different_ts_sent_are_separate_responses() -> None:
    df = _df(
        [
            _tcmd(text="A1", seq_num=1, max_seq_num=2, ts_sent=_BASE_EPOCH_MS),
            _tcmd(text="B1", seq_num=1, max_seq_num=2, ts_sent=_BASE_EPOCH_MS + 1),
            _tcmd(text="A2", seq_num=2, max_seq_num=2, ts_sent=_BASE_EPOCH_MS),
            _tcmd(text="B2", seq_num=2, max_seq_num=2, ts_sent=_BASE_EPOCH_MS + 1),
        ]
    )
    result = compute_reassembled_tcmd_responses(df)
    assert sorted(result["tcmd_response_text"].to_list()) == ["A1A2", "B1B2"]


def test_shared_ts_sent_with_different_headers_kept_apart() -> None:
    # Both sent before time sync (ts_sent == 0), but distinguishable by
    # their other shared header fields.
    df = _df(
        [
            _tcmd(text="first", ts_sent=0, duration_ms=100),
            _tcmd(text="second", ts_sent=0, duration_ms=200),
        ]
    )
    result = compute_reassembled_tcmd_responses(df)
    assert sorted(result["tcmd_response_text"].to_list()) == ["first", "second"]
    assert result["tcmd_sent_at"].is_null().all()


def test_duplicate_part_uses_earliest_received_copy() -> None:
    df = _df(
        [
            _tcmd(text="late", seq_num=1, max_seq_num=2, seconds=50),
            _tcmd(text="early", seq_num=1, max_seq_num=2, seconds=5),
            _tcmd(text="-tail", seq_num=2, max_seq_num=2, seconds=6),
        ]
    )
    row = compute_reassembled_tcmd_responses(df).row(0, named=True)
    assert row["tcmd_response_text"] == "early-tail"
    assert row["received_part_count"] == 2


def test_invalid_crc_packets_ignored() -> None:
    df = _df(
        [
            _tcmd(text="A", seq_num=1, max_seq_num=2),
            _tcmd(text="garbage", seq_num=2, max_seq_num=2, csp_crc_valid=False),
        ]
    )
    row = compute_reassembled_tcmd_responses(df).row(0, named=True)
    assert row["tcmd_response_text"] == "A" + MISSING_PART_FILL
    assert row["missing_seq_nums"] == "[2]"


def test_newest_first() -> None:
    df = _df(
        [
            _tcmd(text="old", ts_sent=_BASE_EPOCH_MS, seconds=1),
            _tcmd(text="new", ts_sent=_BASE_EPOCH_MS + 1, seconds=100),
        ]
    )
    result = compute_reassembled_tcmd_responses(df)
    assert result["tcmd_response_text"].to_list() == ["new", "old"]


def test_run_writes_parquet(tmp_path: Any) -> None:
    _df(
        [
            _tcmd(text="A", seq_num=1, max_seq_num=2),
            _tcmd(text="B", seq_num=2, max_seq_num=2),
        ]
    ).write_parquet(tmp_path / "everything_decoded.parquet")
    run(data_dir=tmp_path)
    out = pl.read_parquet(tmp_path / "reassembled_tcmd_responses.parquet")
    assert out["tcmd_response_text"].to_list() == ["AB"]
