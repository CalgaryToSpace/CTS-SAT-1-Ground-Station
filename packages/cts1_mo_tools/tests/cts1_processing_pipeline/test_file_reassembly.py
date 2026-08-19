from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.file_reassembly import (
    MAX_REASSEMBLY_SPAN_BYTES,
    find_header_candidates,
    reassemble_bulk_chunks,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _chunk(
    *, offset: int, data: bytes, received_at: str = "2026-01-01T00:00:00"
) -> dict[str, Any]:
    return {
        "bulk_file_offset": offset,
        "bulk_data_len": len(data),
        "bulk_data_hex": data.hex(),
        "received_at": _dt(received_at),
    }


def _chunks_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# reassemble_bulk_chunks
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    result = reassemble_bulk_chunks(_chunks_df([]))
    assert result.total_chunks == 0
    assert result.unique_offsets == 0
    assert result.data == b""
    assert not result.duplicates
    assert not result.has_conflicts
    assert result.is_gapless


def test_clean_contiguous_chunks_reassemble_exactly() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=4, data=b"BBBB"),
            _chunk(offset=8, data=b"CC"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.data == b"AAAABBBBCC"
    assert result.total_chunks == 3
    assert result.unique_offsets == 3
    assert not result.duplicates
    assert not result.has_conflicts
    assert result.is_gapless
    assert result.span_bytes == 10
    assert result.covered_bytes == 10
    assert result.sha256 == __import__("hashlib").sha256(b"AAAABBBBCC").hexdigest()
    # Every distinct offset is summarized in `.offsets`, not just duplicates
    # -- so a clean, gapless download still has something to show in the
    # UI's chunks table.
    assert [o.offset for o in result.offsets] == [0, 4, 8]
    assert all(o.count == 1 for o in result.offsets)
    assert all(o.consistent for o in result.offsets)


def test_offsets_are_summarized_even_with_gaps_and_no_duplicates() -> None:
    """A partial/sparse download (missing chunks, but no offset repeated)
    should still populate `.offsets` -- this is the case the web UI's
    chunks table needs to keep rendering rather than going blank.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            # bytes 4..8 missing -- never downlinked (yet)
            _chunk(offset=8, data=b"CCCC"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert not result.is_gapless
    assert not result.duplicates
    assert [o.offset for o in result.offsets] == [0, 8]
    assert [o.count for o in result.offsets] == [1, 1]


def test_out_of_order_chunks_still_reassemble_correctly() -> None:
    df = _chunks_df(
        [
            _chunk(offset=8, data=b"CC"),
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=4, data=b"BBBB"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.data == b"AAAABBBBCC"


def test_gap_in_the_middle_is_reported_and_zero_filled() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            # bytes 4..8 missing
            _chunk(offset=8, data=b"CCCC"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert not result.is_gapless
    assert result.gaps == ((4, 8),)
    assert result.data == b"AAAA\x00\x00\x00\x00CCCC"
    assert result.covered_bytes == 8
    assert result.span_bytes == 12


def test_gap_at_the_start_is_reported() -> None:
    df = _chunks_df([_chunk(offset=4, data=b"BBBB")])
    result = reassemble_bulk_chunks(df)
    assert result.gaps == ((0, 4),)
    assert result.data == b"\x00\x00\x00\x00BBBB"


def test_duplicate_offset_with_agreeing_bytes_is_not_a_conflict() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"AAAA", received_at="2026-01-01T00:00:05"),
            _chunk(offset=4, data=b"BBBB"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    # Duplicated (e.g. retransmitted) offsets are routine and listed, but
    # since the copies agree, this isn't a conflict.
    assert len(result.duplicates) == 1
    dup = result.duplicates[0]
    assert dup.offset == 0
    assert dup.count == 2
    assert dup.distinct_contents_count == 1
    assert dup.lengths == (4,)
    assert dup.consistent
    assert not result.has_conflicts
    assert result.conflicts == ()
    assert result.data == b"AAAABBBB"


def test_duplicate_offset_with_conflicting_bytes_is_a_conflict() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"ZZZZ"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.has_conflicts
    assert len(result.conflicts) == 1
    dup = result.duplicates[0]
    assert dup.count == 2
    assert dup.distinct_contents_count == 2
    assert not dup.consistent
    # Last row in the given DataFrame order wins.
    assert result.data == b"ZZZZ"


def test_duplicate_offset_with_differing_lengths_reports_both() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"AAAAAA"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    dup = result.duplicates[0]
    assert dup.lengths == (4, 6)


def test_span_over_safety_cap_raises() -> None:
    df = _chunks_df([_chunk(offset=MAX_REASSEMBLY_SPAN_BYTES + 1, data=b"A")])
    with pytest.raises(ValueError, match="safety cap"):
        reassemble_bulk_chunks(df)


# ---------------------------------------------------------------------------
# find_header_candidates
# ---------------------------------------------------------------------------


def _tcmd_row(
    *,
    text: str,
    ts_sent: int = 1,
    seq_num: int = 0,
    max_seq_num: int = 0,
    received_at: str = "2026-01-01T00:00:00",
) -> dict[str, Any]:
    return {
        "received_at": _dt(received_at),
        "tcmd_ts_sent": ts_sent,
        "tcmd_response_seq_num": seq_num,
        "tcmd_response_max_seq_num": max_seq_num,
        "tcmd_response_text": text,
    }


def _tcmd_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def test_no_tcmd_responses_returns_empty() -> None:
    assert find_header_candidates(_tcmd_df([])) == []


def test_non_header_response_is_ignored() -> None:
    df = _tcmd_df([_tcmd_row(text="OK")])
    assert find_header_candidates(df) == []


def test_full_single_packet_header_is_parsed() -> None:
    text = (
        '{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/log_b51a.TLM",'
        '"file_size":24874,"crc16":"0xb51a","sd_card_index":2}'
    )
    df = _tcmd_df([_tcmd_row(text=text)])
    candidates = find_header_candidates(df)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.action == "adcs_get_latest_sd_file_blob"
    assert c.file == "ADCS/log_b51a.TLM"
    assert c.file_size == 24874
    assert c.crc16 == "0xb51a"
    assert c.offset is None
    assert c.length is None


def test_header_with_offset_and_length_is_parsed() -> None:
    text = (
        '{"action":"bulk_downlink_start_blob","file":"mpi_data/2026-08-08.mpi",'
        '"file_size":1443001,"sha256":'
        '"543751B0D1ADE688FB6654EB5DD4B70F840AB9DCAA95F7C432CEBCBC7B94EBA4",'
        '"offset":160485,"length":1170}'
    )
    df = _tcmd_df([_tcmd_row(text=text)])
    (c,) = find_header_candidates(df)
    assert c.file == "mpi_data/2026-08-08.mpi"
    assert c.file_size == 1443001
    assert (
        c.sha256 == "543751b0d1ade688fb6654eb5dd4b70f840ab9dcaa95f7c432cebcbc7b94eba4"
    )
    assert c.offset == 160485
    assert c.length == 1170


def test_truncated_json_still_yields_partial_fields() -> None:
    # A long file path pushes the sha256 value past the 186-byte TCMD_RESPONSE
    # cap -- real firmware behavior this decoder has to tolerate, per the
    # module docstring. No closing brace, sha256 cut off mid-hex-string.
    text = (
        '{"action":"bulk_downlink_start_blob","file":"mpi_data/2026-08-08.mpi",'
        '"file_size":1443001,"sha256":"543751B0D1ADE688FB6654EB5DD4B70F840AB9DCAA95F7C4'
    )
    df = _tcmd_df([_tcmd_row(text=text)])
    (c,) = find_header_candidates(df)
    assert c.file == "mpi_data/2026-08-08.mpi"
    assert c.file_size == 1443001
    assert c.sha256 == "543751b0d1ade688fb6654eb5dd4b70f840ab9dcaa95f7c4"
    assert len(c.sha256) < 64


def test_multi_fragment_response_is_reassembled_by_seq_num() -> None:
    full_text = (
        '{"action":"bulk_downlink_start_blob","file":"mpi_data/2026-08-08.mpi",'
        '"file_size":1443001,"sha256":'
        '"97B1681F8D5E95F7EA5443DDDA92FE2071697E5FB22B4B8137356E3D6E83C508",'
        '"offset":185055,"length":1170}'
    )
    mid = len(full_text) // 2
    df = _tcmd_df(
        [
            # Deliberately out of order in the input -- seq_num must drive
            # reassembly order, not row order.
            _tcmd_row(text=full_text[mid:], ts_sent=42, seq_num=1, max_seq_num=1),
            _tcmd_row(text=full_text[:mid], ts_sent=42, seq_num=0, max_seq_num=1),
        ]
    )
    candidates = find_header_candidates(df)
    # Both the reassembled (full-text) candidate and each individual
    # fragment's own best-effort parse are returned -- the fragment alone
    # doesn't reach the "offset"/"length" fields, so the reassembled one is
    # identified as whichever candidate has them.
    reassembled = next(c for c in candidates if c.offset is not None)
    assert reassembled.file == "mpi_data/2026-08-08.mpi"
    assert reassembled.file_size == 1443001
    assert reassembled.offset == 185055
    assert reassembled.length == 1170


def test_multiple_distinct_headers_are_all_returned() -> None:
    df = _tcmd_df(
        [
            _tcmd_row(
                text='{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/a.TLM",'
                '"file_size":100}',
                ts_sent=1,
                received_at="2026-01-01T00:00:00",
            ),
            _tcmd_row(
                text='{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/b.TLM",'
                '"file_size":200}',
                ts_sent=2,
                received_at="2026-01-01T00:05:00",
            ),
        ]
    )
    candidates = find_header_candidates(df)
    assert [c.file for c in candidates] == ["ADCS/a.TLM", "ADCS/b.TLM"]
    assert [c.received_at for c in candidates] == sorted(
        c.received_at for c in candidates
    )
