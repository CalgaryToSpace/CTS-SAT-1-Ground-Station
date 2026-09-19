import hashlib
import io
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.file_reassembly import (
    BULK_DOWNLINK_MAX_DATA,
    CONFLICT_ISLAND_MAX_BYTES,
    COVERAGE_OFFSET_LABEL_INTERVAL_ROWS,
    COVERAGE_ROW_WIDTH_BYTES,
    MAX_REASSEMBLY_SPAN_BYTES,
    ByteSegment,
    ByteStatus,
    ConflictPolicy,
    ReassemblyResult,
    coverage_gutter_width_px,
    coverage_label_interval_rows,
    coverage_png_size,
    coverage_ruler_height_px,
    detect_image,
    find_header_candidates,
    reassemble_bulk_chunks,
    render_coverage_png,
)
from PIL import Image


def _segments(result: ReassemblyResult) -> list[tuple[int, int, str]]:
    """`(start, end, status)` triples -- the shape the assertions below read
    most clearly, and the shape the UI's segment table renders.
    """
    return [(s.start, s.end, str(s.status)) for s in result.segments]


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
    assert result.segments == ()
    assert not result.duplicates
    assert not result.has_conflicts
    assert result.is_gapless
    # An empty selection has nothing wrong with it, but it isn't a file
    # either -- `is_complete` gating the export's "_partial" suffix must not
    # call it one.
    assert not result.is_complete


def test_missing_required_column_raises() -> None:
    df = pl.DataFrame({"bulk_file_offset": [0], "bulk_data_hex": ["00"]})
    with pytest.raises(ValueError, match="missing required column"):
        reassemble_bulk_chunks(df)


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
    assert result.is_complete
    assert result.span_bytes == 10
    assert result.covered_bytes == 10
    assert result.sha256 == hashlib.sha256(b"AAAABBBBCC").hexdigest()
    # Three packets, but one run of good bytes -- the assessment is per byte,
    # and consecutive bytes of the same verdict collapse into one segment.
    assert _segments(result) == [(0, 10, "Good")]
    # Per-offset packet bookkeeping is still summarized alongside it.
    assert [o.offset for o in result.offsets] == [0, 4, 8]
    assert all(o.count == 1 for o in result.offsets)


def test_offsets_are_summarized_even_with_gaps_and_no_duplicates() -> None:
    """A partial/sparse download (missing chunks, but no offset repeated)
    should still populate `.offsets` -- the per-offset packet bookkeeping the
    UI shows alongside the per-byte verdict.
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
    assert _segments(result) == [(0, 10, "Good")]


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
    assert not result.is_complete
    assert result.gaps == ((4, 8),)
    assert result.data == b"AAAA\x00\x00\x00\x00CCCC"
    assert result.covered_bytes == 8
    assert result.missing_bytes == 4
    assert result.span_bytes == 12
    assert _segments(result) == [(0, 4, "Good"), (4, 8, "Missing"), (8, 12, "Good")]


def test_assessment_starts_at_byte_zero_when_the_first_chunk_is_missing() -> None:
    """The whole point of assessing from byte 0: a download whose opening
    chunk never arrived must *say* its first bytes are missing, not quietly
    start the report at the first byte that did arrive.
    """
    df = _chunks_df([_chunk(offset=4, data=b"BBBB")])
    result = reassemble_bulk_chunks(df)
    assert result.segments[0].start == 0
    assert result.segments[0].status is ByteStatus.MISSING
    assert result.gaps == ((0, 4),)
    assert result.data == b"\x00\x00\x00\x00BBBB"
    assert _segments(result) == [(0, 4, "Missing"), (4, 8, "Good")]


def test_segments_tile_the_whole_span_with_no_holes_or_repeats() -> None:
    df = _chunks_df(
        [
            _chunk(offset=10, data=b"AAAA"),
            _chunk(offset=10, data=b"ZZZZ"),
            _chunk(offset=20, data=b"BBBB"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.segments[0].start == 0
    assert result.segments[-1].end == result.span_bytes
    for previous, segment in zip(result.segments, result.segments[1:], strict=False):
        assert previous.end == segment.start
        # Maximal runs: two neighbours never share a status.
        assert previous.status is not segment.status
    assert (
        result.good_bytes + result.missing_bytes + result.conflict_bytes
        == result.span_bytes
    )


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
    # since the copies agree, every byte is good.
    assert len(result.duplicates) == 1
    dup = result.duplicates[0]
    assert dup.offset == 0
    assert dup.count == 2
    assert dup.distinct_contents_count == 1
    assert dup.lengths == (4,)
    assert dup.consistent
    assert not result.has_conflicts
    assert result.is_complete
    assert _segments(result) == [(0, 8, "Good")]
    assert result.data == b"AAAABBBB"


def test_only_the_disagreeing_bytes_are_flagged_conflicting() -> None:
    """The per-byte assessment's reason for existing: two copies of an offset
    that differ in one byte make *that byte* conflicting, not the whole
    packet's worth of bytes around it.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"AAZA"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.conflict_bytes == 1
    assert result.good_bytes == 3
    assert result.conflict_ranges == ((2, 3),)
    assert _segments(result) == [(0, 2, "Good"), (2, 3, "Conflicting"), (3, 4, "Good")]


def test_unaligned_overlap_between_different_offsets_is_assessed() -> None:
    """Two chunks at *different* offsets can overlap and disagree -- invisible
    to a per-offset check (neither offset repeats), caught per byte.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAAAA"),
            _chunk(offset=4, data=b"AZZZ"),  # overlaps bytes 4..6, differs at 5
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert not result.duplicates  # no offset repeats
    assert result.conflict_ranges == ((5, 6),)
    assert _segments(result) == [(0, 5, "Good"), (5, 6, "Conflicting"), (6, 8, "Good")]


def test_conflicting_bytes_keep_the_file_from_being_complete() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"ZZZZ"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.has_conflicts
    assert result.is_gapless  # every byte arrived...
    assert not result.is_complete  # ...but not trustworthily
    assert _segments(result) == [(0, 4, "Conflicting")]


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
    # The shorter copy simply has nothing to say about bytes 4..6, so those
    # aren't a conflict -- only what both copies cover can disagree.
    assert not result.has_conflicts
    assert result.data == b"AAAAAA"


def test_span_over_safety_cap_raises() -> None:
    df = _chunks_df([_chunk(offset=MAX_REASSEMBLY_SPAN_BYTES + 1, data=b"A")])
    with pytest.raises(ValueError, match="safety cap"):
        reassemble_bulk_chunks(df)


# ---------------------------------------------------------------------------
# conflict islands
# ---------------------------------------------------------------------------


def test_coincidentally_agreeing_bytes_are_absorbed_into_the_conflict() -> None:
    """Two transmissions of different data will share the odd byte by
    chance; those must not speckle the conflicted region with one-byte
    "good" runs, which is neither readable nor true corroboration.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAAAAAAAA"),
            #                       ^    ^  -- agree at bytes 2 and 7 only
            _chunk(offset=0, data=b"ZZAZZZZAZZ"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [(0, 10, "Conflicting")]
    assert result.conflict_bytes == 10
    assert result.good_bytes == 0


def test_a_long_agreeing_run_between_conflicts_is_kept() -> None:
    """The flip side: a stretch two transmissions really do share is worth
    seeing as its own run, so only runs up to
    `CONFLICT_ISLAND_MAX_BYTES` get absorbed.
    """
    agreed = b"S" * (CONFLICT_ISLAND_MAX_BYTES + 1)
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AA" + agreed + b"AA"),
            _chunk(offset=0, data=b"ZZ" + agreed + b"ZZ"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [
        (0, 2, "Conflicting"),
        (2, 2 + len(agreed), "Good"),
        (2 + len(agreed), 4 + len(agreed), "Conflicting"),
    ]


def test_absorption_needs_conflict_on_both_sides() -> None:
    """A short good run at the edge of a contested region -- bordering
    missing bytes or the end of the file -- is a boundary, not an island,
    so it keeps its own verdict.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"ZZZA"),  # byte 3 agrees, at the edge
            # bytes 4..8 never arrive, so the good byte borders a gap
            _chunk(offset=8, data=b"BBBB"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [
        (0, 3, "Conflicting"),
        (3, 4, "Good"),
        (4, 8, "Missing"),
        (8, 12, "Good"),
    ]


def test_absorbed_island_keeps_the_agreed_byte_value() -> None:
    """Re-labelling is a display decision, not a data one: an absorbed
    byte's copies agreed, so it already holds the value any policy would
    pick, whichever policy is in force.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAA", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"ZAZ", received_at="2026-01-01T00:00:01"),
        ]
    )
    for policy in ConflictPolicy:
        result = reassemble_bulk_chunks(df, policy=policy)
        assert _segments(result) == [(0, 3, "Conflicting")]
        assert result.data[1:2] == b"A"


# ---------------------------------------------------------------------------
# ConflictPolicy
# ---------------------------------------------------------------------------


def _conflicted_df() -> pl.DataFrame:
    """One byte, four disagreeing copies: "B" twice (the most common value,
    first and last to arrive), "A" once (earliest overall), "C" once.
    """
    return _chunks_df(
        [
            _chunk(offset=0, data=b"A", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:01"),
            _chunk(offset=0, data=b"C", received_at="2026-01-01T00:00:02"),
            _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:03"),
        ]
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ConflictPolicy.EARLIEST, b"A"),
        (ConflictPolicy.LATEST, b"B"),
        # "B" wins the tally either way; the tie-break only decides between
        # equally common values, of which there are none here.
        (ConflictPolicy.MOST_COMMON_THEN_EARLIEST, b"B"),
        (ConflictPolicy.MOST_COMMON_THEN_LATEST, b"B"),
    ],
)
def test_conflict_policy_picks_the_expected_copy(
    policy: ConflictPolicy, expected: bytes
) -> None:
    result = reassemble_bulk_chunks(_conflicted_df(), policy=policy)
    assert result.data == expected
    assert result.policy is policy
    # Resolving a conflict never hides it.
    assert result.has_conflicts
    assert result.conflict_ranges == ((0, 1),)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ConflictPolicy.MOST_COMMON_THEN_EARLIEST, b"A"),
        (ConflictPolicy.MOST_COMMON_THEN_LATEST, b"B"),
    ],
)
def test_most_common_tie_is_broken_by_recency(
    policy: ConflictPolicy, expected: bytes
) -> None:
    # "A" and "B" both arrive twice; only the tie-break separates them.
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"A", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:01"),
            _chunk(offset=0, data=b"A", received_at="2026-01-01T00:00:02"),
            _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:03"),
        ]
    )
    assert reassemble_bulk_chunks(df, policy=policy).data == expected


def test_policy_resolves_each_byte_independently() -> None:
    """A policy picks a winner per *byte*, not per packet: the later packet
    can be right about one byte and the earlier one right about another.
    """
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AX", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"AY", received_at="2026-01-01T00:00:01"),
            _chunk(offset=1, data=b"Y", received_at="2026-01-01T00:00:02"),
        ]
    )
    result = reassemble_bulk_chunks(df, policy=ConflictPolicy.MOST_COMMON_THEN_EARLIEST)
    # Byte 0: unanimous "A". Byte 1: "Y" twice beats "X" once.
    assert result.data == b"AY"
    assert _segments(result) == [(0, 1, "Good"), (1, 2, "Conflicting")]


def test_row_order_does_not_change_the_result() -> None:
    rows = [
        _chunk(offset=0, data=b"A", received_at="2026-01-01T00:00:00"),
        _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:01"),
        _chunk(offset=1, data=b"CC", received_at="2026-01-01T00:00:02"),
    ]
    forward = reassemble_bulk_chunks(_chunks_df(rows), policy=ConflictPolicy.EARLIEST)
    reversed_ = reassemble_bulk_chunks(
        _chunks_df(list(reversed(rows))), policy=ConflictPolicy.EARLIEST
    )
    assert forward.data == reversed_.data == b"ACC"
    assert forward.segments == reversed_.segments


# ---------------------------------------------------------------------------
# find_header_candidates
# ---------------------------------------------------------------------------


def _tcmd_row(
    *,
    text: str,
    ts_sent: int = 1,
    seq_num: int = 1,  # real firmware numbers frames 1..max_seq_num, not 0-based
    max_seq_num: int = 1,
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


def test_only_first_frame_is_parsed_later_frames_are_ignored() -> None:
    # Real firmware numbers frames 1..max_seq_num. The header fields this
    # module looks for are always written before TCMD_RESPONSE's 186-byte
    # cap bites, so seq_num 1 alone is trusted -- a later frame (here,
    # seq_num 2, arriving on its own row/timestamp) is never even looked at,
    # rather than being joined on or tried as its own candidate.
    df = _tcmd_df(
        [
            _tcmd_row(
                text='{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/log_aa21'
                '.TLM","file_size":26470,"crc16":"0xaa21","sha256":'
                '"74aa7f5b1ae416a9864a05b71cc2bfe8729d921f52d6405d2de1ddee2d4bc3'
                'c2","sd_car',
                ts_sent=7,
                seq_num=1,
                max_seq_num=2,
                received_at="2026-01-01T00:00:00",
            ),
            _tcmd_row(
                text='d_index":19}',  # trailing bytes only, no captured field
                ts_sent=7,
                seq_num=2,
                max_seq_num=2,
                received_at="2026-01-01T00:00:05",
            ),
        ]
    )
    candidates = find_header_candidates(df)
    assert len(candidates) == 1
    assert candidates[0].file == "ADCS/log_aa21.TLM"
    assert candidates[0].received_at == _dt("2026-01-01T00:00:00")


def test_second_frame_alone_is_not_a_candidate() -> None:
    # A row whose seq_num isn't 1 is skipped outright, even if its own text
    # happens to look header-shaped in isolation.
    df = _tcmd_df(
        [_tcmd_row(text='{"file":"x","file_size":1}', seq_num=2, max_seq_num=2)]
    )
    assert find_header_candidates(df) == []


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


# ---------------------------------------------------------------------------
# render_coverage_png
# ---------------------------------------------------------------------------


def _decode_indexed_png(png: bytes) -> tuple[int, int, list[bytes]]:
    """(width, height, rows of raw palette-index bytes) of an indexed PNG's
    *data area* -- the rulers `render_coverage_png` draws around it are
    cropped off, so these assertions stay about the coverage bitmap itself
    and don't shift every time the rulers' size changes.
    """
    image = Image.open(io.BytesIO(png))
    assert image.mode == "P"
    if image.size != (1, 1):  # the empty-result image has no rulers
        # The gutter is sized to the labels it has to print, so it's read
        # back off the image rather than assumed.
        gutter = image.size[0] - COVERAGE_ROW_WIDTH_BYTES
        image = image.crop((gutter, coverage_ruler_height_px(), *image.size))
    width, height = image.size
    pixels = image.tobytes()
    return (
        width,
        height,
        [pixels[y * width : (y + 1) * width] for y in range(height)],
    )


# Palette indices, per `file_reassembly._COVERAGE_STATUS_INDEX`.
_MISSING_PX, _GOOD_PX, _CONFLICT_PX, _PAD_PX, _INK_PX = 0, 1, 2, 3, 4


def test_coverage_png_empty_result_is_a_single_pixel() -> None:
    result = reassemble_bulk_chunks(_chunks_df([]))
    width, height, rows = _decode_indexed_png(render_coverage_png(result))
    assert (width, height) == (1, 1)
    assert rows[0] == bytes([_GOOD_PX])


def test_coverage_png_marks_gaps_and_good_bytes_correctly() -> None:
    # Spans two rows whatever COVERAGE_ROW_WIDTH_BYTES is set to, so the
    # wrap-onto-the-next-row arithmetic is exercised either way.
    last_offset = COVERAGE_ROW_WIDTH_BYTES + 195
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"A" * 195),
            # bytes 195..last_offset missing
            _chunk(offset=last_offset, data=b"A" * 195),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.gaps  # sanity: there is a gap
    width, height, rows = _decode_indexed_png(render_coverage_png(result))
    assert width == COVERAGE_ROW_WIDTH_BYTES
    assert height == 2

    # Recompute expected row/col for each gap byte and check it decoded red.
    for start, end in result.gaps:
        for byte_index in range(start, end):
            row, col = divmod(byte_index, COVERAGE_ROW_WIDTH_BYTES)
            assert rows[row][col] == _MISSING_PX
    # And received bytes decode green.
    for byte_index in (0, 194, last_offset, last_offset + 194):
        row, col = divmod(byte_index, COVERAGE_ROW_WIDTH_BYTES)
        assert rows[row][col] == _GOOD_PX


def test_coverage_png_pads_last_row_without_marking_it_good_or_missing() -> None:
    # A span that doesn't divide evenly into COVERAGE_ROW_WIDTH_BYTES --
    # the padding pixels must be distinguishable from the other
    # (good/missing/conflicting) states so they don't get misread as one.
    df = _chunks_df([_chunk(offset=0, data=b"A" * 10)])
    result = reassemble_bulk_chunks(df)
    _width, height, rows = _decode_indexed_png(render_coverage_png(result))
    assert height == 1
    assert rows[0][:10] == bytes([_GOOD_PX]) * 10
    assert set(rows[0][10:]) == {_PAD_PX}


def test_coverage_png_pad_index_is_transparent() -> None:
    """The padding block must not paint: it's past the end of the file, and a
    visible block there would read as real (missing or good) data.
    """
    result = reassemble_bulk_chunks(_chunks_df([_chunk(offset=0, data=b"A" * 10)]))
    image = Image.open(io.BytesIO(render_coverage_png(result)))
    assert image.info["transparency"] == _PAD_PX


def test_coverage_png_marks_conflicting_range_yellow() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"ZZZZ"),  # conflicts with the row above
            _chunk(offset=4, data=b"BBBB"),  # clean, no conflict
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.conflict_ranges == ((0, 4),)

    _width, _height, rows = _decode_indexed_png(render_coverage_png(result))
    assert list(rows[0][0:4]) == [_CONFLICT_PX] * 4
    assert list(rows[0][4:8]) == [_GOOD_PX] * 4


def test_conflicting_segments_merge_across_adjacent_offsets() -> None:
    # Two back-to-back conflicting offsets (0..4 and 4..8) are one run of
    # conflicting bytes, not two adjacent ones.
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"ZZZZ"),
            _chunk(offset=4, data=b"BBBB"),
            _chunk(offset=4, data=b"YYYY"),
            # A separate, non-adjacent conflict shouldn't merge with those.
            _chunk(offset=20, data=b"CCCC"),
            _chunk(offset=20, data=b"XXXX"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert result.conflict_ranges == ((0, 8), (20, 24))
    assert _segments(result) == [
        (0, 8, "Conflicting"),
        (8, 20, "Missing"),
        (20, 24, "Conflicting"),
    ]


# ---------------------------------------------------------------------------
# coverage map rulers
# ---------------------------------------------------------------------------


def _full_png_pixels(png: bytes) -> tuple[Image.Image, list[bytes]]:
    """The whole image (rulers included) and its rows of palette indices."""
    image = Image.open(io.BytesIO(png))
    width, _height = image.size
    pixels = image.tobytes()
    return image, [pixels[y * width : (y + 1) * width] for y in range(image.size[1])]


def test_coverage_png_size_matches_the_encoded_image() -> None:
    result = reassemble_bulk_chunks(
        _chunks_df([_chunk(offset=0, data=b"A" * COVERAGE_ROW_WIDTH_BYTES * 3)])
    )
    image = Image.open(io.BytesIO(render_coverage_png(result)))
    assert image.size == coverage_png_size(result)
    # Rulers plus three full rows of data.
    assert image.size == (
        coverage_gutter_width_px(3, COVERAGE_ROW_WIDTH_BYTES)
        + COVERAGE_ROW_WIDTH_BYTES,
        coverage_ruler_height_px() + 3,
    )


def test_top_ruler_ticks_every_packet_boundary() -> None:
    """The tick column must land exactly on the first pixel of each packet,
    which is the whole reason the rulers are drawn into the image instead of
    being positioned over it in HTML.
    """
    result = reassemble_bulk_chunks(
        _chunks_df([_chunk(offset=0, data=b"A" * COVERAGE_ROW_WIDTH_BYTES)])
    )
    image, rows = _full_png_pixels(render_coverage_png(result))
    gutter = image.size[0] - COVERAGE_ROW_WIDTH_BYTES
    # The row just above the data is where the ticks bottom out.
    tick_row = rows[coverage_ruler_height_px() - 1]
    # Only the columns above the data: the left gutter's row-0 label is
    # centred on row 0, so its glyphs reach up into this row too.
    inked = {x for x, index in enumerate(tick_row) if index == _INK_PX and x >= gutter}
    assert inked == {
        gutter + offset
        for offset in range(0, COVERAGE_ROW_WIDTH_BYTES, BULK_DOWNLINK_MAX_DATA)
    }


def test_left_ruler_ticks_the_labelled_rows() -> None:
    row_count = COVERAGE_OFFSET_LABEL_INTERVAL_ROWS + 1
    result = reassemble_bulk_chunks(
        _chunks_df([_chunk(offset=0, data=b"A" * COVERAGE_ROW_WIDTH_BYTES * row_count)])
    )
    image, rows = _full_png_pixels(render_coverage_png(result))
    # The pixel column just left of the data is where the ticks end.
    tick_column = image.size[0] - COVERAGE_ROW_WIDTH_BYTES - 1
    ruler = coverage_ruler_height_px()
    inked = {y for y in range(len(rows)) if rows[y][tick_column] == _INK_PX}
    assert inked == {ruler, ruler + COVERAGE_OFFSET_LABEL_INTERVAL_ROWS}


def test_offset_label_interval_widens_for_very_tall_maps() -> None:
    assert coverage_label_interval_rows(1_000) == COVERAGE_OFFSET_LABEL_INTERVAL_ROWS
    # Past the label cap the interval grows in multiples, keeping the count
    # bounded rather than emitting thousands of labels.
    tall = COVERAGE_OFFSET_LABEL_INTERVAL_ROWS * 500 + 1
    interval = coverage_label_interval_rows(tall)
    assert interval == COVERAGE_OFFSET_LABEL_INTERVAL_ROWS * 2
    assert -(-tall // interval) <= 500


def test_rulers_do_not_overwrite_the_data_area() -> None:
    """Ink belongs in the margins only -- a tick bleeding into the bitmap
    would read as a byte with a status it doesn't have.
    """
    result = reassemble_bulk_chunks(
        _chunks_df([_chunk(offset=0, data=b"A" * COVERAGE_ROW_WIDTH_BYTES * 2)])
    )
    _width, _height, rows = _decode_indexed_png(render_coverage_png(result))
    assert all(set(row) == {_GOOD_PX} for row in rows)


def test_byte_segment_length() -> None:
    assert ByteSegment(4, 10, ByteStatus.GOOD).length == 6


# ---------------------------------------------------------------------------
# per-segment copy counts
# ---------------------------------------------------------------------------


def _copies(result: ReassemblyResult) -> list[tuple[int, int]]:
    return [(s.min_copies, s.max_copies) for s in result.segments]


def test_copies_counted_per_segment() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=0, data=b"AAAA"),  # bytes 0..4 arrived twice
            _chunk(offset=4, data=b"BBBB"),  # bytes 4..8 arrived once
        ]
    )
    result = reassemble_bulk_chunks(df)
    # One `Good` run either way -- the copy count varies *within* it, which
    # is exactly what the min/max spread is for.
    assert _segments(result) == [(0, 8, "Good")]
    assert _copies(result) == [(1, 2)]


def test_missing_segment_has_zero_copies() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAA"),
            _chunk(offset=8, data=b"CCCC"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [(0, 4, "Good"), (4, 8, "Missing"), (8, 12, "Good")]
    assert _copies(result) == [(1, 1), (0, 0), (1, 1)]


def test_conflicting_segment_counts_the_disagreeing_copies() -> None:
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"A", received_at="2026-01-01T00:00:00"),
            _chunk(offset=0, data=b"B", received_at="2026-01-01T00:00:01"),
            _chunk(offset=0, data=b"C", received_at="2026-01-01T00:00:02"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [(0, 1, "Conflicting")]
    assert _copies(result) == [(3, 3)]


def test_copies_span_partially_overlapping_chunks() -> None:
    # Bytes 0..4 from one packet, 4..6 from two (an overlap that agrees),
    # 6..8 from one -- all one good run, copies 1-2 across it.
    df = _chunks_df(
        [
            _chunk(offset=0, data=b"AAAAAA"),
            _chunk(offset=4, data=b"AAAA"),
        ]
    )
    result = reassemble_bulk_chunks(df)
    assert _segments(result) == [(0, 8, "Good")]
    assert _copies(result) == [(1, 2)]


def _jpeg_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _bmp_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 200, 30)).save(buffer, format="BMP")
    return buffer.getvalue()


def test_detect_image_finds_a_downlinked_jpeg() -> None:
    data = _jpeg_bytes()
    image = detect_image(data)

    assert image is not None
    assert image.media_type == "image/jpeg"
    assert image.suffix == ".jpg"
    # Downlinked as-is, so the preview serves the reassembled bytes
    # themselves rather than a re-encode of them.
    assert image.data == data
    assert not image.is_converted


def test_detect_image_finds_a_downlinked_bmp() -> None:
    data = _bmp_bytes()
    image = detect_image(data)

    assert image is not None
    assert image.media_type == "image/bmp"
    assert image.suffix == ".bmp"
    assert image.data == data
    assert not image.is_converted


def test_detect_image_still_detects_a_truncated_image() -> None:
    """A download that's missing its tail is exactly when the preview earns
    its keep -- the browser renders as far as the bytes go.
    """
    image = detect_image(_jpeg_bytes((64, 64))[:200])

    assert image is not None
    assert image.media_type == "image/jpeg"


def test_detect_image_ignores_an_image_missing_its_start() -> None:
    """The magic number is what identifies the file, so a download whose
    first bytes haven't arrived (zero-filled by the reassembly) isn't
    claimed as an image rather than previewed as a broken one.
    """
    assert detect_image(b"\x00" * 16 + _jpeg_bytes()) is None


def test_detect_image_ignores_a_non_image() -> None:
    assert detect_image(b"timestamp,voltage_mV\n2026-01-01T00:00:00Z,3300\n") is None
    assert detect_image(b"") is None
