from datetime import UTC, datetime

from cts1_mo_tools.cts1_processing_pipeline.web_ui.file_reassembler_page import (
    _downlinked_byte_range,  # pyright: ignore[reportPrivateUsage]
    _group_consecutive_candidates,  # pyright: ignore[reportPrivateUsage]
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.file_reassembly import (
    BulkHeaderCandidate,
)


def _candidate(
    *,
    offset: int | None = None,
    length: int | None = None,
    file_size: int | None = 1000,
    is_complete: bool = True,
    second: int = 0,
) -> BulkHeaderCandidate:
    return BulkHeaderCandidate(
        received_at=datetime(2026, 11, 17, 0, 0, second, tzinfo=UTC),
        tcmd_sent_at=None,
        action="bulk_downlink_start_v3_blob",
        file="x",
        file_size=file_size,
        sha256=None,
        crc16=None,
        offset=offset,
        length=length,
        adcs_modified_timestamp_utc=None,
        is_complete=is_complete,
        raw_text="",
        properties=(),
    )


def test_partial_range() -> None:
    r = _downlinked_byte_range(_candidate(offset=100, length=50))
    assert r is not None
    assert (r.start, r.end, r.is_whole_file) == (100, 150, False)
    assert r.label() == "100 - 149"


def test_range_covering_the_file_is_whole() -> None:
    r = _downlinked_byte_range(_candidate(offset=0, length=1000))
    assert r is not None
    assert r.is_whole_file


def test_range_past_eof_is_clamped_and_whole() -> None:
    r = _downlinked_byte_range(_candidate(offset=0, length=5000))
    assert r is not None
    assert (r.end, r.is_whole_file) == (1000, True)


def test_range_with_unknown_size_is_never_whole() -> None:
    r = _downlinked_byte_range(_candidate(offset=0, length=5000, file_size=None))
    assert r is not None
    assert not r.is_whole_file


def test_no_offset_or_length_means_whole_file() -> None:
    r = _downlinked_byte_range(_candidate())
    assert r is not None
    assert (r.start, r.end, r.is_whole_file) == (0, 1000, True)


def test_incomplete_header_without_offset_is_unknown() -> None:
    assert _downlinked_byte_range(_candidate(is_complete=False)) is None


def test_only_one_of_offset_and_length_is_unknown() -> None:
    assert _downlinked_byte_range(_candidate(offset=10)) is None


def test_different_chunks_of_one_file_are_separate_rows() -> None:
    groups = _group_consecutive_candidates(
        [
            _candidate(offset=0, length=500, second=0),
            _candidate(offset=0, length=500, second=1),
            _candidate(offset=500, length=500, second=2),
        ]
    )
    assert [len(g.members) for g in groups] == [2, 1]
