from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_10_list_satellite_files.pipeline import (  # noqa: E501
    ListingStatus,
    MentionSource,
    extract_file_mentions,
    normalize_path,
    path_sha256_prefix,
    summarize_satellite_files,
)

_BASE = datetime(2026, 11, 1, tzinfo=UTC)


def _responses(*texts: str | tuple[str, bool]) -> pl.DataFrame:
    """Step 5-shaped responses, one minute apart, oldest first. A
    `(text, False)` is an incomplete response.
    """
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(texts):
        text, complete = item if isinstance(item, tuple) else (item, True)
        at = _BASE + timedelta(minutes=i)
        rows.append(
            {
                "first_received_at": at,
                "tcmd_sent_at": at,
                "is_complete": complete,
                "tcmd_response_text": text,
            }
        )
    return pl.DataFrame(rows)


def _files(*texts: str | tuple[str, bool]) -> pl.DataFrame:
    return summarize_satellite_files(extract_file_mentions(_responses(*texts)))


def _status(files: pl.DataFrame, path: str) -> str:
    return files.filter(pl.col("file_path") == path)["listing_status"].item()


def test_normalize_path() -> None:
    assert normalize_path("/camera/x.img") == "camera/x.img"
    assert normalize_path("camera//x.img") == "camera/x.img"
    assert normalize_path("/./camera/./x.img") == "camera/x.img"
    assert normalize_path("/ADCS/../camera/x.img") == "camera/x.img"
    assert normalize_path("/") is None
    assert normalize_path("/mpi_da???04Z.mpi") is None


def test_path_sha256_prefix_matches_the_beacon() -> None:
    # A real pair: the v5+ beacons reported 797031 during this file's downlink.
    assert path_sha256_prefix("camera/2026-11-13_010000Z.img") == "797031"


def test_empty_input_has_the_schema() -> None:
    files = _files()
    assert files.is_empty()
    assert "listing_status" in files.columns


def test_listing_entries_become_files_and_dirs_are_skipped() -> None:
    mentions = extract_file_mentions(
        _responses(
            '{"/camera/./":null,"/camera/../":null,"/camera/a.img":10,'
            '"/camera/b.run":20,"/camera/2026-11/":null}'
        )
    )
    assert mentions["file_path"].to_list() == ["camera/a.img", "camera/b.run"]
    assert mentions["file_size"].to_list() == [10, 20]
    assert set(mentions["listing_directory"]) == {"camera"}
    assert mentions["raw_path"].to_list() == ["/camera/a.img", "/camera/b.run"]


def test_root_listing_directory_is_empty_string() -> None:
    mentions = extract_file_mentions(
        _responses('{"/./":null,"/../":null,"/x.run":5,"/camera/":null}')
    )
    assert mentions["file_path"].to_list() == ["x.run"]
    assert mentions["listing_directory"].to_list() == [""]


def test_action_responses_give_size_and_whole_file_sha256() -> None:
    sha = "AB" * 32
    mentions = extract_file_mentions(
        _responses(
            '{"action":"bulk_downlink_start_v3","file":"camera/x.img",'
            f'"file_size":500,"offset":100,"length":200,"file_sha256":"{sha}",'
            '"file_crc16":"0x1234"}',
            '{"action":"get_file_map_v1","file":"blobs/b.blob","sha256":"'
            + "cd" * 32
            + '","size":7}',
            '{"action":"analyze_mpi_data_v2","file":"mpi_data/m.mpi","size":9}',
            # chunk_sha256 describes only part of the file.
            '{"action":"bulk_downlink_start_v3","file":"x.run","file_size":9,'
            '"chunk_sha256":"' + "ef" * 32 + '"}',
        )
    )
    rows = {row["file_path"]: row for row in mentions.to_dicts()}
    header = rows["camera/x.img"]
    assert header["source"] == MentionSource.BULK_DOWNLINK_HEADER
    assert header["file_size"] == 500
    assert header["sha256"] == sha.lower()
    assert (header["downlink_offset"], header["downlink_length"]) == (100, 200)
    assert rows["blobs/b.blob"]["source"] == MentionSource.FILE_MAP
    assert rows["blobs/b.blob"]["file_size"] == 7
    assert rows["mpi_data/m.mpi"]["source"] == MentionSource.MPI_ANALYSIS
    assert rows["mpi_data/m.mpi"]["downlink_offset"] is None
    assert rows["x.run"]["sha256"] is None
    # No `length`: the rest of the file from `offset`.
    assert rows["x.run"]["downlink_length"] == 9


def test_garbled_response_still_gives_up_what_survived() -> None:
    mentions = extract_file_mentions(
        _responses(
            (
                '{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/log_1.TLM",'
                '"file_size":77,"sha256":"60130c2c????????',
                False,
            ),
            ('{"/camera/./":null,"/camera/a.img":1,"/cam??????b.img":2}', False),
        )
    )
    assert mentions["file_path"].to_list() == ["ADCS/log_1.TLM", "camera/a.img"]
    adcs = mentions.row(0, named=True)
    assert adcs["source"] == MentionSource.ADCS_FILE_HEADER
    assert adcs["file_size"] == 77
    assert adcs["sha256"] is None  # truncated
    assert mentions.row(1, named=True)["listing_complete"] is False


def test_not_a_listing_or_file_response_is_ignored() -> None:
    mentions = extract_file_mentions(
        _responses(
            '{"fs_used_size_bytes":61210624}',
            "LittleFS List Directory Error: -1",
            '{"action":"something","no_file":1}',
        )
    )
    assert mentions.is_empty()


def test_summary_takes_latest_size_and_sha256_and_all_sources() -> None:
    files = _files(
        '{"/c/./":null,"/c/x.img":10}',
        '{"action":"bulk_downlink_start_blob","file":"c/x.img","file_size":12,'
        '"sha256":"' + "aa" * 32 + '"}',
    )
    row = files.row(0, named=True)
    assert row["file_size"] == 12
    assert row["file_sizes_seen"] == "[10,12]"
    assert row["sha256"] == "aa" * 32
    assert row["sources"] == '["bulk_downlink_header","listing"]'
    assert row["mention_count"] == 2
    assert row["path_sha256_prefix"] == path_sha256_prefix("c/x.img")


def test_prefixes_include_the_raw_form_a_header_used() -> None:
    files = _files(
        '{"action":"bulk_downlink_start_v3","file":"/c/x.img","file_size":1}'
    )
    row = files.row(0, named=True)
    assert row["file_path"] == "c/x.img"
    assert row["path_sha256_prefixes"] == (
        f'["{path_sha256_prefix("c/x.img")}","{path_sha256_prefix("/c/x.img")}"]'
    )


def test_listing_status_deleted_and_created_since() -> None:
    files = _files(
        '{"/c/./":null,"/c/../":null,"/c/a":1,"/c/b":2,"/c/c":3}',
        '{"/c/./":null,"/c/../":null,"/c/a":1,"/c/c":3}',
        '{"action":"bulk_downlink_start_v3","file":"c/bb","file_size":3}',
    )
    assert _status(files, "c/a") == ListingStatus.LISTED
    assert _status(files, "c/b") == ListingStatus.NOT_IN_LATEST_LISTING
    assert _status(files, "c/bb") == ListingStatus.NEWER_THAN_LATEST_LISTING


def test_count_limited_page_only_covers_names_up_to_its_last_entry() -> None:
    # As seen in flight: a full listing, then one `count`-limited to its
    # first entries. Names after the page's last entry aren't judged by it
    # -- so deleting the last-sorted files can't be told from a short page,
    # and is only noticed once a later listing reaches past them.
    files = _files(
        '{"/d/./":null,"/d/../":null,"/d/a":1,"/d/b":1,"/d/c":1}',
        '{"/d/./":null,"/d/../":null,"/d/a":1}',
    )
    for path in ("d/a", "d/b", "d/c"):
        assert _status(files, path) == ListingStatus.LISTED
    assert files.filter(pl.col("file_path") == "d/c")["last_listed_at"].item() == _BASE
    assert files.filter(pl.col("file_path") == "d/a")[
        "last_listed_at"
    ].item() == _BASE + timedelta(minutes=1)


def test_offset_page_only_covers_names_from_its_first_entry() -> None:
    # A page past offset 0 has no "./" entry: its directory comes from its
    # entries, and it says nothing about names before its first.
    files = _files(
        '{"/d/./":null,"/d/../":null,"/d/a":1,"/d/m":1}',
        '{"/d/m":1,"/d/n":1}',
    )
    assert _status(files, "d/a") == ListingStatus.LISTED
    assert _status(files, "d/n") == ListingStatus.LISTED


def test_incomplete_listing_does_not_count() -> None:
    files = _files(
        '{"/d/./":null,"/d/a":1,"/d/b":1}',
        ('{"/d/./":null,"/d/a":1}', False),
    )
    assert _status(files, "d/b") == ListingStatus.LISTED


def test_never_listed_directory() -> None:
    files = _files('{"action":"get_file_map_v1","file":"blobs/x.blob","size":1}')
    assert _status(files, "blobs/x.blob") == ListingStatus.NEVER_LISTED
