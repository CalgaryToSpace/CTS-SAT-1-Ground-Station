import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.file_reassembly import ByteStatus
from cts1_mo_tools.cts1_processing_pipeline.step_10_list_satellite_files.pipeline import (  # noqa: E501
    extract_file_mentions,
    path_sha256_prefix,
    summarize_satellite_files,
)
from cts1_mo_tools.cts1_processing_pipeline.step_11_reassemble_downlinked_files.pipeline import (  # noqa: E501
    Attribution,
    attribute_chunks,
    decode_segments,
    reassemble_downlinked_files,
    reject_inconsistent_downlinks,
    rescue_nearby_chunks,
    summarize_downlinks,
)

_BASE = datetime(2026, 11, 10, tzinfo=UTC)
_CHUNK = 195


def _at(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


def _content(seed: str, size: int) -> bytes:
    """`size` deterministic bytes, different for each `seed`."""
    out = b""
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}{counter}".encode()).digest()
        counter += 1
    return out[:size]


_packet_counter = 0


def _chunks(
    data: bytes, start_seconds: float, *, offsets: list[int] | None = None
) -> list[dict[str, Any]]:
    """`data`'s chunks (or just those at `offsets`), 0.25 s apart."""
    global _packet_counter  # noqa: PLW0603
    rows: list[dict[str, Any]] = []
    for i, offset in enumerate(offsets or list(range(0, len(data), _CHUNK))):
        _packet_counter += 1
        payload = data[offset : offset + _CHUNK]
        rows.append(
            {
                "packet_id": f"p{_packet_counter}",
                "received_at": _at(start_seconds + i * 0.25),
                "bulk_file_offset": offset,
                "bulk_data_len": len(payload),
                "bulk_data_hex": payload.hex(),
            }
        )
    return rows


def _header(path: str, data: bytes, *, offset: int = 0) -> str:
    sha = hashlib.sha256(data).hexdigest()
    return (
        f'{{"action":"bulk_downlink_start_v3","file":"{path}",'
        f'"file_size":{len(data)},"offset":{offset},'
        f'"length":{len(data) - offset},"file_sha256":"{sha}"}}'
    )


def _responses(*items: tuple[float, str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "first_received_at": _at(seconds),
                "tcmd_sent_at": _at(seconds),
                "is_complete": True,
                "tcmd_response_text": text,
            }
            for seconds, text in items
        ],
        schema={
            "first_received_at": pl.Datetime("us", "UTC"),
            "tcmd_sent_at": pl.Datetime("us", "UTC"),
            "is_complete": pl.Boolean,
            "tcmd_response_text": pl.String,
        },
    )


def _beacons(*items: tuple[float, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"received_at": _at(s), "bulk_downlink_file_path_sha256_prefix": p}
            for s, p in items
        ],
        schema={
            "received_at": pl.Datetime("us", "UTC"),
            "bulk_downlink_file_path_sha256_prefix": pl.String,
        },
    )


def _run(
    chunks: list[dict[str, Any]],
    responses: pl.DataFrame,
    beacons: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(attributed chunks, downlinks, downlinked files)."""
    mentions = extract_file_mentions(responses)
    files = summarize_satellite_files(mentions)
    attributed = attribute_chunks(
        pl.DataFrame(chunks),
        beacons if beacons is not None else _beacons(),
        mentions,
        files,
    )
    attributed = attributed.join(
        reject_inconsistent_downlinks(attributed, files),
        on="downlink_index",
        how="left",
    )
    attributed = rescue_nearby_chunks(attributed, files)
    return (
        attributed,
        summarize_downlinks(attributed),
        reassemble_downlinked_files(attributed, files),
    )


def _file(downlinked: pl.DataFrame, key: str) -> dict[str, Any]:
    return downlinked.filter(pl.col("file_key") == key).row(0, named=True)


def test_chunks_after_a_header_are_that_file_and_verify() -> None:
    data = _content("a", 1000)
    attributed, downlinks, downlinked = _run(
        _chunks(data, 5), _responses((0, _header("camera/a.img", data)))
    )
    assert set(attributed["attribution"]) == {Attribution.HEADER}
    assert downlinks.height == 1
    row = _file(downlinked, "camera/a.img")
    assert row["data"] == data
    assert row["is_complete"] is True
    assert row["sha256_verified"] is True
    assert row["downlink_count"] == 1


def test_header_only_vouches_until_its_downlink_should_be_over() -> None:
    data = _content("a", 1000)
    # Expires: 1000 B at 500 B/s, plus the 2 min margin = 122 s.
    in_time = _chunks(data, 100, offsets=[0])
    too_late = _chunks(data, 200, offsets=[195])
    attributed, _, _ = _run(in_time + too_late, _responses((0, _header("c/a", data))))
    by_offset = dict(
        zip(attributed["bulk_file_offset"], attributed["attribution"], strict=True)
    )
    assert by_offset[0] == Attribution.HEADER
    # Not the header's any more -- but near enough to fill the file's gap.
    assert by_offset[195] == Attribution.TIME_WINDOW


def test_header_offset_range_is_respected() -> None:
    data = _content("a", 1000)
    header = _header("c/a", data, offset=390)
    attributed, _, _ = _run(_chunks(data, 5, offsets=[0, 390]), _responses((0, header)))
    by_offset = dict(
        zip(attributed["bulk_file_offset"], attributed["attribution"], strict=True)
    )
    assert by_offset[0] == Attribution.UNATTRIBUTED
    assert by_offset[390] == Attribution.HEADER


def test_beacon_prefix_attributes_headerless_chunks() -> None:
    data = _content("b", 600)
    prefix = path_sha256_prefix("camera/b.img")
    # The file is known from a listing; its header was never received.
    attributed, _, downlinked = _run(
        _chunks(data, 10),
        _responses((0, '{"/camera/./":null,"/camera/b.img":600}')),
        _beacons((8, prefix)),
    )
    assert set(attributed["attribution"]) == {Attribution.BEACON}
    row = _file(downlinked, "camera/b.img")
    assert row["data"] == data
    assert row["is_complete"] is True
    assert row["sha256_verified"] is None  # no SHA-256 ever reported


def test_next_beacon_attributes_a_downlink_that_started_between_beacons() -> None:
    data = _content("b", 400)
    prefix = path_sha256_prefix("b.run")
    attributed, _, _ = _run(
        _chunks(data, 10),
        _responses((0, '{"/./":null,"/b.run":400}')),
        # Nothing downlinking at 5 s; downlinking b.run at 20 s.
        _beacons((5, None), (20, prefix)),
    )
    assert set(attributed["attribution"]) == {Attribution.NEXT_BEACON}
    assert set(attributed["file_path"]) == {"b.run"}


def test_beacon_naming_another_file_ends_a_headers_context() -> None:
    data = _content("a", 1000)
    attributed, _, _ = _run(
        _chunks(data, 30, offsets=[0]),
        _responses((0, _header("c/a", data))),
        _beacons((10, path_sha256_prefix("c/other"))),
    )
    assert attributed["attribution"].to_list() == [Attribution.UNATTRIBUTED]


def _sent(rows: list[dict[str, Any]], received_late_by: float) -> list[dict[str, Any]]:
    """`rows` as sent when they say, but received `received_late_by` s later."""
    return [
        row
        | {
            "sent_at": row["received_at"],
            "received_at": row["received_at"] + timedelta(seconds=received_late_by),
        }
        for row in rows
    ]


def test_chunks_are_placed_by_when_the_satellite_sent_them() -> None:
    a, b = _content("a", 3900), _content("b", 3900)
    # c/a's chunks came from a recording whose times are 30 s late: by
    # `received_at`, they'd all be after c/b's header.
    attributed, _, downlinked = _run(
        _sent(_chunks(a, 1), received_late_by=30) + _sent(_chunks(b, 21), 0),
        _responses((0.5, _header("c/a", a)), (20.5, _header("c/b", b))),
    )
    assert set(attributed["attribution"]) == {Attribution.HEADER}
    assert _file(downlinked, "c/a")["sha256_verified"] is True
    assert _file(downlinked, "c/b")["sha256_verified"] is True


def test_previous_downlinks_tail_after_a_new_header_stays_with_it() -> None:
    a, b = _content("a", 50_000), _content("b", 50_000)
    # c/a's chunk at 23,400 is due 30 s in -- just after c/b's header.
    attributed, _, _ = _run(
        _chunks(a, 1, offsets=[0, 195]) + _chunks(a, 30.5, offsets=[23_400]),
        _responses((0.5, _header("c/a", a)), (30, _header("c/b", b))),
    )
    assert set(attributed["file_path"]) == {"c/a"}
    assert set(attributed["attribution"]) == {Attribution.HEADER}


def test_restarting_without_a_header_is_a_new_downlink() -> None:
    data = _content("a", 2000)
    # Sent again 30 s later, by a telecommand whose header was never heard.
    _, downlinks, downlinked = _run(
        _chunks(data, 1) + _chunks(data, 31),
        _responses((0.5, _header("c/a", data))),
    )
    assert downlinks.height == 2
    assert _file(downlinked, "c/a")["sha256_verified"] is True


def test_unnamed_chunks_are_grouped_but_left_out_of_the_outputs() -> None:
    data = _content("x", 400)
    attributed, downlinks, downlinked = _run(
        _chunks(data, 10) + _chunks(data, 1000) + _chunks(data, 5000),
        _responses(),
        _beacons((9, "abcdef")),
    )
    keys = set(attributed["file_key"])
    assert "sha256-prefix:abcdef" in keys
    unidentified = [k for k in keys if k.startswith("unidentified:")]
    assert len(unidentified) == 2
    assert set(attributed["file_path"].drop_nulls()) == set()
    assert downlinks.is_empty()
    assert downlinked.is_empty()


def test_gap_and_new_header_split_downlinks() -> None:
    # Big enough that its header still vouches past the gap (100 s + 2 min).
    data = _content("a", 50_000)
    header = _header("c/a", data)
    _, downlinks, downlinked = _run(
        _chunks(data, 5, offsets=[0, 195])
        + _chunks(data, 61, offsets=[390])  # new header at 60 s
        + _chunks(data, 61 + 150, offsets=[585]),  # gap over 2 min
        _responses((0, header), (60, header)),
    )
    assert downlinks.height == 3
    row = _file(downlinked, "c/a")
    assert row["downlink_count"] == 3
    # Everything past 780 never arrived.
    assert row["missing_bytes"] == 50_000 - 780
    assert row["byte_length"] == 50_000  # padded out to the known size
    assert decode_segments(row["segments"])[-1].status is ByteStatus.MISSING
    assert row["is_complete"] is False
    assert row["sha256_verified"] is None


def test_downlink_carrying_another_files_content_is_rejected() -> None:
    # Firmware issue #653: after a new header, the previously open file's
    # bytes came down instead.
    real = _content("real", 800)
    other = _content("other", 800)
    _, downlinks, downlinked = _run(
        _chunks(other, 5)  # c/other, verified
        + _chunks(real, 1000)  # c/real
        + _chunks(other, 3000),  # c/real's header, c/other's bytes
        _responses(
            (0, _header("c/other", other)),
            (995, _header("c/real", real)),
            (2995, _header("c/real", real)),
        ),
    )
    row = _file(downlinked, "c/real")
    assert row["data"] == real
    assert row["sha256_verified"] is True
    assert row["rejected_downlink_count"] == 1
    reasons = downlinks.filter(pl.col("file_path") == "c/real")["rejected_reason"]
    assert reasons.drop_nulls().to_list() == ["carries c/other's content"]
    assert reasons.null_count() == 1


def test_without_a_hash_the_bigger_consistent_set_wins() -> None:
    real = _content("real", 800)
    wrong = _content("wrong", 800)
    _, downlinks, downlinked = _run(
        _chunks(real, 10)
        + _chunks(real, 1000, offsets=[0, 195])
        + _chunks(wrong, 2000),
        _responses((0, '{"/./":null,"/f.bin":800}')),
        _beacons(*((s, path_sha256_prefix("f.bin")) for s in (9, 999, 1999))),
    )
    row = _file(downlinked, "f.bin")
    assert row["data"] == real
    assert row["downlink_count"] == 2
    assert row["rejected_downlink_count"] == 1
    rejected = downlinks.filter(pl.col("rejected_reason").is_not_null())
    assert rejected["rejected_reason"].to_list() == [
        "disagrees with the file's other downlinks"
    ]


def test_image_is_detected_for_preview() -> None:
    data = b"\xff\xd8\xff\xe0" + _content("jpg", 400)
    _, _, downlinked = _run(_chunks(data, 5), _responses((0, _header("c/x.jpg", data))))
    row = _file(downlinked, "c/x.jpg")
    assert row["content_kind"] == "image"
    assert row["preview_media_type"] == "image/jpeg"
    assert row["preview_image"] is None  # the data itself is the image


def test_text_file_is_detected() -> None:
    data = b"CTS1+do_something(1,2)!\n" * 20
    _, _, downlinked = _run(_chunks(data, 5), _responses((0, _header("x.run", data))))
    assert _file(downlinked, "x.run")["content_kind"] == "text"


def test_empty_input() -> None:
    assert summarize_downlinks(pl.DataFrame()).is_empty()
    assert reassemble_downlinked_files(pl.DataFrame(), pl.DataFrame()).is_empty()
    assert reject_inconsistent_downlinks(pl.DataFrame(), pl.DataFrame()).is_empty()


def test_picam_image_without_end_line_still_previews() -> None:
    # Two PiCAM sentences (28 bytes each) of a JPEG's start, and no "@FACE"
    # end line.
    jpg = b"\xff\xd8\xff\xdb" + bytes(range(52))
    lines = [
        f"@{i:04X}0002{jpg[i * 28 : (i + 1) * 28].hex().upper()}" for i in range(2)
    ]
    data = ("START_CAM:\n" + "\r\n".join(lines) + "\r\n").encode()
    _, _, downlinked = _run(_chunks(data, 5), _responses((0, _header("c/x.img", data))))
    row = _file(downlinked, "c/x.img")
    assert row["content_kind"] == "image"
    assert row["preview_label"] == "Detected PiCAM image (no end-of-image line)"
    assert row["preview_image"] == jpg


def _attribution_at(attributed: pl.DataFrame, offset: int, seconds: float) -> str:
    return attributed.filter(
        (pl.col("bulk_file_offset") == offset) & (pl.col("received_at") == _at(seconds))
    )["attribution"].item()


def test_chunk_just_past_the_headers_window_fills_the_gap() -> None:
    # 975 B: its header vouches until 975 / 500 + 120 = 122 s.
    data = _content("a", 975)
    attributed, downlinks, downlinked = _run(
        _chunks(data, 5, offsets=[0, 195, 390, 585])
        + _chunks(data, 200, offsets=[780]),
        _responses((0, _header("c/a", data))),
    )
    assert _attribution_at(attributed, 780, 200) == Attribution.TIME_WINDOW
    row = _file(downlinked, "c/a")
    assert row["sha256_verified"] is True
    assert row["rescued_chunk_count"] == 1
    assert downlinks.height == 2  # the rescued chunk is a downlink of its own


def test_rescue_is_all_or_nothing_per_downlink_and_never_collides() -> None:
    data = _content("a", 975)
    wrong = bytearray(data)
    wrong[585:780] = _content("other", 195)
    late = _chunks(bytes(wrong), 200, offsets=[585, 780])  # one downlink
    attributed, _, downlinked = _run(
        _chunks(data, 5, offsets=[0, 195, 390, 585]) + late,
        _responses((0, _header("c/a", data))),
    )
    assert set(attributed.filter(pl.col("received_at") >= _at(200))["attribution"]) == {
        Attribution.UNATTRIBUTED
    }
    row = _file(downlinked, "c/a")
    assert row["rescued_chunk_count"] == 0
    assert row["conflict_bytes"] == 0
    assert row["missing_bytes"] == 975 - 780


def test_chunk_from_before_a_header_needs_the_sha256_to_be_taken() -> None:
    # Received before c/a's header: whatever was downlinking then.
    for size, taken in ((975, True), (1200, False)):
        data = _content("a", size)
        attributed, _, downlinked = _run(
            _chunks(data, 30, offsets=[780])
            + _chunks(data, 61, offsets=[0, 195, 390, 585]),
            _responses((60, _header("c/a", data))),
        )
        expected = Attribution.TIME_WINDOW if taken else Attribution.UNATTRIBUTED
        assert _attribution_at(attributed, 780, 30) == expected, size
        assert _file(downlinked, "c/a")["sha256_verified"] is (True if taken else None)


def test_chunk_near_two_files_is_left_alone() -> None:
    a = _content("a", 975)
    b = _content("b", 390)
    attributed, _, downlinked = _run(
        _chunks(a, 5, offsets=[0, 195, 390, 585])
        + _chunks(b, 105)
        # c/a's missing last chunk -- but c/b's downlink is just as near.
        + _chunks(a, 150, offsets=[780]),
        _responses((0, _header("c/a", a)), (100, _header("c/b", b))),
    )
    assert _attribution_at(attributed, 780, 150) == Attribution.UNATTRIBUTED
    assert _file(downlinked, "c/a")["rescued_chunk_count"] == 0


def test_known_but_never_downlinked_file_gets_a_row_without_bytes() -> None:
    sent = _content("sent", 400)
    unsent = _content("unsent", 300)
    _, _, downlinked = _run(
        _chunks(sent, 5),
        _responses((0, _header("a.bin", sent)), (3600, _header("b.bin", unsent))),
    )
    assert set(downlinked["file_key"]) == {"a.bin", "b.bin"}
    got, never = _file(downlinked, "a.bin"), _file(downlinked, "b.bin")
    assert got["sha256_verified"] is True
    assert got["mention_count"] == 1
    assert got["listing_status"] is not None
    assert never["data"] is None
    assert never["downlink_count"] == 0
    assert never["file_size"] == len(unsent)
    assert never["reported_sha256"] == hashlib.sha256(unsent).hexdigest()
    assert never["file_path"] == "b.bin"


def test_file_timestamp_is_from_the_file_or_the_adcs_header() -> None:
    meta = b'START_CAM:{"blob":"camera_capture_v2","epoch_ms":1795330803433}\n'
    data = meta + _content("image", 300)
    adcs = (
        '{"action":"adcs_get_latest_sd_file_blob_v2","file":"ADCS/log_1.TLM",'
        '"file_size":10,"datetime":"2026-11-21 22:27:36"}'
    )
    # A later response without the date doesn't lose it.
    adcs_v1 = (
        '{"action":"adcs_get_latest_sd_file_blob","file":"ADCS/log_1.TLM",'
        '"file_size":10}'
    )
    other = '{"action":"analyze_x","file":"b","datetime":"2026-11-21 22:27:36"}'
    _, _, downlinked = _run(
        _chunks(data, 5),
        _responses(
            (-900, adcs), (-800, adcs_v1), (-700, other), (0, _header("c/x.img", data))
        ),
    )

    image = _file(downlinked, "c/x.img")
    assert image["file_timestamp"] == datetime(
        2026, 11, 22, 7, 0, 3, 433000, tzinfo=UTC
    )
    assert image["file_metadata"] == meta[len("START_CAM:") : -1].decode()
    assert _file(downlinked, "ADCS/log_1.TLM")["file_metadata"] is None
    # Never downlinked, but the header alone dates it.
    assert _file(downlinked, "ADCS/log_1.TLM")["file_timestamp"] == datetime(
        2026, 11, 21, 22, 27, 36, tzinfo=UTC
    )
    assert _file(downlinked, "b")["file_timestamp"] is None


def test_mpi_file_timestamp_is_from_its_header_or_the_analysis() -> None:
    header = b'{"mpi_start":1,"timestamp_ms":1794645210166}'
    with_header = header + _content("with header", 300)
    headerless = _content("first packet lost", 600)

    def analysis(path: str) -> str:
        return (
            f'{{"action":"analyze_mpi_data_v2","file":"{path}","size":600,'
            '"earliest":{"timestamp_ms":1793988090847},'
            '"latest":{"timestamp_ms":1793988190847}}'
        )

    _, _, downlinked = _run(
        _chunks(with_header, 5) + _chunks(headerless, 905, offsets=[195, 390]),
        _responses(
            (-900, analysis("m/never.mpi")),
            (-800, analysis("m/headerless.mpi")),
            (0, _header("m/header_only.mpi", with_header)),
            (900, _header("m/headerless.mpi", headerless)),
        ),
    )

    analyzed = datetime(2026, 11, 6, 18, 1, 30, 847000, tzinfo=UTC)
    assert _file(downlinked, "m/header_only.mpi")["file_timestamp"] == datetime(
        2026, 11, 14, 8, 33, 30, 166000, tzinfo=UTC
    )
    assert _file(downlinked, "m/headerless.mpi")["file_timestamp"] == analyzed
    assert _file(downlinked, "m/never.mpi")["file_timestamp"] == analyzed
