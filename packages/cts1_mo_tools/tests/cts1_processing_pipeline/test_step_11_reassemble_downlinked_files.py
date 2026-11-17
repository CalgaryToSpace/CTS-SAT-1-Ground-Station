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
    assert by_offset[195] == Attribution.UNATTRIBUTED


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


def test_null_beacon_after_header_ends_its_context() -> None:
    data = _content("a", 1000)
    attributed, _, _ = _run(
        _chunks(data, 30, offsets=[0]),
        _responses((0, _header("c/a", data))),
        _beacons((10, None)),
    )
    assert attributed["attribution"].to_list() == [Attribution.UNATTRIBUTED]


def test_unknown_prefix_is_its_own_file_and_unattributed_downlinks_each_theirs() -> (
    None
):
    data = _content("x", 400)
    attributed, downlinks, downlinked = _run(
        _chunks(data, 10) + _chunks(data, 1000) + _chunks(data, 5000),
        _responses(),
        _beacons((9, "abcdef")),
    )
    keys = set(downlinked["file_key"])
    assert "sha256-prefix:abcdef" in keys
    unidentified = [k for k in keys if k.startswith("unidentified:")]
    assert len(unidentified) == 2
    assert downlinks.height == 3
    assert set(attributed["file_path"].drop_nulls()) == set()


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
