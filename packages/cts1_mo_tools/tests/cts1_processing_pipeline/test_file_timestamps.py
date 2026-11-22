from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest
from cts1_mo_tools.cts1_processing_pipeline.file_timestamps import (
    parse_adcs_modified_time,
    parse_capture_time,
    parse_mpi_analysis_start_time,
)
from PIL import ExifTags, Image

_META = (
    b'{"blob":"camera_capture_v2","lighting":"d","epoch_ms":1795330803433,"adcs":{}}'
)
_MPI_HEADER = (
    b'{"mpi_start":1,"uptime_ms":95274207,"timestamp":"1794642427000+3551356693_E",'
    b'"datetime":"2026-11-14T083330.166Z_E","timestamp_ms":1794645210166}'
)
_MPI_STARTED = datetime(2026, 11, 14, 8, 33, 30, 166000, tzinfo=UTC)
_CAPTURED = datetime(2026, 11, 22, 7, 0, 3, 433000, tzinfo=UTC)


def _jpeg(
    taken: str | None = None, offset: str | None = None, description: str = ""
) -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.ImageDescription] = description
    tags = exif.get_ifd(ExifTags.IFD.Exif)
    if taken:
        tags[ExifTags.Base.DateTimeOriginal] = taken
    if offset:
        tags[ExifTags.Base.OffsetTimeOriginal] = offset
    out = io.BytesIO()
    Image.new("RGB", (8, 8)).save(out, "JPEG", exif=exif)
    return out.getvalue()


def _partial(jpeg: bytes) -> bytes:
    return jpeg[: jpeg.index(b"\xff\xdb")] + bytes(500)


@pytest.mark.parametrize(
    ("image", "expected_hour"),
    [
        (_jpeg("2026:11:22 12:00:00", "+02:00"), 10),
        (_partial(_jpeg("2026:11:22 11:00:00")), 11),
        (_jpeg(), None),
        (_jpeg("not a date"), None),
        (_jpeg("2026:11:22 11:00:0"), None),  # cut short
        (_jpeg("1970:01:01 00:20:00"), None),  # unset clock
        (b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00garbage", None),
        (b"BM", None),
    ],
)
def test_exif_capture_time(image: bytes, expected_hour: int | None) -> None:
    expected = expected_hour and datetime(2026, 11, 22, expected_hour, tzinfo=UTC)
    assert parse_capture_time("camera/a.jpg", image, image) == expected


@pytest.mark.parametrize(
    ("path", "data", "expected"),
    [
        ("camera/a.img", b"START_CAM:" + _META + b"\n@0000", _CAPTURED),
        # The first packet never arrived, so there's no EXIF header to read.
        ("camera/a.jpg", bytes(195) + b"+00:00\x00" + _META, _CAPTURED),
        # Digits lost to a missing packet.
        ("camera/a.img", b"START_CAM:" + _META[:60] + bytes(50), None),
        # A .run log quotes the capture's response, metadata and all.
        ("camera.run", b'{"meta":' + _META + b"}", None),
        ("camera/a.img", b"START_CAM:\n@0000", None),  # before camera_capture_v2
    ],
)
def test_metadata_capture_time(
    path: str, data: bytes, expected: datetime | None
) -> None:
    assert parse_capture_time(path, data, None) == expected


def test_metadata_is_preferred_to_exif_for_its_milliseconds() -> None:
    image = _jpeg("2026:11:22 07:00:03", "+00:00", _META.decode())
    assert parse_capture_time("camera/a.jpg", image, image) == _CAPTURED


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-29 08:05:08", datetime(2026, 9, 29, 8, 5, 8, tzinfo=UTC)),
        ("1980-00-00 00:00:00", None),
        ("1980-01-01 00:00:00", None),
        ("2026-09-2????????", None),
        (None, None),
    ],
)
def test_adcs_modified_time(value: object, expected: datetime | None) -> None:
    assert parse_adcs_modified_time(value) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_MPI_HEADER + b"\x0c\xff\xff\x0c", _MPI_STARTED),
        (_MPI_HEADER[:-8] + bytes(8), None),  # digits lost to a missing packet
        # A later recording's header isn't when the file started.
        (bytes(195) + _MPI_HEADER, None),
        # Never synced.
        (b'{"mpi_start":1,"timestamp_ms":0000000042000}', None),
    ],
)
def test_mpi_header_start_time(data: bytes, expected: datetime | None) -> None:
    assert parse_capture_time("mpi_data/a.mpi", data, None) == expected


@pytest.mark.parametrize(
    ("earliest", "expected"),
    [
        ({"timestamp_ms": 1794645210166}, _MPI_STARTED),
        ({"timestamp_ms": 42000}, None),  # never synced
        ({"timestamp_ms": 99999999999999999999}, None),
        ({"timestamp_ms": "1794645210166"}, None),
        ({}, None),
        (None, None),  # no time syncs in the file
    ],
)
def test_mpi_analysis_start_time(earliest: object, expected: datetime | None) -> None:
    assert parse_mpi_analysis_start_time(earliest) == expected
