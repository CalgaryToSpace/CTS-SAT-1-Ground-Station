from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui import downlinked_files
from PIL import ExifTags, Image

if TYPE_CHECKING:
    from pathlib import Path


def _jpeg(taken: str | None = None, offset: str | None = None) -> bytes:
    exif = Image.Exif()
    tags = exif.get_ifd(ExifTags.IFD.Exif)
    if taken:
        tags[ExifTags.Base.DateTimeOriginal] = taken
    if offset:
        tags[ExifTags.Base.OffsetTimeOriginal] = offset
    out = io.BytesIO()
    Image.new("RGB", (8, 8)).save(out, "JPEG", exif=exif)
    return out.getvalue()


@pytest.mark.parametrize(
    ("order", "descending", "expected"),
    [
        ("Last downlinked", True, "bcdaef"),
        ("Last downlinked", False, "feadcb"),
        ("File name", True, "ebdfca"),
        ("File name", False, "acfdbe"),
        ("File path", True, "edcafb"),
        ("File path", False, "bfacde"),
        # b is 10:00 UTC. Undated go last in both directions, sorted by name.
        ("Capture date", True, "cbedfa"),
        ("Capture date", False, "bcafde"),
    ],
)
def test_gallery_sorts(
    tmp_path: Path, order: str, *, descending: bool, expected: str
) -> None:
    partial = _jpeg("2026:11:22 11:00:00")
    partial = partial[: partial.index(b"\xff\xdb")] + bytes(500)
    files = [
        ("a", "m/1.jpg", 3, _jpeg()),
        ("b", "A/8.jpg", 6, _jpeg("2026:11:22 12:00:00", "+02:00")),
        ("c", "m/2.jpg", 5, partial),
        ("d", "n/4.jpg", 4, _jpeg("not a date")),
        ("e", "z/9.jpg", 2, b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00garbage"),
        ("f", "b/3.bmp", 1, b"BM"),
        ("not_an_image", "log.txt", 7, b"text"),
    ]
    pl.DataFrame(
        {
            "file_key": [key for key, *_ in files],
            "file_path": [path for _, path, *_ in files],
            "last_downlinked_at": [
                datetime(2026, 11, day, tzinfo=UTC) for _, _, day, _ in files
            ],
            "preview_media_type": ["image/jpeg"] * 6 + [None],
            "preview_image": pl.Series([None] * 7, dtype=pl.Binary),
            "data": [data for *_, data in files],
            "segments": ["[]"] * 7,
        }
    ).write_parquet(tmp_path / "downlinked_files.parquet")

    images = downlinked_files.load_gallery_images(tmp_path)
    ordered = downlinked_files.sort_gallery_images(images, order, descending=descending)

    assert "".join(ordered["file_key"]) == expected
