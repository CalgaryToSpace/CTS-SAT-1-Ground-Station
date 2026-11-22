from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui import downlinked_files

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("order", "descending", "expected"),
    [
        ("Last downlinked", True, "bcdaef"),
        ("Last downlinked", False, "feadcb"),
        ("File name", True, "ebdfca"),
        ("File name", False, "acfdbe"),
        ("File path", True, "edcafb"),
        ("File path", False, "bfacde"),
        # Undated go last in both directions, sorted by name.
        ("File timestamp", True, "cbedfa"),
        ("File timestamp", False, "bcafde"),
    ],
)
def test_gallery_sorts(
    tmp_path: Path, order: str, *, descending: bool, expected: str
) -> None:
    files = [
        ("a", "m/1.jpg", 3, None),
        ("b", "A/8.jpg", 6, 10),
        ("c", "m/2.jpg", 5, 11),
        ("d", "n/4.jpg", 4, None),
        ("e", "z/9.jpg", 2, None),
        ("f", "b/3.bmp", 1, None),
        ("not_an_image", "log.txt", 7, 12),
    ]
    pl.DataFrame(
        {
            "file_key": [key for key, *_ in files],
            "file_path": [path for _, path, *_ in files],
            "last_downlinked_at": [
                datetime(2026, 11, day, tzinfo=UTC) for _, _, day, _ in files
            ],
            "file_timestamp": [
                datetime(2026, 11, 22, hour, tzinfo=UTC) if hour else None
                for *_, hour in files
            ],
            "preview_media_type": ["image/jpeg"] * 6 + [None],
            "preview_image": pl.Series([None] * 7, dtype=pl.Binary),
            "data": [b""] * 7,
            "segments": ["[]"] * 7,
        }
    ).write_parquet(tmp_path / "downlinked_files.parquet")

    images = downlinked_files.load_gallery_images(tmp_path)
    ordered = downlinked_files.sort_gallery_images(images, order, descending=descending)

    assert "".join(ordered["file_key"]) == expected


def test_gallery_reads_files_written_before_file_timestamp(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "file_key": ["a"],
            "file_path": ["a.jpg"],
            "preview_media_type": ["image/jpeg"],
            "preview_image": pl.Series([None], dtype=pl.Binary),
            "data": [b""],
            "segments": ["[]"],
        }
    ).write_parquet(tmp_path / "downlinked_files.parquet")

    images = downlinked_files.load_gallery_images(tmp_path)
    ordered = downlinked_files.sort_gallery_images(
        images, "File timestamp", descending=True
    )

    assert ordered["file_timestamp"].to_list() == [None]
