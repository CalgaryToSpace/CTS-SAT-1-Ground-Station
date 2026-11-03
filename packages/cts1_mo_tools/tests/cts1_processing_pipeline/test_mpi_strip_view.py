from datetime import UTC, datetime

import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.mpi_strip_view import (
    _next_zoom,  # pyright: ignore[reportPrivateUsage]
    collection_key,
)


def test_collection_key_is_millisecond_iso_utc() -> None:
    started_at = datetime(2026, 7, 21, 17, 23, 0, 208_000, tzinfo=UTC)
    assert collection_key(started_at) == "2026-07-21T17:23:00.208Z"
    # Zero-padded, and sub-millisecond digits dropped.
    started_at = datetime(2026, 7, 21, 17, 23, 0, 5_999, tzinfo=UTC)
    assert collection_key(started_at) == "2026-07-21T17:23:00.005Z"


@pytest.mark.parametrize(
    ("current", "direction", "expected"),
    [
        # A long collection: fit-width (0.5 px/frame) is below 1x.
        ("fit", 1, 1),
        (1, -1, "fit"),
        (2, 1, 4),
        (4, -1, 2),
        (8, 1, None),
        ("fit", -1, None),
    ],
)
def test_next_zoom_long_collection(
    current: int | str, direction: int, expected: int | str | None
) -> None:
    got = _next_zoom(current, direction=direction, client_width=1000, column_count=2000)
    assert got == expected


@pytest.mark.parametrize(
    ("current", "direction", "expected"),
    [
        # A short collection: fit-width (3.33 px/frame) sits between 2x and 4x.
        ("fit", 1, 4),
        ("fit", -1, 2),
        (2, 1, "fit"),
        (4, -1, "fit"),
        (1, -1, None),
    ],
)
def test_next_zoom_short_collection(
    current: int | str, direction: int, expected: int | str | None
) -> None:
    got = _next_zoom(current, direction=direction, client_width=1000, column_count=300)
    assert got == expected
