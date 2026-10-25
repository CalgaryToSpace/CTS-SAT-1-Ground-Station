from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.packet_browser import (
    BORING,
    INTERESTING,
    VERY_BORING,
    PacketBrowserFilters,
    load_page,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)

_CASES = [
    ("extended_beacon_blob_v2 success", VERY_BORING),
    ("extended_beacon_blob_v12 success", VERY_BORING),
    ("extended_beacon_blob_v2 success, but then something", INTERESTING),
    ("extended_beacon_blob_v2 failed", INTERESTING),
    (
        "1792303052000+0015417646_E [S:TCMD:INFO]: 🟢 Telecommand "
        "'exec_blob_from_fs' executed. Duration=593ms, err=0",
        VERY_BORING,
    ),
    (
        "1792303052000+0015417646_E [S:TCMD:INFO]: Telecommand "
        "'exec_blob_from_fs' executed. Duration=593ms, err=3",
        INTERESTING,
    ),
    ("Executing telecommand 'hello_world'.", VERY_BORING),
    ("Successfully opened file: /data/x_1.blob", VERY_BORING),
    ("Successfully closed file: x.blob", VERY_BORING),
    ("Successfully opened file: x.txt", INTERESTING),
    ("gnss_bestxyzb_ring_blob_v1 ok, 12 entries", BORING),
    (
        "gnss_bestxyzb_ring_blob_v2: gnss=OFF (eclipse, vbatt=14817mV), "
        "sample=GNSS_POWERED_OFF, cursor=r8/41 (wrapped), stored=3441, "
        "bad_fixes=0, fetch_fails=2, sync=no(61), sent=4/4",
        BORING,
    ),
    ("gnss_bestxyzb_ring_blob_v2: fetch_fails=2, sample=FETCH_FAILED", INTERESTING),
    ("gnss_bestxyzb_ring_blob_v1 FAILED", INTERESTING),
    ("gnss_bestxyzb_ring_blob_v1 err=-2", INTERESTING),
    ("something gnss_bestxyzb_ring_blob_v1", INTERESTING),
    ("hello", INTERESTING),
    (None, INTERESTING),
]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "everything_decoded.parquet"
    pl.DataFrame(
        {
            "received_at": [_BASE + timedelta(seconds=i) for i in range(len(_CASES))],
            "packet_type": ["LOG_MESSAGE"] * len(_CASES),
            "general_message": [message for message, _ in _CASES],
        }
    ).write_parquet(path)
    return path


@pytest.mark.parametrize("reassemble", [False, True])
def test_boringness_classification(tmp_path: Path, *, reassemble: bool) -> None:
    path = _write(tmp_path)
    page = load_page(
        path, PacketBrowserFilters(reassemble_tcmd_responses=reassemble), offset=0
    )
    got = dict(
        zip(
            page.rows["general_message"].to_list(),
            page.rows["boringness"].to_list(),
            strict=True,
        )
    )
    assert got == dict(_CASES)
    # Pinned next to packet_type.
    assert page.columns.index("boringness") == page.columns.index("packet_type") + 1


def test_boringness_filter(tmp_path: Path) -> None:
    path = _write(tmp_path)
    expected = {
        level: sum(lvl == level for _, lvl in _CASES)
        for level in (INTERESTING, BORING, VERY_BORING)
    }

    page = load_page(path, PacketBrowserFilters(boringness=(INTERESTING,)), offset=0)
    assert page.total_rows == expected[INTERESTING]
    assert set(page.rows["boringness"]) == {INTERESTING}

    page = load_page(
        path, PacketBrowserFilters(boringness=(INTERESTING, BORING)), offset=0
    )
    assert page.total_rows == expected[INTERESTING] + expected[BORING]

    assert load_page(path, PacketBrowserFilters(), offset=0).total_rows == len(_CASES)
