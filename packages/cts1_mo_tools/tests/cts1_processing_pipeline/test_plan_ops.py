import io
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import openpyxl
import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_12_fetch_satnogs_schedule import (
    pipeline as step_12_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui import plan_ops
from cts1_mo_tools.cts1_processing_pipeline.web_ui.plan_ops import (
    PlanOpsStore,
    build_pass_agenda,
    check_login,
    dark_region_photo_rows,
    downlink_ranges,
    missing_chunk_targets,
    parse_stored_time,
    satnogs_downlink_rows,
    table_rows,
    to_tsv,
    to_xlsx,
)

from .test_overpasses import TLE, TLE_EPOCH

AOS = datetime(2026, 10, 25, 17, 10, tzinfo=UTC)


def test_store_round_trip(tmp_path: Path) -> None:
    store = PlanOpsStore(tmp_path / "ops.sqlite")
    rev = store.revision()
    store.add([{"telecommand": "CTS1+hello_world()!"}, {}], "alice")
    assert store.revision() == rev + 1
    a, b = store.list_commands()
    assert (a["uplink_aos"], a["repeat"], a["random"]) == (None, "1", "5")

    store.update_field(a["id"], "notes", "hi", "bob")
    store.move([a["id"]], "2026-10-25 17:10:00", "bob")
    store.delete([b["id"]])
    (a,) = PlanOpsStore(tmp_path / "ops.sqlite").list_commands()
    assert (a["notes"], a["uplink_aos"], a["updated_by"]) == (
        "hi",
        "2026-10-25 17:10:00",
        "bob",
    )
    with pytest.raises(ValueError, match="editable"):
        store.update_field(a["id"], "id = 0; --", "x", "eve")


def test_check_login(tmp_path: Path) -> None:
    users = tmp_path / "users.json"
    assert not check_login(users, "alice", "")
    users.write_text(json.dumps({"alice": "s3cret"}))
    assert check_login(users, "alice", "s3cret")
    assert not check_login(users, "alice", "wrong")
    assert not check_login(users, "bob", "s3cret")


def test_build_pass_agenda() -> None:
    commands = [
        dict(plan_ops.FIELD_DEFAULTS, telecommand="CTS1+a()!", random="0"),
        dict(
            plan_ops.FIELD_DEFAULTS,
            mode="Interval",
            tsexec_start="2026-10-26 00:50:00",  # past midnight UTC
            tsexec_end="2026-10-26 00:54:00",
            interval="2 min",
            telecommand="CTS1+b()!",
        ),
    ]
    lines = [
        line
        for line in build_pass_agenda(commands, AOS).splitlines()
        if line.startswith("CTS1+")
    ]
    exec_ms = int(datetime(2026, 10, 26, 0, 50, tzinfo=UTC).timestamp() * 1000)
    assert lines[0].startswith("CTS1+a()@tssent=")
    for i, line in enumerate(lines[1:]):
        assert f"@tsexec={exec_ms + i * 120_000}!" in line
    assert len(lines) == 4

    commands[0]["tsexec_start"] = "2026-10-27 00:00:00"  # > 24 h after AOS
    with pytest.raises(ValueError, match="24 h"):
        build_pass_agenda(commands, AOS)


def test_downlink_ranges() -> None:
    size = 100_000
    # Small gaps close together share one >= 30 KB downlink.
    assert downlink_ranges([(0, 10), (20, 40)], size) == [(0, 30_000)]
    # A big gap stays whole; one near the end is pulled back inside the file.
    assert downlink_ranges([(1000, 50_000), (99_990, 100_000)], size) == [
        (1000, 49_000),
        (70_000, 30_000),
    ]
    # A gap starting inside the previous downlink extends it.
    assert downlink_ranges([(0, 10), (29_999, 40_000)], size) == [(0, 40_000)]
    # A file under 30 KB goes down whole.
    assert downlink_ranges([(5, 6)], 1000) == [(0, 1000)]


def test_missing_chunk_targets() -> None:
    files = pl.DataFrame(
        {
            "file_path": ["a.img", "b.img"],
            "byte_length": [100_000, 10],
            "missing_bytes": [30, 0],
            "segments": [
                '[[0,10,"M",0,0],[10,20,"G",1,1],[20,40,"M",0,0]]',
                '[[0,10,"G",1,1]]',
            ],
        }
    )
    assert missing_chunk_targets(files) == ["a.img;0;30000"]


def test_satnogs_downlink_rows_round_robin() -> None:
    windows = [
        (AOS + timedelta(hours=i), AOS + timedelta(hours=i, minutes=9), "")
        for i in range(3)
    ]
    rows = satnogs_downlink_rows(windows, ["x;0;30000", "y;0;30000"], "2 min")
    assert [r["telecommand"] for r in rows] == [
        "CTS1+exec_blob_from_fs(blobs/bulk_downlink_start_v3.blob,0,x;0;30000)",
        "CTS1+exec_blob_from_fs(blobs/bulk_downlink_start_v3.blob,0,y;0;30000)",
        "CTS1+exec_blob_from_fs(blobs/bulk_downlink_start_v3.blob,0,x;0;30000)",
    ]
    assert satnogs_downlink_rows(windows, [], "2 min") == []


def test_load_satnogs_windows(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="step 12"):
        plan_ops.load_satnogs_windows(tmp_path, AOS, AOS + timedelta(days=1))
    pl.DataFrame(
        {
            "observation_id": [1, 2, 3],
            "ground_station_id": [10, 11, 12],
            "start": [AOS, AOS + timedelta(minutes=5), AOS + timedelta(hours=2)],
            "end": [
                AOS + timedelta(minutes=10),
                AOS + timedelta(minutes=12),
                AOS + timedelta(hours=2, minutes=10),
            ],
            "fetched_at": [AOS] * 3,
        },
        schema=step_12_pipeline.SCHEDULE_SCHEMA,
    ).write_parquet(tmp_path / step_12_pipeline.OUTPUT_FILENAME)
    windows = plan_ops.load_satnogs_windows(
        tmp_path, AOS + timedelta(minutes=1), AOS + timedelta(hours=2, minutes=5)
    )
    assert windows == [
        (
            AOS + timedelta(minutes=1),
            AOS + timedelta(minutes=12),
            "2 station(s)",
        ),
        (
            AOS + timedelta(hours=2),
            AOS + timedelta(hours=2, minutes=5),
            "1 station(s)",
        ),
    ]


def test_dark_region_photo_rows() -> None:
    start = TLE_EPOCH
    rows = dark_region_photo_rows(
        TLE.to_satkit(),
        start,
        start + timedelta(hours=6),
        latitude_range="55 to 65",
        longitude_range=None,
        every_sec=30,
    )
    assert rows
    # A setup 30 s before every single capture.
    for setup, capture in zip(rows[::2], rows[1::2], strict=True):
        assert setup["telecommand"] == "CTS1+camera_setup()"
        assert re.fullmatch(
            r"CTS1\+exec_blob_from_fs\(blobs/camera_capture_v2\.blob,0,"
            r"camera/\d{4}-\d{2}/\d{4}-\d{2}-\d{2}_\d{6}Z\.img;d\)",
            capture["telecommand"],
        )
        assert parse_stored_time(capture["tsexec_start"]) - parse_stored_time(
            setup["tsexec_start"]
        ) == timedelta(seconds=30)


def test_table_exports() -> None:
    rows = table_rows([{"uplink": "p", "telecommand": "CTS1+a(x;y)", "notes": "a\tb"}])
    assert rows[0][0] == "Uplink pass"
    tsv = to_tsv(rows)
    assert tsv.splitlines()[1].startswith("p\t\t")
    assert '"a\tb"' in tsv  # Quoted, so a tab in a cell doesn't split it.
    sheet = openpyxl.load_workbook(io.BytesIO(to_xlsx(rows))).active
    assert sheet is not None
    assert sheet["F2"].value == "CTS1+a(x;y)"
    assert sheet["F2"].font.name == "Consolas"
    assert sheet["A1"].font.bold


def test_step_12_writes_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations: list[dict[str, object]] = [
        {
            "id": 7,
            "ground_station": 3,
            "start": "2026-10-25T17:10:00Z",
            "end": "2026-10-25T17:20:00Z",
        }
    ]

    def fetch(*_a: object, **_k: object) -> list[dict[str, object]]:
        return observations

    monkeypatch.setattr(step_12_pipeline, "fetch_all_observations", fetch)
    step_12_pipeline.run(data_dir=tmp_path)
    df = pl.read_parquet(tmp_path / step_12_pipeline.OUTPUT_FILENAME)
    assert df["start"].to_list() == [AOS]

    # A failed refetch keeps the old file.
    def fail(*_a: object, **_k: object) -> None:
        raise OSError

    monkeypatch.setattr(step_12_pipeline, "fetch_all_observations", fail)
    monkeypatch.setattr(step_12_pipeline, "REFRESH_INTERVAL", timedelta(0))
    step_12_pipeline.run(data_dir=tmp_path)
    assert pl.read_parquet(tmp_path / step_12_pipeline.OUTPUT_FILENAME).height == 1


def test_load_incomplete_files(tmp_path: Path) -> None:
    assert plan_ops.load_incomplete_files(tmp_path).is_empty()
    pl.DataFrame(
        {
            "file_path": ["a", "b"],
            "byte_length": [1, 1],
            "missing_bytes": [1, 0],
            "segments": ['[[0,1,"M",0,0]]', "[]"],
            "last_downlinked_at": [AOS, AOS],
            "data": [b"", b""],
        }
    ).write_parquet(tmp_path / "downlinked_files.parquet")
    assert plan_ops.load_incomplete_files(tmp_path)["file_path"].to_list() == ["a"]
