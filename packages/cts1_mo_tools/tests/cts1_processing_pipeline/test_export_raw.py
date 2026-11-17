# pyright: standard
# Starlette's `TestClient` is typed through an httpx shim pyright can't
# follow in strict mode.

from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_raw import (
    raw_export_url,
    raw_export_urls,
    register_raw_export_route,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_head_reports_size_without_body(tmp_path: Path) -> None:
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(tmp_path / "raw_observations.parquet")
    app = FastAPI()
    register_raw_export_route(app, tmp_path)
    client = TestClient(app)

    head = client.head(raw_export_url("raw_observations"))
    get = client.get(raw_export_url("raw_observations"))

    assert head.status_code == 200
    assert head.content == b""
    assert int(head.headers["content-length"]) == len(get.content)


def test_partitioned_table_serves_each_month_off_disk(tmp_path: Path) -> None:
    table = tmp_path / "raw_packets.parquet"
    table.mkdir()
    for month in ["2026-10", "2026-11"]:
        pl.DataFrame({"m": [month]}).write_parquet(table / f"{month}.parquet")
    app = FastAPI()
    register_raw_export_route(app, tmp_path)
    client = TestClient(app)

    urls = raw_export_urls("raw_packets", tmp_path)

    assert urls == [
        "/raw-export/raw_packets/2026-10.parquet",
        "/raw-export/raw_packets/2026-11.parquet",
    ]
    assert client.get(urls[1]).content == (table / "2026-11.parquet").read_bytes()
    assert client.head(urls[0]).status_code == 200
    assert client.get("/raw-export/raw_packets/2026-12.parquet").status_code == 404
    assert client.get("/raw-export/raw_packets/..%2F..%2Fx.parquet").status_code == 404
    assert raw_export_urls("raw_observations", tmp_path) == [
        "/raw-export/raw_observations.parquet"
    ]
