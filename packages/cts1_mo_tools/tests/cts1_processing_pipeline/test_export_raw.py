# pyright: standard
# Starlette's `TestClient` is typed through an httpx shim pyright can't
# follow in strict mode.

from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_raw import (
    raw_export_url,
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
