from cts1_mo_tools.cts1_processing_pipeline import daemon
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_page import (
    PIPELINE_FLOWCHART,
    duckdb_bootstrap_sql,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_tables import (
    TABLE_SPECS,
    TableSpec,
)


def test_flowchart_has_every_table() -> None:
    # The flowchart is kept by hand -- see `PIPELINE_FLOWCHART`.
    missing = [s.key for s in TABLE_SPECS if f'[("{s.key}")]' not in PIPELINE_FLOWCHART]
    assert not missing


def test_flowchart_has_every_step() -> None:
    missing = [
        n for n in daemon.STEP_NAMES if f'"Step {n}\\n' not in PIPELINE_FLOWCHART
    ]
    assert not missing


def test_duckdb_bootstrap_sql() -> None:
    a, b, c = (
        TableSpec(key, step, "", "", None, has_packet_type=False)
        for key, step in [("a", "Step 0"), ("b", "Step 0"), ("c", "Step 1")]
    )
    urls = {
        "a": ["/raw-export/a.parquet"],
        "b": ["/raw-export/b.parquet"],
        "c": ["/raw-export/c/2026-10.parquet", "/raw-export/c/2026-11.parquet"],
    }
    sql = duckdb_bootstrap_sql("https://x.test/", {a: [], b: [], c: ["c1", "c2"]}, urls)
    view = (
        "CREATE OR REPLACE VIEW {0} AS SELECT * FROM "
        "read_parquet(['https://x.test/raw-export/{0}.parquet']);"
    )
    assert sql.splitlines() == [
        "-- Step 0",
        view.format("a"),
        view.format("b"),
        "",
        "-- Step 1",
        "CREATE OR REPLACE VIEW c AS SELECT * FROM read_parquet(["
        "'https://x.test/raw-export/c/2026-10.parquet', "
        "'https://x.test/raw-export/c/2026-11.parquet']);",
        "",
        "-- Step 0",
        "SELECT * FROM a;",
        "SELECT * FROM b;",
        "",
        "-- Step 1",
        'SELECT "c1", "c2" FROM c;',
    ]
