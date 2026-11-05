import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline import landing_db, partitions
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import db


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _frame(*rows: tuple[str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"received_at": [_dt(t) for t, _ in rows], "value": [v for _, v in rows]},
        schema={"received_at": pl.Datetime("us", "UTC"), "value": pl.Int64},
    ).sort("received_at")


def test_write_frame_full_then_some_months(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    df = _frame(
        ("2026-08-01T00:00:00", 1),
        ("2026-08-31T23:59:59", 2),
        ("2026-09-15T00:00:00", 3),
        ("2026-10-01T00:00:00", 4),
    )

    partitions.write_frame(df, path)

    assert sorted(partitions.month_files(path)) == ["2026-08", "2026-09", "2026-10"]
    assert pl.read_parquet(path).equals(df)

    # September changes, October goes away, August is left alone.
    august_mtime = partitions.month_files(path)["2026-08"].stat().st_mtime_ns
    partitions.write_frame(
        _frame(("2026-09-16T00:00:00", 5)), path, months={"2026-09", "2026-10"}
    )

    assert sorted(partitions.month_files(path)) == ["2026-08", "2026-09"]
    assert partitions.month_files(path)["2026-08"].stat().st_mtime_ns == august_mtime
    assert pl.read_parquet(path)["value"].to_list() == [1, 2, 5]
    # Nothing but partition files in the directory, or polars can't read it.
    assert {p.suffix for p in path.iterdir()} == {".parquet"}


def test_full_write_replaces_a_single_file(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    _frame(("2026-01-01T00:00:00", 0)).write_parquet(path)

    partitions.write_frame(_frame(("2026-08-01T00:00:00", 1)), path)

    assert path.is_dir()
    assert pl.read_parquet(path)["value"].to_list() == [1]


def test_empty_output_is_still_readable(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"

    partitions.write_frame(_frame(), path)

    assert pl.read_parquet(path).is_empty()


def test_changed_input_months(tmp_path: Path) -> None:
    inputs = tmp_path / "in.parquet"
    output = tmp_path / "out.parquet"
    partitions.write_frame(
        _frame(("2026-08-01T00:00:00", 1), ("2026-09-01T00:00:00", 2)), inputs
    )
    partitions.write_frame(_frame(("2026-08-01T00:00:00", 1)), output)

    # No record of a previous run: has to be a full one.
    assert (
        partitions.changed_input_months(output, partitions.signatures(inputs)) is None
    )

    partitions.write_state(output, partitions.signatures(inputs))
    assert (
        partitions.changed_input_months(output, partitions.signatures(inputs)) == set()
    )

    time.sleep(0.01)  # So the rewrite gets a new mtime.
    partitions.write_frame(
        _frame(("2026-09-02T00:00:00", 3)), inputs, months={"2026-09"}
    )
    assert partitions.changed_input_months(output, partitions.signatures(inputs)) == {
        "2026-09"
    }

    # A month gone from the input: has to be a full run.
    partitions.write_frame(_frame(), inputs, months={"2026-09"})
    assert (
        partitions.changed_input_months(output, partitions.signatures(inputs)) is None
    )


def test_raw_packets_export_rewrites_only_changed_months(tmp_path: Path) -> None:
    db_path = tmp_path / landing_db.DB_FILENAME
    out_path = tmp_path / "raw_packets.parquet"
    # A single-file export from before raw_packets was partitioned.
    _frame(("2026-01-01T00:00:00", 0)).write_parquet(out_path)

    with landing_db.connect(db_path) as con:
        rows = _frame(("2026-08-01T00:00:00", 1), ("2026-09-01T00:00:00", 2))
        con.register("_rows", rows.with_columns(observation_id=pl.lit(1)))
        con.execute("CREATE TABLE raw_packets AS SELECT * FROM _rows")
        db.export_parquets(con, db_path)

        files = partitions.month_files(out_path)
        assert sorted(files) == ["2026-08", "2026-09"]
        august_mtime = files["2026-08"].stat().st_mtime_ns

        con.execute(
            "INSERT INTO raw_packets "
            "VALUES (TIMESTAMPTZ '2026-09-02 00:00:00+00', 3, 1)"
        )
        db.export_parquets(con, db_path)

    files = partitions.month_files(out_path)
    assert files["2026-08"].stat().st_mtime_ns == august_mtime
    assert pl.read_parquet(out_path)["value"].to_list() == [1, 2, 3]
