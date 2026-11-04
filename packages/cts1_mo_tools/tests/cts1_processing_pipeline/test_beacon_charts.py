from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.charts import (
    ChartSpec,
    chart_option,
)

_BASE = datetime(2026, 11, 1, tzinfo=UTC)


def _beacons(values: Sequence[object]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "utc_time": [_BASE + timedelta(seconds=10 * i) for i in range(len(values))],
            "x": values,
        }
    ).with_columns(pl.col("utc_time").cast(pl.Datetime("us", "UTC")))


def test_series_is_sent_as_epoch_ms() -> None:
    option = chart_option(ChartSpec("X", ["x"]), _beacons([1.0, 2.0, 3.0]))
    assert option is not None
    data = option["series"][0]["data"]
    base_ms = int(_BASE.timestamp() * 1000)
    assert data == [[base_ms, 1.0], [base_ms + 10_000, 2.0], [base_ms + 20_000, 3.0]]
