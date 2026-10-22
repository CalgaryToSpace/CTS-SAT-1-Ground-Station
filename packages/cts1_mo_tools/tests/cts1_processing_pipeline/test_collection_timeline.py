from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.web_ui.charts import (
    collection_timeline_option,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.collection_timeline import (
    TIMELINE_SOURCES,
    load_collection_timeline,
)

_COLLECTED = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
_RECEIVED = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _ms(at: datetime) -> int:
    return round(at.timestamp() * 1000)


def _write_inputs(data_dir: Path) -> None:
    pl.DataFrame(
        {
            "packet_type": [
                "BEACON_BASIC",
                "BEACON_BASIC",  # Downlinked twice.
                "BEACON_EXTENDED",
                "BEACON_BASIC",  # Never-synced clock.
                "BEACON_BASIC",  # Bad CRC.
                "GNSS_BESTXYZB_SAMPLE",
                "GNSS_BESTXYZB_SAMPLE",  # Same sample, another ring dump.
                "LOG_MESSAGE",
            ],
            "received_at": [_RECEIVED + timedelta(minutes=i) for i in range(8)],
            "csp_crc_valid": [True, True, True, True, False, True, True, True],
            "unix_epoch_time_ms": [
                _ms(_COLLECTED),
                _ms(_COLLECTED),
                _ms(_COLLECTED + timedelta(seconds=10)),
                123_456,
                _ms(_COLLECTED + timedelta(hours=1)),
                None,
                None,
                None,
            ],
            "gnss_crc_valid": [None, None, None, None, None, True, True, None],
            "gnss_solution_utc_time": [None] * 5
            + ["2026-08-09T05:00:00.500+00:00"] * 2
            + [None],
        }
    ).write_parquet(data_dir / "everything_decoded.parquet")
    pl.DataFrame(
        {
            "collected_at": [
                _COLLECTED + timedelta(milliseconds=31 * i) for i in range(3)
            ],
            "frame_counter": [0, 1, 2],
        },
        schema_overrides={"collected_at": pl.Datetime("us", "UTC")},
    ).write_parquet(data_dir / "mpi_frames.parquet")


def test_timeline_uses_collection_times_deduplicated(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    timeline = load_collection_timeline(tmp_path)

    by_source = {
        source: part["collected_at"].to_list()
        for (source,), part in timeline.group_by("source")
    }
    assert by_source == {
        "Basic beacons": [_COLLECTED],
        "Extended beacons": [_COLLECTED + timedelta(seconds=10)],
        "MPI frames": [_COLLECTED + timedelta(milliseconds=31 * i) for i in range(3)],
        "GNSS samples": [datetime(2026, 8, 9, 5, 0, 0, 500_000, tzinfo=UTC)],
    }


def test_timeline_without_pipeline_output_is_empty(tmp_path: Path) -> None:
    timeline = load_collection_timeline(tmp_path)
    assert timeline.is_empty()
    assert (
        collection_timeline_option(timeline, sources=TIMELINE_SOURCES, title="") is None
    )


def test_timeline_option_has_a_lane_per_source(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    option = collection_timeline_option(
        load_collection_timeline(tmp_path), sources=TIMELINE_SOURCES, title=""
    )
    assert option is not None
    assert option["yAxis"]["data"] == [
        "GNSS samples (1)",
        "MPI frames (3)",
        "Extended beacons (1)",
        "Basic beacons (1)",
    ]
    mpi = next(s for s in option["series"] if s["name"] == "MPI frames (3)")
    assert mpi["data"][0] == [_ms(_COLLECTED), "MPI frames (3)"]
