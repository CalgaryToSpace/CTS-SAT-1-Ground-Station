import json
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames.pipeline import (
    assign_mpi_collections,
    compute_mpi_frames,
)
from cts1_mo_tools.cts1_processing_pipeline.step_8_summarize_mpi_collections.pipeline import (  # noqa: E501
    count_downlinked_strong_signal_frames,
    extract_all_mpi_analyses,
    extract_mpi_analyses,
    summarize_mpi_collections,
)

from .test_orbit import GNSS_FIX_AT, TLE_HISTORY

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_RECORDED_AT = datetime(2026, 8, 9, 6, 33, 10, tzinfo=UTC)
_DOWNLINKED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _epoch_ms(at: datetime) -> int:
    return round((at - _EPOCH).total_seconds() * 1000)


def _frames(
    *runs: tuple[datetime, list[int]],
    integration_period: int = 260,
    time_sync_source: str = "G",
) -> pl.DataFrame:
    """Frames 31ms apart from each (start time, counters) run, warming up by
    0.5 C per frame from 10 C, with the clock set 1 minute before the run.
    """
    times: list[datetime] = []
    counters: list[int] = []
    temperatures: list[float] = []
    since_sync: list[int] = []
    for start, run_counters in runs:
        times += [
            start + timedelta(milliseconds=31 * i) for i in range(len(run_counters))
        ]
        counters += run_counters
        temperatures += [10 + 0.5 * i for i in range(len(run_counters))]
        since_sync += [60_000 + 31 * i for i in range(len(run_counters))]
    return pl.DataFrame(
        {
            "collected_at": times,
            "frame_counter": counters,
            "board_temperature_c": temperatures,
            "integration_period": [integration_period] * len(times),
            "time_sync_source": [time_sync_source] * len(times),
            "ms_since_time_sync": since_sync,
        },
        schema={
            "collected_at": pl.Datetime("us", "UTC"),
            "frame_counter": pl.Int64,
            "board_temperature_c": pl.Float64,
            "integration_period": pl.Int64,
            "time_sync_source": pl.String,
            "ms_since_time_sync": pl.Int64,
        },
    )


def test_collection_summary_counts_frames_and_largest_gap() -> None:
    second_start = _RECORDED_AT + timedelta(minutes=5)
    present = [0, 1, 2, 6, 7, 9, 10]  # Gaps of 3 (3..5) and 1 (8).
    frames = assign_mpi_collections(
        _frames((_RECORDED_AT, present), (second_start, [0, 1, 2]))
    )
    summary = summarize_mpi_collections(frames)

    first, second = summary.rows(named=True)
    analysis_columns = [c for c in summary.columns if c.startswith("analysis_")]
    assert analysis_columns
    # No TLE history given, so not placed in orbit.
    orbit_columns = [
        "start_latitude_deg",
        "start_longitude_deg",
        "end_latitude_deg",
        "end_longitude_deg",
        "sun_fraction_min",
        "sun_fraction_max",
        "sunlit",
        "orbit_tle_epoch",
    ]
    assert first == {
        "collection_started_at": _RECORDED_AT,
        "collection_ended_at": _RECORDED_AT + timedelta(milliseconds=31 * 6),
        "frames_present": 7,
        "frames_expected": 11,
        "frames_missing": 4,
        "first_frame_counter": 0,
        "last_frame_counter": 10,
        "largest_gap_frames": 3,
        "largest_gap_started_at": _RECORDED_AT + timedelta(milliseconds=31 * 2),
        "largest_gap_duration_ms": 31,
        "integration_periods": "[260]",
        "time_sync_sources": '["G"]',
        "time_synced_at": _RECORDED_AT - timedelta(minutes=1),
        "last_time_synced_at": _RECORDED_AT - timedelta(minutes=1),
        "time_resyncs_during": 0,
        "ms_since_time_sync_at_start": 60_000,
        "board_temperature_min_c": 10.0,
        "board_temperature_max_c": 13.0,
        "collection_time_source": "frames",
        "collection_duration_ms": 31 * 6,
        "frame_collection_count": 1,
        **dict.fromkeys(orbit_columns),
        **dict.fromkeys(analysis_columns),
    }
    assert second["frames_missing"] == 0
    assert second["largest_gap_frames"] == 0
    assert second["largest_gap_duration_ms"] is None


def test_contiguous_frames_summarize_with_none_missing() -> None:
    (summary,) = summarize_mpi_collections(
        assign_mpi_collections(_frames((_RECORDED_AT, list(range(50)))))
    ).rows(named=True)
    assert summary["frames_present"] == summary["frames_expected"] == 50
    assert summary["frames_missing"] == 0
    assert summary["collection_time_source"] == "frames"


def test_empty_frames_summarize_to_empty_collections() -> None:
    summary = summarize_mpi_collections(compute_mpi_frames(pl.DataFrame()))
    assert summary.height == 0
    assert summary.schema["collection_started_at"] == pl.Datetime("us", "UTC")


def _analysis_response(  # noqa: PLR0913
    *,
    earliest: datetime,
    latest: datetime,
    file: str = "mpi_data/2026-08-09_063310Z.mpi",
    size: int = 103285,
    received_at: datetime = _DOWNLINKED_AT,
    is_complete: bool = True,
    response_code: int = 0,
    action: str = "analyze_mpi_data_v1",
    strong_threshold_dn: int = 200,
    strong_signal_count: int = 0,
    strong_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = json.dumps(
        {
            "action": action,
            "file": file,
            "sha256": "2EA8096F" + "0" * 56,
            "size": size,
            "frame_count": 673,
            "valid_frame_count": 667,
            "bad_frame_count": 6,
            "background_frame_count": 3,
            "warmup_frame_count": 78,
            "frame_byte_range": [259, 102919],
            "time_sync_count": 6,
            "mpi_start_count": 1,
            "malformed_time_sync_count": 0,
            "earliest": {
                "timestamp_ms": _epoch_ms(earliest),
                "datetime": "2026-08-09T063310.000Z_E",
            },
            "latest": {
                "timestamp_ms": _epoch_ms(latest),
                "datetime": "2026-08-09T063410.000Z_E",
            },
            "strong_threshold_dn": strong_threshold_dn,
            "strong_frame_count": 0,
            "strong_signal_count": strong_signal_count,
            "strong_signals": strong_signals or [],
        }
    )
    return {
        "first_received_at": received_at,
        "tcmd_sent_at": received_at - timedelta(seconds=5),
        "tcmd_response_code": response_code,
        "is_complete": is_complete,
        "tcmd_response_text": text,
    }


def _tcmd_responses(*responses: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            *responses,
            # Not an analysis at all.
            {
                "first_received_at": _DOWNLINKED_AT,
                "tcmd_sent_at": None,
                "tcmd_response_code": 0,
                "is_complete": True,
                "tcmd_response_text": "pong",
            },
        ],
        schema={
            "first_received_at": pl.Datetime("us", "UTC"),
            "tcmd_sent_at": pl.Datetime("us", "UTC"),
            "tcmd_response_code": pl.Int64,
            "is_complete": pl.Boolean,
            "tcmd_response_text": pl.String,
        },
    )


def test_extract_mpi_analyses_keeps_complete_successful_ones() -> None:
    earliest = _RECORDED_AT
    latest = _RECORDED_AT + timedelta(minutes=1)
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(earliest=earliest, latest=latest),
            _analysis_response(
                earliest=earliest, latest=latest, file="b.mpi", is_complete=False
            ),
            _analysis_response(
                earliest=earliest, latest=latest, file="c.mpi", response_code=1
            ),
        )
    )
    (analysis,) = analyses.rows(named=True)
    assert analysis["analysis_started_at"] == earliest
    assert analysis["analysis_ended_at"] == latest
    assert analysis["analysis_file"] == "mpi_data/2026-08-09_063310Z.mpi"
    assert analysis["analysis_file_size"] == 103285
    assert analysis["analysis_sha256"] == "2ea8096f" + "0" * 56
    assert analysis["analysis_frame_count"] == 673
    assert analysis["analysis_valid_frame_count"] == 667
    assert analysis["analysis_first_frame_byte"] == 259
    assert analysis["analysis_last_frame_byte"] == 102919
    assert analysis["analysis_strong_signals"] == "[]"
    assert analysis["analysis_action"] == "analyze_mpi_data_v1"
    assert analysis["analysis_version"] == 1


def test_extract_mpi_analyses_keeps_the_largest_per_file() -> None:
    earliest = _RECORDED_AT
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=earliest, latest=earliest + timedelta(seconds=30), size=50
            ),
            _analysis_response(
                earliest=earliest, latest=earliest + timedelta(seconds=60), size=99
            ),
        )
    )
    assert analyses["analysis_file_size"].to_list() == [99]


def test_extract_mpi_analyses_prefers_the_latest_version_per_file() -> None:
    earliest = _RECORDED_AT
    latest = earliest + timedelta(seconds=60)
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(earliest=earliest, latest=latest, size=99),
            _analysis_response(
                earliest=earliest,
                latest=latest,
                size=50,
                action="analyze_mpi_data_v2",
            ),
            _analysis_response(
                earliest=earliest, latest=latest, size=999, action="analyze_mpi_data"
            ),
        )
    )
    assert analyses.select("analysis_version", "analysis_file_size").rows() == [(2, 50)]


def test_analysis_times_and_describes_its_collection() -> None:
    earliest = _RECORDED_AT - timedelta(seconds=5)
    latest = _RECORDED_AT + timedelta(minutes=1)
    unrelated = _RECORDED_AT + timedelta(hours=2)
    no_frames = _RECORDED_AT + timedelta(days=1)
    frames = assign_mpi_collections(
        _frames((_RECORDED_AT, [0, 1, 2, 5]), (unrelated, [0, 1]))
    )
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(earliest=earliest, latest=latest),
            _analysis_response(
                earliest=no_frames,
                latest=no_frames + timedelta(minutes=1),
                file="other.mpi",
            ),
        )
    )
    summary = summarize_mpi_collections(frames, analyses)

    merged, frames_only, analysis_only = summary.rows(named=True)
    assert merged["collection_started_at"] == earliest
    assert merged["collection_ended_at"] == latest
    assert merged["collection_time_source"] == "analysis"
    assert merged["collection_duration_ms"] == 65_000
    assert merged["frame_collection_count"] == 1
    assert merged["frames_present"] == 4
    assert merged["frames_missing"] == 2
    assert merged["analysis_file"] == "mpi_data/2026-08-09_063310Z.mpi"
    assert merged["analysis_frame_count"] == 673

    assert frames_only["collection_started_at"] == unrelated
    assert frames_only["collection_time_source"] == "frames"
    assert frames_only["analysis_file"] is None

    assert analysis_only["collection_started_at"] == no_frames
    assert analysis_only["frame_collection_count"] == 0
    assert analysis_only["frames_present"] == 0
    assert analysis_only["frames_expected"] is None
    assert analysis_only["analysis_file"] == "other.mpi"


def test_analysis_merges_every_recording_in_its_file() -> None:
    second_start = _RECORDED_AT + timedelta(seconds=30)
    frames = assign_mpi_collections(
        _frames((_RECORDED_AT, [0, 1, 3]), (second_start, [0, 1]))
    )
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=_RECORDED_AT, latest=second_start + timedelta(seconds=10)
            )
        )
    )
    (merged,) = summarize_mpi_collections(frames, analyses).rows(named=True)
    assert merged["frame_collection_count"] == 2
    assert merged["frames_present"] == 5
    assert merged["frames_expected"] == 6
    assert merged["first_frame_counter"] == 0
    assert merged["last_frame_counter"] == 1
    assert merged["largest_gap_frames"] == 1


def test_all_analyses_keeps_each_distinct_response_once() -> None:
    earliest = _RECORDED_AT
    latest = earliest + timedelta(seconds=60)
    response = _analysis_response(earliest=earliest, latest=latest)
    redownlinked = {**response, "first_received_at": _DOWNLINKED_AT + timedelta(1)}
    other_threshold = _analysis_response(
        earliest=earliest, latest=latest, strong_threshold_dn=100
    )
    analyses = extract_all_mpi_analyses(
        _tcmd_responses(redownlinked, response, other_threshold)
    )
    assert analyses.select("analysis_strong_threshold_dn", "analysis_received_at").sort(
        "analysis_strong_threshold_dn"
    ).rows() == [
        (100, _DOWNLINKED_AT),
        (200, _DOWNLINKED_AT),
    ]
    assert json.loads(analyses["analysis_response_text"][0])["file"] == (
        "mpi_data/2026-08-09_063310Z.mpi"
    )


def test_primary_analysis_counts_strong_signals_at_every_threshold() -> None:
    earliest = _RECORDED_AT
    latest = earliest + timedelta(seconds=60)
    (analysis,) = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=earliest,
                latest=latest,
                strong_threshold_dn=100,
                strong_signal_count=5,
            ),
            _analysis_response(earliest=earliest, latest=latest, size=50),
            # Same threshold, but a smaller file: not the one counted.
            _analysis_response(
                earliest=earliest,
                latest=latest,
                strong_threshold_dn=100,
                strong_signal_count=9,
                size=10,
            ),
        )
    ).rows(named=True)
    assert analysis["analysis_count"] == 3
    assert analysis["analysis_file_size"] == 103285
    assert json.loads(analysis["analysis_strong_signal_counts"]) == {
        "100": 5,
        "200": 0,
    }


def test_counts_downlinked_frames_of_each_strong_signal() -> None:
    earliest = _RECORDED_AT
    latest = earliest + timedelta(seconds=60)
    signals = [
        {"bytes": [1000, 1304], "frames": [5, 6]},  # Both frames downlinked.
        {"bytes": [2000, 2304], "frames": [9, 10]},  # One of two.
        {"bytes": [9000, 9152], "frames": [50, 50]},  # Not downlinked.
    ]
    analyses = extract_all_mpi_analyses(
        _tcmd_responses(
            _analysis_response(earliest=earliest, latest=latest, strong_signals=signals)
        )
    )
    in_window = earliest + timedelta(seconds=10)
    frames = pl.DataFrame(
        {
            "collected_at": [in_window] * 3 + [earliest + timedelta(days=1)],
            "file_offset": [1000, 1152, 2152, 9000],  # The last: another file's.
            "frame_counter": [5, 6, 10, 50],
        },
        schema={
            "collected_at": pl.Datetime("us", "UTC"),
            "file_offset": pl.Int64,
            "frame_counter": pl.Int64,
        },
    )
    (counted,) = count_downlinked_strong_signal_frames(analyses, frames).rows(
        named=True
    )
    assert json.loads(counted["analysis_strong_signal_frames_downlinked"]) == [2, 1, 0]


def test_analysis_merges_frame_temperatures_periods_and_sync_sources() -> None:
    second_start = _RECORDED_AT + timedelta(seconds=30)
    frames = assign_mpi_collections(
        pl.concat(
            [
                _frames((_RECORDED_AT, [0, 1, 2])),
                _frames(
                    (second_start, [0, 1]),
                    integration_period=200,
                    time_sync_source="E",
                ),
            ]
        )
    )
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=_RECORDED_AT, latest=second_start + timedelta(seconds=10)
            )
        )
    )
    (merged,) = summarize_mpi_collections(frames, analyses).rows(named=True)
    assert merged["integration_periods"] == "[200,260]"
    assert merged["time_sync_sources"] == '["E","G"]'
    # The second run's clock was set 30s after the first's.
    assert merged["time_resyncs_during"] == 1
    assert merged["ms_since_time_sync_at_start"] == 60_000
    assert merged["board_temperature_min_c"] == 10.0
    assert merged["board_temperature_max_c"] == 11.0


def test_collections_are_placed_in_orbit_from_the_tle_history() -> None:
    frames = assign_mpi_collections(_frames((GNSS_FIX_AT, list(range(100)))))
    analysis_only_at = GNSS_FIX_AT + timedelta(hours=1)
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=analysis_only_at,
                latest=analysis_only_at + timedelta(minutes=1),
            )
        )
    )
    in_sun, analysis_only = summarize_mpi_collections(
        frames, analyses, TLE_HISTORY
    ).rows(named=True)
    assert abs(in_sun["start_latitude_deg"] - 9.42) < 0.02
    assert abs(in_sun["start_longitude_deg"] - 133.62) < 0.02
    # 3.1 s later: a little further north.
    assert in_sun["end_latitude_deg"] > in_sun["start_latitude_deg"]
    assert in_sun["sunlit"] == "yes"
    assert in_sun["sun_fraction_min"] == in_sun["sun_fraction_max"] == 1.0
    assert in_sun["orbit_tle_epoch"] == TLE_HISTORY["epoch"][0]
    # Placed in orbit even with no frames downlinked.
    assert analysis_only["start_latitude_deg"] is not None
    assert analysis_only["sunlit"] in {"yes", "no", "partly"}


def test_resync_between_frames_is_counted() -> None:
    frames = _frames((_RECORDED_AT, list(range(10)))).with_columns(
        # The clock was set again just before frame 6.
        ms_since_time_sync=pl.when(pl.col("frame_counter") >= 6)
        .then(pl.col("ms_since_time_sync") - 60_000 - 31 * 5)
        .otherwise(pl.col("ms_since_time_sync"))
    )
    (summary,) = summarize_mpi_collections(assign_mpi_collections(frames)).rows(
        named=True
    )
    assert summary["time_resyncs_during"] == 1
    assert summary["time_synced_at"] == _RECORDED_AT - timedelta(minutes=1)
    assert summary["last_time_synced_at"] == _RECORDED_AT + timedelta(
        milliseconds=31 * 5
    )
    assert summary["ms_since_time_sync_at_start"] == 60_000


def test_resync_before_the_first_frame_received_is_counted() -> None:
    # The recording started 2 minutes before the first frame received, and
    # the clock was set 1 minute before that frame.
    first_frame_at = _RECORDED_AT + timedelta(minutes=2)
    frames = assign_mpi_collections(_frames((first_frame_at, [100, 101])))
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=_RECORDED_AT, latest=first_frame_at + timedelta(seconds=1)
            )
        )
    )
    (merged,) = summarize_mpi_collections(frames, analyses).rows(named=True)
    assert merged["time_resyncs_during"] == 1
    # So it's given at the first frame received.
    assert merged["ms_since_time_sync_at_start"] == 60_000


def test_age_at_start_is_extrapolated_back_from_the_first_frame() -> None:
    first_frame_at = _RECORDED_AT + timedelta(seconds=20)
    frames = assign_mpi_collections(_frames((first_frame_at, [100, 101])))
    analyses = extract_mpi_analyses(
        _tcmd_responses(
            _analysis_response(
                earliest=_RECORDED_AT, latest=first_frame_at + timedelta(seconds=1)
            )
        )
    )
    (merged,) = summarize_mpi_collections(frames, analyses).rows(named=True)
    assert merged["time_resyncs_during"] == 0
    assert merged["ms_since_time_sync_at_start"] == 40_000
