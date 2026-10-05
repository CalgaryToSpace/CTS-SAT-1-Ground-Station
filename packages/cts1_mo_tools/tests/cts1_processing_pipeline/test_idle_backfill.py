"""Tests for the daemon's idle backfill: the `end` bounds it runs steps 0
and 1 with, how it picks which window to backfill next, and the daemon's
sleep loop handing its idle time over to it.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cts1_mo_tools.cts1_processing_pipeline import (
    daemon,
    daemon_signals,
    idle_backfill,
    landing_db,
)
from cts1_mo_tools.cts1_processing_pipeline.daemon_signals import StatusReporter
from cts1_mo_tools.cts1_processing_pipeline.idle_backfill import IdleBackfill
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations import (
    db as step_0_db,
)
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations import (
    pipeline as step_0_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations.pipeline import (
    LISTING_WINDOW,
    ListingWindow,
    window_containing,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    db as step_1_db,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)

_DAY = datetime(2026, 9, 1, tzinfo=UTC)


def _api_observation(obs_id: int, start: datetime) -> dict[str, Any]:
    """A trimmed-down observation, shaped like the SatNOGS API returns it."""
    return {
        "id": obs_id,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (start + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "norad_cat_id": 69015,
        "payload": f"https://example.invalid/satnogs_{obs_id}.ogg",
        "demoddata": [],
        "status": "good",
    }


def _fake_api(
    monkeypatch: pytest.MonkeyPatch,
    observations: list[dict[str, Any]],
    calls: list[tuple[datetime, datetime]] | None = None,
) -> None:
    """Stand in for the SatNOGS API, filtering `observations` by start."""

    def fake_fetch_all_observations(
        _norad_id: str,
        *,
        start_gt_filter: datetime,
        start_lt_filter: datetime,
        statuses: None,
    ) -> Iterator[list[dict[str, Any]]]:
        assert statuses is None
        if calls is not None:
            calls.append((start_gt_filter, start_lt_filter))
        page = [
            obs
            for obs in observations
            if start_gt_filter <= datetime.fromisoformat(obs["start"]) < start_lt_filter
        ]
        if page:
            yield page

    monkeypatch.setattr(
        step_0_pipeline, "fetch_all_observations", fake_fetch_all_observations
    )


def _backfill(tmp_path: Path, *, since: datetime, until: datetime) -> IdleBackfill:
    return IdleBackfill(
        norad_id="69015",
        data_dir=tmp_path,
        until=until,
        workers=1,
        temp_dir=None,
        tools=None,
        since=since,
    )


# ---------------------------------------------------------------------------
# The `end` bounds on steps 0 and 1.
# ---------------------------------------------------------------------------


def test_window_containing_rounds_down_to_a_window_boundary() -> None:
    assert window_containing(_DAY + timedelta(hours=13)).start == _DAY + LISTING_WINDOW
    assert window_containing(_DAY).start == _DAY


def test_step_0_end_lists_only_the_windows_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[datetime, datetime]] = []
    _fake_api(monkeypatch, [], calls)

    step_0_pipeline.run(
        data_dir=tmp_path,
        start=_DAY.isoformat(),
        end=(_DAY + LISTING_WINDOW).isoformat(),
    )

    assert len(calls) == 1
    with landing_db.connect(tmp_path / landing_db.DB_FILENAME) as con:
        assert set(step_0_db.window_states(con)) == {_DAY}


def test_step_1_loads_observations_bounded_on_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_api(
        monkeypatch,
        [_api_observation(n, _DAY + timedelta(hours=n)) for n in (1, 2, 3)],
    )
    step_0_pipeline.run(data_dir=tmp_path, start=_DAY.isoformat())

    with landing_db.connect(tmp_path / landing_db.DB_FILENAME) as con:
        loaded = step_1_db.load_observations(
            con,
            norad_id="69015",
            start_gte=_DAY + timedelta(hours=2),
            start_lt=_DAY + timedelta(hours=3),
        )
    assert [obs["id"] for obs in loaded] == [2]


# ---------------------------------------------------------------------------
# Picking what to backfill.
# ---------------------------------------------------------------------------


def test_pending_windows_are_unlisted_or_undecoded_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Of four windows: two listed with every observation decoded (done),
    one listed with an observation left to decode, one never listed.
    """
    windows = [ListingWindow(_DAY + LISTING_WINDOW * n) for n in range(4)]
    observations = [
        _api_observation(1, windows[0].start + timedelta(hours=1)),
        _api_observation(2, windows[1].start + timedelta(hours=1)),
        _api_observation(3, windows[2].start + timedelta(hours=1)),
    ]
    _fake_api(monkeypatch, observations)
    now = windows[3].end + timedelta(days=1)
    with landing_db.connect(tmp_path / landing_db.DB_FILENAME) as con:
        for window in windows[:3]:
            step_0_db.record_listing(
                con,
                step_0_pipeline.list_window(
                    con, norad_id="69015", window=window, now=now
                ),
            )
        every_decoder: dict[str, str | None] = dict.fromkeys(
            step_1_pipeline.DECODERS, "v1"
        )
        step_1_db.record_decoder_runs(con, 1, every_decoder, runtime_ms=1)
        step_1_db.record_decoder_runs(con, 3, every_decoder, runtime_ms=1)
        # Partly decoded -- still pending for the decoders that didn't run.
        step_1_db.record_decoder_runs(con, 2, {"sso_rx_replay": "v1"}, runtime_ms=1)

    backfill = _backfill(tmp_path, since=windows[0].start, until=windows[3].end)

    assert backfill.pending_windows() == [windows[3], windows[1]]


def test_each_window_is_backfilled_once_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, str]] = []

    def record_step(step: str) -> Any:
        def fake_run(*, start: str, end: str, **_kwargs: Any) -> None:
            calls.append((step, start, end))

        return fake_run

    # Neither step records anything, so every window stays pending -- the
    # backfill still has to move on rather than retrying the same window.
    monkeypatch.setattr(step_0_pipeline, "run", record_step("step_0"))
    monkeypatch.setattr(step_1_pipeline, "run", record_step("step_1"))
    backfill = _backfill(tmp_path, since=_DAY, until=_DAY + LISTING_WINDOW * 2)

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        assert backfill.run_next_chunk(reporter)
        assert backfill.run_next_chunk(reporter)
        assert not backfill.run_next_chunk(reporter)

    newer, older = _DAY + LISTING_WINDOW, _DAY
    assert calls == [
        ("step_0", newer.isoformat(), (newer + LISTING_WINDOW).isoformat()),
        ("step_1", newer.isoformat(), (newer + LISTING_WINDOW).isoformat()),
        ("step_0", older.isoformat(), (older + LISTING_WINDOW).isoformat()),
        ("step_1", older.isoformat(), (older + LISTING_WINDOW).isoformat()),
    ]


def test_a_failing_chunk_does_not_stop_the_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kwargs: Any) -> None:
        msg = "SatNOGS is down"
        raise RuntimeError(msg)

    monkeypatch.setattr(step_0_pipeline, "run", boom)
    backfill = _backfill(tmp_path, since=_DAY, until=_DAY + LISTING_WINDOW)

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        assert backfill.run_next_chunk(reporter)
        assert not backfill.run_next_chunk(reporter)


def test_nothing_to_backfill_before_history_starts(tmp_path: Path) -> None:
    backfill = _backfill(tmp_path, since=_DAY, until=_DAY - timedelta(days=1))
    assert backfill.pending_windows() == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" Yes ", True), ("0", False), ("", False)],
)
def test_enabled_by_env(
    monkeypatch: pytest.MonkeyPatch, value: str, *, expected: bool
) -> None:
    monkeypatch.setenv(idle_backfill.ENV_VAR, value)
    assert idle_backfill.enabled_by_env() is expected


# ---------------------------------------------------------------------------
# The daemon's sleep loop running idle work.
# ---------------------------------------------------------------------------


def test_sleep_runs_idle_work_until_it_runs_out(tmp_path: Path) -> None:
    remaining_chunks = [3]

    def idle_work(_reporter: StatusReporter) -> bool:
        if remaining_chunks[0] == 0:
            return False
        remaining_chunks[0] -= 1
        return True

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        was_triggered = daemon.sleep_until_next_run(
            data_dir=tmp_path, interval=0.005, reporter=reporter, idle_work=idle_work
        )

    assert not was_triggered
    assert remaining_chunks == [0]


def test_a_trigger_request_interrupts_idle_work_between_chunks(
    tmp_path: Path,
) -> None:
    chunks_run = [0]

    def idle_work(_reporter: StatusReporter) -> bool:
        chunks_run[0] += 1
        daemon_signals.request_pipeline_run(tmp_path, note="mid-backfill")
        return True

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        was_triggered = daemon.sleep_until_next_run(
            data_dir=tmp_path, interval=60.0, reporter=reporter, idle_work=idle_work
        )

    assert was_triggered
    assert chunks_run == [1]
