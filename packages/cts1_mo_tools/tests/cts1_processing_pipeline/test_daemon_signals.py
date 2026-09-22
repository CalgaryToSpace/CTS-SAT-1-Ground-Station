"""Tests for the daemon <-> web UI file signalling, for the daemon's sleep
loop honouring a trigger request, and for what it announces per step.
"""

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cts1_mo_tools.cts1_processing_pipeline import daemon, daemon_signals
from cts1_mo_tools.cts1_processing_pipeline.daemon_signals import (
    DaemonState,
    StatusReporter,
)
from loguru import logger

# ---------------------------------------------------------------------------
# Trigger requests (web UI -> daemon).
# ---------------------------------------------------------------------------


def test_trigger_request_round_trip(tmp_path: Path) -> None:
    assert daemon_signals.read_trigger_request(tmp_path) is None

    daemon_signals.request_pipeline_run(tmp_path, note="web UI")
    request = daemon_signals.read_trigger_request(tmp_path)
    assert request is not None
    assert request.note == "web UI"
    assert request.requested_at is not None
    assert request.requested_at.tzinfo is not None

    daemon_signals.clear_trigger_request(tmp_path)
    assert daemon_signals.read_trigger_request(tmp_path) is None


def test_clear_trigger_request_without_one_is_a_no_op(tmp_path: Path) -> None:
    daemon_signals.clear_trigger_request(tmp_path)  # must not raise


def test_repeated_requests_do_not_queue_up(tmp_path: Path) -> None:
    daemon_signals.request_pipeline_run(tmp_path)
    daemon_signals.request_pipeline_run(tmp_path)
    daemon_signals.clear_trigger_request(tmp_path)
    assert daemon_signals.read_trigger_request(tmp_path) is None


def test_malformed_request_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / daemon_signals.TRIGGER_REQUEST_FILENAME).write_text("not json{")
    assert daemon_signals.read_trigger_request(tmp_path) is None


def test_request_without_timestamp_still_counts(tmp_path: Path) -> None:
    (tmp_path / daemon_signals.TRIGGER_REQUEST_FILENAME).write_text("{}")
    request = daemon_signals.read_trigger_request(tmp_path)
    assert request is not None
    assert request.requested_at is None


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    daemon_signals.request_pipeline_run(tmp_path)
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


# ---------------------------------------------------------------------------
# Status (daemon -> web UI).
# ---------------------------------------------------------------------------


def test_status_round_trip(tmp_path: Path) -> None:
    assert daemon_signals.read_status(tmp_path) is None

    next_run_at = datetime.now(UTC) + timedelta(minutes=5)
    daemon_signals.write_status(
        tmp_path,
        state=DaemonState.SLEEPING,
        detail="next requery in 5.0 minute(s)",
        next_run_at=next_run_at,
    )

    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert status.state is DaemonState.SLEEPING
    assert status.detail == "next requery in 5.0 minute(s)"
    assert status.next_run_at == next_run_at
    assert status.pid is not None
    assert status.is_live
    assert not status.is_processing


def test_processing_status_is_processing(tmp_path: Path) -> None:
    daemon_signals.write_status(tmp_path, state=DaemonState.PROCESSING)
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert status.is_processing


def test_stale_status_is_not_live(tmp_path: Path) -> None:
    stale = datetime.now(UTC) - daemon_signals.STATUS_STALE_AFTER - timedelta(minutes=1)
    (tmp_path / daemon_signals.DAEMON_STATUS_FILENAME).write_text(
        json.dumps({"state": "processing", "updated_at": stale.isoformat()})
    )
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert not status.is_live
    assert not status.is_processing


def test_stopped_status_is_not_live(tmp_path: Path) -> None:
    daemon_signals.write_status(tmp_path, state=DaemonState.STOPPED)
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert not status.is_live


def test_unknown_state_degrades_rather_than_raising(tmp_path: Path) -> None:
    (tmp_path / daemon_signals.DAEMON_STATUS_FILENAME).write_text(
        json.dumps(
            {"state": "from-a-newer-build", "updated_at": datetime.now(UTC).isoformat()}
        )
    )
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert status.state is DaemonState.UNKNOWN
    assert status.is_live


def test_naive_timestamp_is_read_as_utc(tmp_path: Path) -> None:
    naive = datetime.now(UTC).replace(tzinfo=None)
    (tmp_path / daemon_signals.DAEMON_STATUS_FILENAME).write_text(
        json.dumps({"state": "sleeping", "updated_at": naive.isoformat()})
    )
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert status.updated_at is not None
    assert status.updated_at.tzinfo is not None
    assert status.is_live


def test_malformed_status_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / daemon_signals.DAEMON_STATUS_FILENAME).write_text("}{")
    assert daemon_signals.read_status(tmp_path) is None


# ---------------------------------------------------------------------------
# StatusReporter.
# ---------------------------------------------------------------------------


def test_status_reporter_publishes_and_stops(tmp_path: Path) -> None:
    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        started = daemon_signals.read_status(tmp_path)
        assert started is not None
        assert started.state is DaemonState.STARTING

        reporter.set(DaemonState.PROCESSING, detail="backfill: step 1 (download)")
        status = daemon_signals.read_status(tmp_path)
        assert status is not None
        assert status.state is DaemonState.PROCESSING
        assert status.detail == "backfill: step 1 (download)"
        assert status.last_run_finished_at is None

        reporter.set(DaemonState.SLEEPING, detail="waiting")

    stopped = daemon_signals.read_status(tmp_path)
    assert stopped is not None
    assert stopped.state is DaemonState.STOPPED
    assert not stopped.is_live
    # Leaving PROCESSING stamped when the run ended.
    assert stopped.last_run_finished_at is not None


def test_status_reporter_heartbeat_keeps_the_file_fresh(tmp_path: Path) -> None:
    """The heartbeat thread rewrites the file without any `set` call --
    which is what keeps a daemon blocked inside a long pipeline step
    distinguishable from one that died there.
    """
    with StatusReporter(tmp_path, interval_sec=0.02) as reporter:
        reporter.set(DaemonState.PROCESSING, detail="step 1")
        first = daemon_signals.read_status(tmp_path)
        assert first is not None
        assert first.updated_at is not None

        deadline = datetime.now(UTC) + timedelta(seconds=5)
        while datetime.now(UTC) < deadline:
            latest = daemon_signals.read_status(tmp_path)
            assert latest is not None
            assert latest.updated_at is not None
            if latest.updated_at > first.updated_at:
                assert latest.state is DaemonState.PROCESSING
                return
        msg = "status file was never rewritten by the heartbeat thread"
        raise AssertionError(msg)


def test_concurrent_status_writers_leave_no_temp_files(tmp_path: Path) -> None:
    """The status file has two writers (the heartbeat thread and whoever
    calls `set`), so its temp files must not collide -- a shared fixed temp
    name strands half-written files in the data directory.
    """
    barrier = threading.Barrier(8)

    def _write() -> None:
        barrier.wait()
        for _ in range(25):
            daemon_signals.write_status(tmp_path, state=DaemonState.PROCESSING)

    threads = [threading.Thread(target=_write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [p.name for p in tmp_path.glob("*.tmp")] == []
    status = daemon_signals.read_status(tmp_path)
    assert status is not None
    assert status.state is DaemonState.PROCESSING


def test_status_reporter_survives_an_unwritable_data_dir(tmp_path: Path) -> None:
    """A status file that can't be written must never take the daemon
    down -- the pipeline's actual work matters, this indicator doesn't.
    """
    missing_dir = tmp_path / "does-not-exist"
    with StatusReporter(missing_dir, interval_sec=0.05) as reporter:
        reporter.set(DaemonState.PROCESSING, detail="step 1")


# ---------------------------------------------------------------------------
# The daemon's sleep loop.
# ---------------------------------------------------------------------------


def test_sleep_returns_early_on_a_trigger_request(tmp_path: Path) -> None:
    daemon_signals.request_pipeline_run(tmp_path, note="web UI")

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        started_at = datetime.now(UTC)
        was_triggered = daemon.sleep_until_next_run(
            data_dir=tmp_path, interval=60.0, reporter=reporter
        )

    assert was_triggered
    # Returned on the request, not after the 60-minute interval.
    assert datetime.now(UTC) - started_at < timedelta(seconds=10)
    # And consumed it, so the *next* run isn't triggered by the same one.
    assert daemon_signals.read_trigger_request(tmp_path) is None


def test_sleep_returns_normally_when_the_interval_elapses(tmp_path: Path) -> None:
    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        was_triggered = daemon.sleep_until_next_run(
            data_dir=tmp_path, interval=0.0, reporter=reporter
        )
    assert not was_triggered


def test_sleep_wakes_on_a_request_that_arrives_mid_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The poll interval is what makes the button feel immediate -- a
    request written after the sleep has already started still cuts it
    short, rather than waiting out the full interval.
    """
    monkeypatch.setattr(daemon, "TRIGGER_POLL_INTERVAL_SEC", 0.05)

    def _request_soon() -> None:
        daemon_signals.request_pipeline_run(tmp_path, note="mid-sleep")

    timer = threading.Timer(0.2, _request_soon)
    timer.start()
    try:
        with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
            started_at = datetime.now(UTC)
            was_triggered = daemon.sleep_until_next_run(
                data_dir=tmp_path, interval=60.0, reporter=reporter
            )
    finally:
        timer.cancel()

    assert was_triggered
    assert datetime.now(UTC) - started_at < timedelta(seconds=10)


def test_sleeping_publishes_a_next_run_time(tmp_path: Path) -> None:
    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        daemon.sleep_until_next_run(
            data_dir=tmp_path, interval=0.005, reporter=reporter
        )
        status = daemon_signals.read_status(tmp_path)

    assert status is not None
    assert status.state is DaemonState.SLEEPING
    assert status.next_run_at is not None


# ---------------------------------------------------------------------------
# Per-step announcements.
# ---------------------------------------------------------------------------


def test_every_step_announces_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step logs a "Starting step N/4: <name> -- <run label>." line
    before it runs, and publishes the same name to the web UI's status
    indicator.
    """

    def do_nothing(**_kwargs: Any) -> None:
        """Stand in for a step's `run`, so this exercises only the daemon."""

    for module in (
        daemon.step_1_pipeline,
        daemon.step_2_pipeline,
        daemon.step_3_pipeline,
        daemon.step_4_pipeline,
    ):
        monkeypatch.setattr(module, "run", do_nothing)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    details: list[str | None] = []

    with StatusReporter(tmp_path, interval_sec=0.05) as reporter:
        original_set = reporter.set

        def recording_set(state: DaemonState, **kwargs: Any) -> None:
            details.append(kwargs.get("detail"))
            original_set(state, **kwargs)

        monkeypatch.setattr(reporter, "set", recording_set)
        try:
            daemon.run_all_steps(
                norad_id="69015",
                data_dir=tmp_path,
                start=None,
                limit=None,
                workers=1,
                temp_dir=None,
                force_rerun=False,
                tools=None,
                reporter=reporter,
                run_label="backfill",
            )
        finally:
            logger.remove(sink_id)

    log = "\n".join(messages)
    for number, name in daemon.STEP_NAMES.items():
        assert f"Starting step {number}/4: {name} -- backfill." in log
        assert f"backfill: step {number} ({name})" in details

    assert "Finished all 4 steps -- backfill." in log
    # In order, and once each.
    starts = [m for m in messages if m.startswith("Starting step ")]
    assert [m.split("/")[0] for m in starts] == [
        f"Starting step {n}" for n in daemon.STEP_NAMES
    ]
