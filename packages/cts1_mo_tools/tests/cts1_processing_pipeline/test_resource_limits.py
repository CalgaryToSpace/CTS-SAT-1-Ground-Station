"""Tests for the pipeline's resource caps -- see `resource_limits`."""

import os
import sys
from collections.abc import Iterator

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline import common, resource_limits
from loguru import logger


@pytest.fixture
def clean_thread_env() -> Iterator[None]:
    """Run with every thread-limit env var unset, restoring them after."""
    saved = {
        name: os.environ.pop(name, None)
        for name in resource_limits.THREAD_LIMIT_ENV_VARS
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.usefixtures("clean_thread_env")
def test_apply_thread_limits_sets_every_var() -> None:
    resource_limits.apply_thread_limits(3)
    for name in resource_limits.THREAD_LIMIT_ENV_VARS:
        assert os.environ[name] == "3"


@pytest.mark.usefixtures("clean_thread_env")
def test_apply_thread_limits_does_not_override_the_environment() -> None:
    """An operator's explicit POLARS_MAX_THREADS wins -- see the tuning doc."""
    os.environ["POLARS_MAX_THREADS"] = "8"
    resource_limits.apply_thread_limits(1)
    assert os.environ["POLARS_MAX_THREADS"] == "8"
    assert os.environ["RAYON_NUM_THREADS"] == "1"


def test_apply_thread_limits_warns_once_polars_is_imported() -> None:
    """The call is useless after polars builds its pool, so it says so.

    This is the one failure mode of the whole module that's silent
    otherwise: move the call in `cli`/`web_ui.main` below an import that
    pulls polars in and every cap here quietly stops applying.
    """
    assert "polars" in sys.modules, "this test needs polars already imported"
    assert pl.thread_pool_size() >= 1, "polars' pool is built at import time"

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        resource_limits.apply_thread_limits(1)
    finally:
        logger.remove(sink_id)

    assert any("after polars was already imported" in m for m in messages)


@pytest.mark.parametrize(
    "raw",
    ["not-a-number", "0", "-1", ""],
)
def test_env_int_falls_back_on_a_bad_value(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo'd knob in docker-compose.yml must not stop the daemon starting."""
    monkeypatch.setenv("CTS1_TEST_KNOB", raw)
    assert resource_limits.env_int("CTS1_TEST_KNOB", 4) == 4


def test_env_int_reads_a_good_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CTS1_TEST_KNOB", "6")
    assert resource_limits.env_int("CTS1_TEST_KNOB", 4) == 6


def test_defaults_leave_half_the_box_alone() -> None:
    """Whatever machine this runs on, the caps take at most half of it.

    The absolute numbers are a property of the box, so this pins the rule
    instead: nothing here may size itself to every core, since the daemon is
    never the only thing running (see the tuning doc).
    """
    cores = resource_limits.usable_cpu_count()
    thread_ceiling = max(1, cores // 2)
    worker_ceiling = max(2, cores // 2)
    assert thread_ceiling >= resource_limits.DEFAULT_POLARS_THREADS
    assert worker_ceiling >= resource_limits.DEFAULT_DECODER_WORKERS
    # Not scaled with the box at all -- it's SatNOGS's servers on the other
    # end of those sockets, not this machine's cores.
    assert resource_limits.DEFAULT_DEMOD_DOWNLOAD_WORKERS <= 200


@pytest.mark.parametrize(
    ("cores", "expected_threads", "expected_workers"),
    [
        (1, 1, 2),  # minimum=2 keeps one slow observation from stalling the queue
        (2, 1, 2),  # the deployment box: exactly the values it was tuned to
        (4, 2, 2),
        (16, 8, 8),  # a development machine
    ],
)
def test_half_the_cores_scales_with_the_box(
    cores: int,
    expected_threads: int,
    expected_workers: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_affinity(_pid: int) -> set[int]:
        return set(range(cores))

    monkeypatch.setattr(os, "sched_getaffinity", fake_affinity)
    assert resource_limits.half_the_cores() == expected_threads
    assert resource_limits.half_the_cores(minimum=2) == expected_workers


def test_usable_memory_bytes_is_a_plausible_amount() -> None:
    """None is an allowed answer; a nonsense number is not."""
    usable = resource_limits.usable_memory_bytes()
    assert usable is None or 64 * 1024**2 < usable < 1 << 50


@pytest.mark.parametrize(
    ("usable_gib", "expected"),
    [
        (None, "500MB"),  # couldn't tell -- stay conservative
        (3, "500MB"),  # the 3 GB deployment box, floored
        (8, "1073MB"),
        (64, "4000MB"),  # ceiling: no reason to hand DuckDB more than this
    ],
)
def test_duckdb_memory_limit_tracks_the_box(
    usable_gib: int | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        resource_limits,
        "usable_memory_bytes",
        lambda: None if usable_gib is None else usable_gib * 1024**3,
    )
    assert common.default_duckdb_memory_limit() == expected


def test_lower_process_priority_is_a_no_op_at_zero() -> None:
    before = os.nice(0)
    resource_limits.lower_process_priority(0)
    assert os.nice(0) == before
