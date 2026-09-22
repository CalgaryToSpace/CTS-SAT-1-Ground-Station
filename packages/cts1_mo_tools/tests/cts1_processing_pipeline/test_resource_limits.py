"""Tests for the pipeline's resource caps -- see `resource_limits`."""

import os
import sys
from collections.abc import Iterator

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline import resource_limits
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


def test_defaults_are_conservative() -> None:
    """These sit on a 2-vCPU box shared with the web server (see the tuning doc)."""
    assert resource_limits.DEFAULT_POLARS_THREADS <= 2
    assert resource_limits.DEFAULT_DECODER_WORKERS <= 2
    assert resource_limits.DEFAULT_DEMOD_DOWNLOAD_WORKERS <= 16


def test_lower_process_priority_is_a_no_op_at_zero() -> None:
    before = os.nice(0)
    resource_limits.lower_process_priority(0)
    assert os.nice(0) == before
