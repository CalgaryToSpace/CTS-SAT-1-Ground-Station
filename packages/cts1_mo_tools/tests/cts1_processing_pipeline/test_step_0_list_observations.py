"""Tests for step 0's listing windows, its listing history in DuckDB, and
step 1 reading the listed observations back out.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cts1_mo_tools.cts1_processing_pipeline import landing_db
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations import (
    db as step_0_db,
)
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations import (
    pipeline as step_0_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_0_list_observations.pipeline import (
    LISTING_WINDOW_OVERLAP,
    REFETCH_SETTLE_PERIOD,
    ListingWindow,
    listing_windows,
    windows_needing_listing,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    db as step_1_db,
)

_DAY = datetime(2026, 9, 1, tzinfo=UTC)


def _api_observation(obs_id: int, start: datetime, **overrides: Any) -> dict[str, Any]:
    """A trimmed-down observation, shaped like the SatNOGS API returns it."""
    return {
        "id": obs_id,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (start + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "norad_cat_id": 69015,
        "payload": f"https://example.invalid/satnogs_{obs_id}.ogg",
        "demoddata": [],
        "status": "good",
        **overrides,
    }


# ---------------------------------------------------------------------------
# Listing windows.
# ---------------------------------------------------------------------------


def test_windows_are_aligned_to_midnight_and_noon_newest_first() -> None:
    windows = listing_windows(
        _DAY + timedelta(hours=5), now=_DAY + timedelta(days=1, hours=1)
    )

    assert [w.start for w in windows] == [
        _DAY + timedelta(days=1),
        _DAY + timedelta(hours=12),
        _DAY,
    ]
    assert windows[-1].end == _DAY + timedelta(hours=12)


def test_query_bounds_overlap_both_neighbours_but_stop_at_now() -> None:
    window = ListingWindow(_DAY)

    assert window.query_bounds(now=_DAY + timedelta(days=2)) == (
        _DAY - timedelta(minutes=25),
        _DAY + timedelta(hours=12, minutes=25),
    )
    now = _DAY + timedelta(hours=3)
    assert window.query_bounds(now=now) == (_DAY - timedelta(minutes=25), now)


def test_only_unlisted_or_unsettled_windows_need_listing() -> None:
    windows = [ListingWindow(_DAY + timedelta(hours=12 * n)) for n in (3, 2, 1, 0)]
    flags = {
        windows[1].start: True,  # listed, but still settling
        windows[2].start: False,  # listed for good
    }

    assert windows_needing_listing(windows, flags) == [
        windows[0],
        windows[1],
        windows[3],
    ]
    assert windows_needing_listing(windows, flags, refetch_all=True) == windows


# ---------------------------------------------------------------------------
# Listing a window into DuckDB.
# ---------------------------------------------------------------------------


@pytest.fixture
def con(tmp_path: Path) -> Iterator[Any]:
    with landing_db.connect(tmp_path / landing_db.DB_FILENAME) as connection:
        yield connection


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


def test_listing_a_settled_window_lands_observations_and_history(
    con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = ListingWindow(_DAY)
    observations = [
        # Caught by the overlap before the window starts...
        _api_observation(1, _DAY - timedelta(minutes=20)),
        _api_observation(2, _DAY + timedelta(hours=6)),
        # ...and after it ends.
        _api_observation(3, window.end + timedelta(minutes=20)),
        # Outside the overlap: the neighbouring window's job.
        _api_observation(4, window.end + timedelta(hours=1)),
    ]
    _fake_api(monkeypatch, observations)

    record = step_0_pipeline.list_window(
        con, norad_id="69015", window=window, now=_DAY + timedelta(days=2)
    )
    step_0_db.record_listing(con, record)

    assert record.succeeded
    assert record.observation_count == 3
    assert not record.needs_refetch
    assert step_0_db.window_refetch_flags(con) == {window.start: False}
    ids = con.execute("SELECT id FROM raw_observations ORDER BY id").fetchall()
    assert ids == [(1,), (2,), (3,)]


def test_a_window_listed_before_it_settles_needs_relisting(
    con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both while the window is still open, and for `REFETCH_SETTLE_PERIOD`
    after it closes -- observations near its end may not have uploaded yet.
    """
    window = ListingWindow(datetime.now(UTC) - timedelta(hours=12, minutes=30))
    _fake_api(monkeypatch, [])
    assert window.settled_at == (
        window.end + LISTING_WINDOW_OVERLAP + REFETCH_SETTLE_PERIOD
    )

    record = step_0_pipeline.list_window(
        con, norad_id="69015", window=window, now=datetime.now(UTC)
    )
    step_0_db.record_listing(con, record)

    assert record.succeeded
    assert record.needs_refetch
    assert step_0_db.window_refetch_flags(con) == {window.start: True}


def test_a_failed_listing_is_logged_but_not_recorded_as_listed(
    con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_fetch(*_args: Any, **_kwargs: Any) -> Iterator[list[dict[str, Any]]]:
        msg = "SatNOGS is down"
        raise RuntimeError(msg)
        yield []  # pragma: no cover

    monkeypatch.setattr(step_0_pipeline, "fetch_all_observations", broken_fetch)

    record = step_0_pipeline.list_window(
        con, norad_id="69015", window=ListingWindow(_DAY), now=_DAY + timedelta(days=2)
    )
    step_0_db.record_listing(con, record)

    assert not record.succeeded
    assert record.error == "RuntimeError: SatNOGS is down"
    assert step_0_db.window_refetch_flags(con) == {}
    history = con.execute(
        "SELECT succeeded, error FROM observation_listing_history"
    ).fetchall()
    assert history == [(False, "RuntimeError: SatNOGS is down")]


def test_relisting_a_window_updates_it_in_place(
    con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = ListingWindow(_DAY)
    observations = [_api_observation(1, _DAY + timedelta(hours=1))]
    _fake_api(monkeypatch, observations)
    for _ in range(2):
        step_0_db.record_listing(
            con,
            step_0_pipeline.list_window(
                con, norad_id="69015", window=window, now=_DAY + timedelta(days=2)
            ),
        )

    windows = con.execute(
        "SELECT listing_count, last_observation_count FROM observation_listing_windows"
    ).fetchall()
    assert windows == [(2, 1)]
    history_count = con.execute(
        "SELECT count(*) FROM observation_listing_history"
    ).fetchone()
    assert history_count == (2,)
    obs_count = con.execute("SELECT count(*) FROM raw_observations").fetchone()
    assert obs_count == (1,)


def test_run_skips_settled_windows_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[datetime, datetime]] = []
    _fake_api(monkeypatch, [], calls)
    start = (datetime.now(UTC) - timedelta(days=3)).isoformat()

    step_0_pipeline.run(data_dir=tmp_path, start=start)
    first_run_calls = len(calls)
    calls.clear()
    step_0_pipeline.run(data_dir=tmp_path, start=start)

    # 3 days back, rounded down to a window boundary, plus the current one.
    assert first_run_calls in (7, 8)
    # Only the windows still settling get listed again.
    assert 1 <= len(calls) <= 2
    assert (tmp_path / "observation_listing_windows.parquet").exists()
    assert (tmp_path / "observation_listing_history.parquet").exists()


# ---------------------------------------------------------------------------
# Step 1 reading step 0's listing back out.
# ---------------------------------------------------------------------------


def test_step_1_loads_listed_observations_newest_first(
    con: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    demoddata = [{"payload_demod": "https://example.invalid/data_2_x"}]
    observations = [
        _api_observation(1, _DAY + timedelta(hours=1)),
        _api_observation(2, _DAY + timedelta(hours=2), demoddata=demoddata),
        _api_observation(3, _DAY + timedelta(hours=3), norad_cat_id=12345),
    ]
    _fake_api(monkeypatch, observations)
    step_0_pipeline.list_window(
        con, norad_id="69015", window=ListingWindow(_DAY), now=_DAY + timedelta(days=2)
    )

    loaded = step_1_db.load_observations(con, norad_id="69015")

    assert [obs["id"] for obs in loaded] == [2, 1]
    assert loaded[0] == {
        "id": 2,
        "start": _DAY + timedelta(hours=2),
        "end": _DAY + timedelta(hours=2, minutes=10),
        "payload": "https://example.invalid/satnogs_2.ogg",
        "demoddata": demoddata,
    }
    assert loaded[1]["demoddata"] == []
    assert loaded[1]["start"].tzinfo is UTC

    since = step_1_db.load_observations(
        con, norad_id="69015", start_gte=_DAY + timedelta(hours=2)
    )
    assert [obs["id"] for obs in since] == [2]
