from datetime import UTC, datetime, timedelta

from cts1_mo_tools.cts1_processing_pipeline.web_ui.overpasses import Overpass
from cts1_mo_tools.cts1_processing_pipeline.web_ui.overpasses_page import (
    _due_notifications,  # pyright: ignore[reportPrivateUsage]
)

AOS = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
PASS = Overpass(
    aos=AOS,
    los=AOS + timedelta(minutes=10),
    max_elevation_at=AOS + timedelta(minutes=5),
    max_elevation_deg=45.0,
    aos_azimuth_deg=0.0,
    los_azimuth_deg=180.0,
)
LEAD = timedelta(minutes=5)


def _kinds(now: datetime, sent: set[tuple[datetime, str]] | None = None) -> list[str]:
    return [
        kind
        for _key, _p, kind in _due_notifications(
            [PASS], [AOS], LEAD, now, sent or set()
        )
    ]


def test_due_at_lead_and_aos() -> None:
    assert _kinds(AOS - LEAD - timedelta(seconds=1)) == []
    assert _kinds(AOS - LEAD) == ["lead"]
    assert _kinds(AOS - LEAD + timedelta(seconds=59)) == ["lead"]
    assert _kinds(AOS - LEAD + timedelta(minutes=1)) == []  # Stale.
    assert _kinds(AOS) == ["aos"]


def test_not_resent() -> None:
    assert _kinds(AOS, {(AOS, "aos")}) == []


def test_unticked_pass_not_notified() -> None:
    assert _due_notifications([PASS], [], LEAD, AOS, set()) == []


def test_ticked_pass_matched_after_small_aos_shift() -> None:
    # e.g. re-predicted from a newer TLE; keyed by the originally ticked AOS.
    ticked = AOS - timedelta(seconds=20)
    due = _due_notifications([PASS], [ticked], LEAD, AOS, set())
    assert [key for key, _p, _kind in due] == [(ticked, "aos")]


def test_zero_lead_only_notifies_at_aos() -> None:
    due = _due_notifications([PASS], [AOS], timedelta(0), AOS, set())
    assert [kind for _key, _p, kind in due] == ["aos"]
