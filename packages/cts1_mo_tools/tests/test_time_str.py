from datetime import UTC, datetime, timedelta

from cts1_mo_tools.time_str import age_str, duration_str


def test_duration_str() -> None:
    assert duration_str(42) == "42s"
    assert duration_str(42.9) == "42s"
    assert duration_str(16 * 60 + 50) == "16m 50s"
    assert duration_str(3600 + 23 * 60 + 5) == "1h 23m"
    assert duration_str(2 * 86400 + 5 * 3600) == "2d 5h"
    assert duration_str(timedelta(minutes=16, seconds=5)) == "16m 5s"


def test_age_str() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert age_str(now - timedelta(minutes=5), now=now) == "5m 0s ago"
    assert age_str(now + timedelta(hours=2), now=now) == "in 2h 0m"
    # Naive datetimes are taken to be UTC.
    naive = datetime(2026, 1, 1, 11, 59, 30)  # noqa: DTZ001
    assert age_str(naive, now=now) == "30s ago"
