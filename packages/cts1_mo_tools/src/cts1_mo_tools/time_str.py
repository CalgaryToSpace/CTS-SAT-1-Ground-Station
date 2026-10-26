"""Human-readable duration and age strings, shared by the CLI tools and the
processing pipeline's web UI so every "1h 23m" in the project looks alike.
"""

from __future__ import annotations

__all__ = ["age_str", "duration_str"]

from datetime import UTC, datetime, timedelta


def duration_str(duration: timedelta | float) -> str:
    """A duration (a `timedelta`, or a number of seconds) for display, in its
    two largest whole units: e.g. "42s", "16m 5s", "1h 23m" or "2d 5h".
    """
    if isinstance(duration, timedelta):
        duration = duration.total_seconds()
    total_sec = int(duration)
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m {total_sec % 60}s"
    if total_sec < 86400:  # noqa: PLR2004
        return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"
    return f"{total_sec // 86400}d {(total_sec % 86400) // 3600}h"


def age_str(value: datetime, *, now: datetime | None = None) -> str:
    """ "X ago" for a past `value`, "in X" for a future one, with X from
    `duration_str` (a future `value` is legitimate, e.g. the end of a
    SatNOGS observation still in progress). A naive `value` is taken to be
    UTC.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = (now or datetime.now(UTC)) - value
    if delta < timedelta(0):
        return f"in {duration_str(-delta)}"
    return f"{duration_str(delta)} ago"
