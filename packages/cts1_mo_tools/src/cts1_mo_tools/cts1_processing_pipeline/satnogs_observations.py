"""Paginated SatNOGS Network observation listing.

Unlike :mod:`cts1_mo_tools.cts1_agenda_maker.satnogs_data`, which only queries
``status=future`` observations for scheduling purposes, this module lists
*all* observations for a satellite (any status), following the API's cursor
pagination via the ``Link`` response header.
"""

from __future__ import annotations

__all__ = ["OBSERVATION_STATUSES", "fetch_all_observations", "iter_observation_pages"]

import os
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import requests
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

SATNOGS_BASE: Final = "https://network.satnogs.org/api"
_LINK_NEXT_RE: Final = re.compile(r'<([^>]+)>;\s*rel="next"')
_SATNOGS_API_DATETIME_FMT: Final = "%Y-%m-%dT%H:%M:%S"

# The SatNOGS API does not reliably return every status when the `status`
# filter is omitted, so an unfiltered listing is built by querying each of
# these explicitly and concatenating the results.
OBSERVATION_STATUSES: Final = ("good", "bad", "failed", "unknown", "future")


def _auth_headers() -> dict[str, str]:
    api_key = os.environ.get("SATNOGS_NETWORK_API_KEY")
    return {"Authorization": f"Token {api_key}"} if api_key else {}


def _next_url_from_headers(headers: Any) -> str | None:
    link = headers.get("Link", "")
    m = _LINK_NEXT_RE.search(link)
    return m.group(1) if m else None


def iter_observation_pages(  # noqa: PLR0913
    norad_cat_id: str,
    *,
    status: str | None = None,
    start_gt: datetime | None = None,
    start_lt: datetime | None = None,
    end_gt: datetime | None = None,
    page_size: int = 100,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of observations for a satellite, following cursor pagination.

    Args:
        norad_cat_id: NORAD catalog ID of the target satellite.
        status: SatNOGS observation status filter (``good``, ``bad``,
            ``failed``, ``unknown``, ``future``), or ``None`` for unfiltered.
        start_gt: Optional lower bound on observation start time.
        start_lt: Optional upper bound on observation start time.
        end_gt: Optional lower bound on observation end time (``end__gt``).
        page_size: Max observations requested per page.
    """
    url: str | None = f"{SATNOGS_BASE}/observations/"
    params: dict[str, Any] = {
        "norad_cat_id": norad_cat_id,
        "format": "json",
        "page_size": page_size,
    }
    if status is not None:
        params["status"] = status
    if start_gt is not None:
        params["start"] = start_gt.astimezone(UTC).strftime(_SATNOGS_API_DATETIME_FMT)
    if start_lt is not None:
        params["start__lt"] = start_lt.astimezone(UTC).strftime(
            _SATNOGS_API_DATETIME_FMT
        )
    if end_gt is not None:
        params["end__gt"] = end_gt.astimezone(UTC).strftime(_SATNOGS_API_DATETIME_FMT)

    headers = _auth_headers()

    while url is not None:
        r = _get_with_retry(url, params=params, headers=headers)
        page: list[dict[str, Any]] = r.json()
        if page:
            yield page
        url = _next_url_from_headers(r.headers)
        params = {}  # cursor URL already encodes all query params


_MAX_RETRIES: Final = 6


def _get_with_retry(
    url: str, *, params: dict[str, Any], headers: dict[str, str]
) -> requests.Response:
    """GET with retry-with-backoff on 429, honoring Retry-After when present."""
    for attempt in range(_MAX_RETRIES):
        logger.debug(f"GET {url} params={params}")
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code != requests.codes.too_many_requests:
            r.raise_for_status()
            return r

        retry_after = r.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else 2.0**attempt
        logger.warning(
            f"429 from SatNOGS API (attempt {attempt + 1}/{_MAX_RETRIES}); "
            f"sleeping {delay:.1f}s"
        )
        time.sleep(delay)

    msg = f"Exceeded {_MAX_RETRIES} retries against SatNOGS API: {url}"
    raise RuntimeError(msg)


def fetch_all_observations(  # noqa: PLR0913
    norad_cat_id: str,
    *,
    statuses: tuple[str, ...] | None = OBSERVATION_STATUSES,
    start_gt: datetime | None = None,
    start_lt: datetime | None = None,
    end_gt: datetime | None = None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Fetch every observation for a satellite, across every requested status.

    Args:
        norad_cat_id: NORAD catalog ID of the target satellite.
        statuses: Status values to query and concatenate; pass ``None`` to
            issue a single unfiltered request instead.
        start_gt: Optional lower bound on observation start time.
        start_lt: Optional upper bound on observation start time.
        end_gt: Optional lower bound on observation end time.
        page_size: Max observations requested per page.

    Returns:
        Flat list of raw observation dicts, deduplicated by ``id``.
    """
    by_id: dict[int, dict[str, Any]] = {}
    for status in statuses if statuses is not None else (None,):
        count_before = len(by_id)
        for page in iter_observation_pages(
            norad_cat_id,
            status=status,
            start_gt=start_gt,
            start_lt=start_lt,
            end_gt=end_gt,
            page_size=page_size,
        ):
            for obs in page:
                by_id[obs["id"]] = obs
        logger.info(
            f"  status={status!r}: {len(by_id) - count_before} observation(s)"
        )

    return list(by_id.values())
