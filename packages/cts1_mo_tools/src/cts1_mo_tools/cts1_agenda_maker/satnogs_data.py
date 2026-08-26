"""SatNOGS Network API helpers."""

import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Final

import requests
from loguru import logger

SATNOGS_BASE = "https://network.satnogs.org/api"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')
_SATNOGS_API_DATETIME_REQUEST_FORMAT = "%Y-%m-%dT%H:%M:%S"
_MAX_RETRIES: Final = 6

# The SatNOGS API does not reliably return every status when the `status`
# filter is omitted, so an unfiltered listing is built by querying each of
# these explicitly and concatenating the results (see fetch_all_observations).
OBSERVATION_STATUSES: Final = ("good", "bad", "failed", "unknown", "future")


def _get_auth_headers() -> dict[str, str]:
    """Return auth headers if SATNOGS_NETWORK_API_KEY is set, else empty dict."""
    api_key = os.environ.get("SATNOGS_NETWORK_API_KEY")
    return {"Authorization": f"Token {api_key}"} if api_key else {}


def _next_url_from_headers(headers: Any) -> str | None:
    """Extract the next-page URL from a Link header, or None if absent."""
    link = headers.get("Link", "")
    m = _LINK_NEXT_RE.search(link)
    return m.group(1) if m else None


def _get_with_retry(
    url: str, *, params: dict[str, Any], headers: dict[str, str]
) -> requests.Response:
    """GET with retry-with-backoff on 429/500, honoring Retry-After when present."""
    for attempt in range(_MAX_RETRIES):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code not in (
            requests.codes.too_many_requests,
            requests.codes.internal_server_error,
        ):
            r.raise_for_status()
            return r

        retry_after = r.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else 2.0**attempt
        logger.warning(
            f"{r.status_code} from SatNOGS API (attempt {attempt + 1}/{_MAX_RETRIES}); "
            f"sleeping {delay:.1f}s (Retry-After: {retry_after})"
        )
        time.sleep(delay)

    msg = f"Exceeded {_MAX_RETRIES} retries against SatNOGS API: {url}"
    raise RuntimeError(msg)


def iter_observation_pages(  # noqa: PLR0913
    norad_cat_id: str,
    *,
    status: str | None = "future",
    start_gt_filter: datetime | None = None,
    start_lt_filter: datetime | None = None,
    end_gt_filter: datetime | None = None,
    page_size: int = 100,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of observations one at a time, following cursor pagination.

    The SatNOGS Network API returns a plain JSON array per page; the next-page
    URL is carried in the HTTP ``Link: <url>; rel="next"`` response header.

    Auth is optional: if the ``SATNOGS_NETWORK_API_KEY`` environment variable is
    set, it is sent as a ``Token`` bearer on every request.

    Args:
        norad_cat_id: NORAD catalog ID of the satellite.
        status: SatNOGS observation status filter (``good``, ``bad``,
            ``failed``, ``unknown``, ``future``), or ``None`` for unfiltered.
        start_gt_filter: Optional lower bound on observation start time.
        start_lt_filter: Optional upper bound on observation start time.
        end_gt_filter: Optional lower bound on observation end time (``end__gt``).
        page_size: Max observations per page requested.
    """
    url: str | None = f"{SATNOGS_BASE}/observations/"
    params: dict[str, Any] = {
        "norad_cat_id": norad_cat_id,
        "format": "json",
        "page_size": page_size,
    }
    if status is not None:
        params["status"] = status
    if start_gt_filter is not None:
        params["start"] = start_gt_filter.astimezone(UTC).strftime(
            _SATNOGS_API_DATETIME_REQUEST_FORMAT
        )
    if start_lt_filter is not None:
        params["start__lt"] = start_lt_filter.astimezone(UTC).strftime(
            _SATNOGS_API_DATETIME_REQUEST_FORMAT
        )
    if end_gt_filter is not None:
        params["end__gt"] = end_gt_filter.astimezone(UTC).strftime(
            _SATNOGS_API_DATETIME_REQUEST_FORMAT
        )

    headers = _get_auth_headers()

    while url is not None:
        r = _get_with_retry(url, params=params, headers=headers)
        page: list[dict[str, Any]] = r.json()
        assert isinstance(page, list), f"expected list, got {type(page)}"
        if page:
            yield page
        url = _next_url_from_headers(r.headers)
        params = {}  # cursor URL already encodes all query params


def iter_future_observation_pages(
    norad_cat_id: str,
    *,
    start_gt_filter: datetime | None = None,
    start_lt_filter: datetime | None = None,
    end_gt_filter: datetime | None = None,
    page_size: int = 100,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of future observations. Thin wrapper for `iter_observation_pages`."""
    yield from iter_observation_pages(
        norad_cat_id,
        status="future",
        start_gt_filter=start_gt_filter,
        start_lt_filter=start_lt_filter,
        end_gt_filter=end_gt_filter,
        page_size=page_size,
    )


def fetch_all_observations(  # noqa: PLR0913
    norad_cat_id: str,
    *,
    statuses: tuple[str, ...] | None = OBSERVATION_STATUSES,
    start_gt_filter: datetime | None = None,
    start_lt_filter: datetime | None = None,
    end_gt_filter: datetime | None = None,
    page_size: int = 100,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of observations across every requested status, one at a time.

    There can be far more observations than fit comfortably in memory, so this
    streams pages straight from the API instead of collecting a full list:
    callers should land each page as it arrives rather than waiting for the
    whole satellite history to be fetched first.

    Args:
        norad_cat_id: NORAD catalog ID of the target satellite.
        statuses: Status values to query in turn; pass ``None`` to issue a
            single unfiltered request instead.
        start_gt_filter: Optional lower bound on observation start time.
        start_lt_filter: Optional upper bound on observation start time.
        end_gt_filter: Optional lower bound on observation end time.
        page_size: Max observations requested per page.

    Yields:
        Pages (lists of raw observation dicts) as they arrive from the API.
    """
    for status in statuses if statuses is not None else (None,):
        page_count = 0
        obs_count = 0
        for page in iter_observation_pages(
            norad_cat_id,
            status=status,
            start_gt_filter=start_gt_filter,
            start_lt_filter=start_lt_filter,
            end_gt_filter=end_gt_filter,
            page_size=page_size,
        ):
            page_count += 1
            obs_count += len(page)
            yield page
        logger.info(
            f"  status={status!r}: {obs_count} observation(s) across "
            f"{page_count} page(s)"
        )
