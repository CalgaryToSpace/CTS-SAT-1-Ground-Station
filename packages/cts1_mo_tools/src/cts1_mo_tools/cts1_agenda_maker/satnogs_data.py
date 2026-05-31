"""SatNOGS Network API helpers."""

import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import requests

SATNOGS_BASE = "https://network.satnogs.org/api"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')
_SATNOGS_API_DATETIME_REQUEST_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _get_auth_headers() -> dict[str, str]:
    """Return auth headers if SATNOGS_NETWORK_API_KEY is set, else empty dict."""
    api_key = os.environ.get("SATNOGS_NETWORK_API_KEY")
    return {"Authorization": f"Token {api_key}"} if api_key else {}


def _next_url_from_headers(headers: Any) -> str | None:
    """Extract the next-page URL from a Link header, or None if absent."""
    link = headers.get("Link", "")
    m = _LINK_NEXT_RE.search(link)
    return m.group(1) if m else None


def iter_future_observation_pages(
    norad_cat_id: str,
    start_lt_filter: datetime | None = None,
    end_gt_filter: datetime | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of future observations one at a time, following cursor pagination.

    The SatNOGS Network API returns a plain JSON array per page; the next-page
    URL is carried in the HTTP ``Link: <url>; rel="next"`` response header.

    Auth is optional: if the ``SATNOGS_NETWORK_API_KEY`` environment variable is
    set, it is sent as a ``Token`` bearer on every request.

    Args:
        norad_cat_id: NORAD catalog ID of the satellite.
        start_lt_filter: Optional upper bound on observation start time.
        end_gt_filter: Optional lower bound on observation end time (``end__gt``).
    """
    url: str | None = f"{SATNOGS_BASE}/observations/"
    params: dict[str, Any] = {
        "norad_cat_id": norad_cat_id,
        "status": "future",
        "format": "json",
        "page_size": 100,
    }
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
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        page: list[dict[str, Any]] = r.json()
        assert isinstance(page, list), f"expected list, got {type(page)}"
        if page:
            yield page
        url = _next_url_from_headers(r.headers)
        params = {}  # cursor URL already encodes all query params
