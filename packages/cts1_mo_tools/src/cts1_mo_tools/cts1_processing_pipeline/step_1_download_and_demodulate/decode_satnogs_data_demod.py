"""Downloader for SatNOGS's own already-demodulated packets (`demoddata`).

Unlike the other decoders in this package, this one never touches the
recorded audio: each observation's `demoddata` list carries one
`payload_demod` URL per frame that SatNOGS's own server-side KISS decoder
already produced, and every URL is just a single raw binary packet (no
framing, no extra bytes).

The filename encodes a per-packet timestamp, with an optional trailing
`_<n>` suffix breaking ties between packets landing in the same whole
second, 100ms apart -- not an exact alignment, just a good-enough estimate
of intra-second order. A `_g<n>` suffix is the same idea with an extra 50ms
offset baked in (observed on some SatNOGS stations' filenames):

    .../data_14759295_2026-08-12T19-36-50     -> 19:36:50.000
    .../data_14759295_2026-08-12T19-36-50_1   -> 19:36:50.100
    .../data_14759295_2026-08-12T19-36-50_g1  -> 19:36:50.150
"""

from __future__ import annotations

__all__ = ["parse_demod_filename_time", "run_satnogs_data_demod"]

import concurrent.futures
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from loguru import logger

DECODER_NAME = "satnogs_data_demod"

_FILENAME_RE = re.compile(
    r"data_\d+_(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}-\d{2}-\d{2})"
    r"(?:_(?P<g>g)?(?P<seq>\d+))?$"
)
_SEQ_STEP_MS = 100
_G_OFFSET_MS = 50


def parse_demod_filename_time(url: str) -> datetime | None:
    """Parse a `payload_demod` filename's embedded UTC timestamp.

    Returns None if the filename doesn't match the expected
    `data_<obs_id>_<date>T<HH-MM-SS>[_<n>]` shape.
    """
    filename = url.rsplit("/", 1)[-1]
    m = _FILENAME_RE.search(filename)
    if m is None:
        return None
    base = datetime.strptime(
        f"{m.group('date')}T{m.group('time')}", "%Y-%m-%dT%H-%M-%S"
    ).replace(tzinfo=UTC)
    seq = int(m.group("seq")) if m.group("seq") else 0
    g_offset_ms = _G_OFFSET_MS if m.group("g") else 0
    return base + timedelta(milliseconds=g_offset_ms + seq * _SEQ_STEP_MS)


def _download_one(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        logger.warning(f"satnogs_data_demod: failed to download {url}")
        return None
    return resp.content


def _timestamped_urls(
    demoddata: list[dict[str, Any]],
    *,
    observation_id: int,
) -> dict[str, datetime]:
    """The `payload_demod` URLs worth downloading, each with its packet time.

    Two kinds of entry are dropped rather than downloaded:

      - `.png` files, which are waterfall images SatNOGS has mislabelled as
        demod data -- not packets at all, so silently skipped.
      - anything whose filename has no parseable timestamp, which would leave
        the packet with no `received_at` we could honestly report. Logged as a
        warning, since it means either a new filename shape worth teaching
        `parse_demod_filename_time` or a genuinely odd upload.
    """
    urls: dict[str, datetime] = {}
    for entry in demoddata:
        url = entry.get("payload_demod")
        if not url:
            continue
        if url.lower().endswith(".png"):
            logger.debug(
                f"satnogs_data_demod: observation {observation_id}: skipping "
                f"mislabelled waterfall image {url}"
            )
            continue
        received_at = parse_demod_filename_time(url)
        if received_at is None:
            logger.warning(
                f"satnogs_data_demod: observation {observation_id}: discarding "
                f"packet with no parseable timestamp in its filename: {url}"
            )
            continue
        urls[url] = received_at
    return urls


def run_satnogs_data_demod(
    demoddata: list[dict[str, Any]],
    *,
    observation_id: int,
    max_workers: int = 50,
) -> list[dict[str, Any]]:
    """Download every `payload_demod` URL, one row per packet.

    Spins up its own thread pool sized for a single observation's demoddata
    (which can carry 500+ tiny files) rather than sharing one process-wide;
    callers already run one observation per thread, so this is a pool nested
    inside that thread rather than a pool shared across observations.

    Args:
        demoddata: The observation's `demoddata` list, as returned by the
            SatNOGS API (each entry has a `payload_demod` URL).
        observation_id: The observation these entries belong to, for logging.
        max_workers: Size of the download pool spun up for this call.

    Returns:
        One dict per successfully downloaded packet whose filename carried a
        parseable timestamp.
    """
    timestamped_urls = _timestamped_urls(demoddata, observation_id=observation_id)
    rows: list[dict[str, Any]] = []
    if not timestamped_urls:
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, url): (url, received_at)
            for url, received_at in timestamped_urls.items()
        }
        for future in concurrent.futures.as_completed(futures):
            url, received_at = futures[future]
            data = future.result()
            if data is None:
                continue
            rows.append(
                {
                    "received_at": received_at,
                    "data_hex": data.hex(),
                    "data_length_bytes": len(data),
                    "satnogs_demod_url": url,
                }
            )

    logger.debug(
        f"satnogs_data_demod: {len(rows)}/{len(timestamped_urls)} packet(s) downloaded"
    )
    return rows
