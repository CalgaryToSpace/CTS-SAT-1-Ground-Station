"""CTS-SAT-1 SatNOGS processing pipeline.

List every SatNOGS observation for a satellite (paged across all statuses),
land it in DuckDB's `raw_observations`, then for every observation with
audio: download the `.ogg` into a throwaway temp dir, run it through both
`sso_rx_replay --forensics-report` and `gr_satellites` (via `sox`-converted
WAV), and land every decoded frame/PDU in DuckDB's `raw_packets`.

Usage (uv):
    uv run cts1_processing_pipeline
    uv run cts1_processing_pipeline --norad-id 69015 --db-path output/cts1.duckdb
    uv run cts1_processing_pipeline --limit 5 --debug
"""

from __future__ import annotations

__all__ = ["Args", "main", "run"]

import base64
import concurrent.futures
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import tyro
from dotenv import load_dotenv
from loguru import logger

from cts1_mo_tools.cts1_agenda_maker.satnogs_data import fetch_all_observations

from . import _subprocess_registry, db
from .audio import convert_ogg_to_wav, download_audio
from .decode_gr_satellites import DEFAULT_SATCFG_PATH, run_gr_satellites
from .decode_gr_satellites_kiss import run_gr_satellites_kiss
from .decode_sso_rx_replay import run_sso_rx_replay

DEFAULT_DB_PATH = Path("output/cts1_processing_pipeline.duckdb")
CHECKPOINT_INTERVAL = 100
DECODERS = ("sso_rx_replay", "gr_satellites", "gr_satellites_kiss")
KISS_DISPATCH_INTERVAL_SECONDS = 1.0

_DURATION_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>second|minute|hour|day|week)s?\s*$",
    re.IGNORECASE,
)
_DURATION_UNIT_TO_TIMEDELTA_KWARG = {
    "second": "seconds",
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
}


def _parse_start_filter(value: str) -> datetime:
    """Parse --start as either a relative duration or an absolute date/datetime.

    A duration like "3 days" or "6 hours" is measured back from now (UTC).
    Anything else is parsed as ISO 8601 ("2026-08-01" or
    "2026-08-01T00:00:00Z"); a value with no timezone is treated as UTC.

    Raises:
        ValueError: If `value` matches neither form.
    """
    m = _DURATION_RE.match(value)
    if m:
        amount = float(m.group("value"))
        kwarg = _DURATION_UNIT_TO_TIMEDELTA_KWARG[m.group("unit").lower()]
        return datetime.now(UTC) - timedelta(**{kwarg: amount})

    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        msg = (
            f"Could not parse --start={value!r}; expected a duration like "
            f"'3 days' or an ISO 8601 date/datetime."
        )
        raise ValueError(msg) from exc

    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _process_fast(
    obs: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path | None, tempfile.TemporaryDirectory[str] | None]:
    """Download one observation's audio and run the two fast decoders on it.

    sso_rx_replay and gr_satellites --hexdump both finish in a couple of
    seconds, so they run at normal (small pool) concurrency. The temp dir is
    deliberately *not* cleaned up here when a WAV was produced: the caller
    hands wav_path off to a later, separately-paced gr_satellites_kiss pass
    (see _process_kiss) and is responsible for calling tmp.cleanup() once
    that finishes.
    """
    obs_id = obs["id"]
    audio_url = obs["payload"]
    rows: list[dict[str, Any]] = []

    tmp = tempfile.TemporaryDirectory(prefix="cts1_pipeline_")
    try:
        ogg_path = download_audio(audio_url, Path(tmp.name))
    except Exception:  # noqa: BLE001
        logger.exception(f"Failed to download audio for observation {obs_id}")
        tmp.cleanup()
        return rows, None, None

    sso_rows = run_sso_rx_replay(ogg_path, report_filename=ogg_path.name)
    for row in sso_rows:
        data_bytes = base64.b64decode(row["sso_data_base64"])

        rows.append(
            {
                "observation_id": obs_id,
                "decoder": "sso_rx_replay",
                "audio_url": audio_url,
                "ingested_at": datetime.now(UTC),
                # Coalesced columns:
                "data_hex": data_bytes.hex(),
                "data_length_bytes": len(data_bytes),
                "time_in_file_ms": row["sso_time_in_file_ms"],
                "rssi": row["sso_rssi"],
                **row,
            }
        )

    wav_path: Path | None = None
    try:
        wav_path = convert_ogg_to_wav(ogg_path)
        gr_rows = run_gr_satellites(wav_path, satcfg=DEFAULT_SATCFG_PATH)
    except Exception:  # noqa: BLE001
        logger.exception(f"gr_satellites decode failed for observation {obs_id}")
        gr_rows = []
    for row in gr_rows:
        data_bytes = bytes.fromhex(row["gr_pdu_hex"])

        rows.append(
            {
                "observation_id": obs_id,
                "decoder": "gr_satellites",
                "audio_url": audio_url,
                "ingested_at": datetime.now(UTC),
                # Coalesced columns:
                "data_hex": data_bytes.hex(),
                "data_length_bytes": len(data_bytes),
                "time_in_file_ms": None,  # FIXME: Get this from KISS output.
                "rssi": None,
                **row,
            }
        )

    if wav_path is None:
        tmp.cleanup()
        return rows, None, None

    return rows, wav_path, tmp


def _process_kiss(obs: dict[str, Any], wav_path: Path) -> list[dict[str, Any]]:
    """Run gr_satellites_kiss (real-time --throttle) against an already-converted WAV.

    Does coalesce bits too.
    """
    obs_id = obs["id"]
    audio_url = obs["payload"]
    rows: list[dict[str, Any]] = []

    try:
        kiss_rows = run_gr_satellites_kiss(wav_path, satcfg=DEFAULT_SATCFG_PATH)
    except Exception:  # noqa: BLE001
        logger.exception(f"gr_satellites (kiss) decode failed for observation {obs_id}")
        kiss_rows = []
    for row in kiss_rows:
        data_bytes = bytes.fromhex(row["gr_kiss_pdu_hex"])

        rows.append(
            {
                "observation_id": obs_id,
                "decoder": "gr_satellites_kiss",
                "audio_url": audio_url,
                "ingested_at": datetime.now(UTC),
                # Coalesced columns:
                "data_hex": data_bytes.hex(),
                "data_length_bytes": len(data_bytes),
                "time_in_file_ms": row["gr_kiss_time_in_file_ms"],
                "rssi": None,
                **row,
            }
        )

    return rows


def _select_candidates(
    page: list[dict[str, Any]],
    *,
    done: set[tuple[int, str]],
    remaining: int | None,
) -> list[dict[str, Any]]:
    """Observations in `page` that have audio and still need decoding.

    Stops early once `remaining` candidates have been picked (None = no cap).
    """
    candidates: list[dict[str, Any]] = []
    for obs in page:
        if remaining is not None and len(candidates) >= remaining:
            break
        if not obs.get("payload"):
            continue
        if all((obs["id"], decoder) in done for decoder in DECODERS):
            continue
        candidates.append(obs)
    return candidates


def run(  # noqa: C901, PLR0913, PLR0915
    *,
    norad_id: str = "69015",
    db_path: Path = DEFAULT_DB_PATH,
    start: str | None = None,
    page_size: int = 100,
    batch_size: int = 500,
    limit: int | None = None,
    workers: int = 4,
    kiss_workers: int = 200,
) -> None:
    """Run the pipeline once.

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        db_path: DuckDB database file to write raw_observations/raw_packets to.
        start: Only pull observations starting after this point: a duration
            like "3 days" (relative to now) or an ISO 8601 date/datetime.
            None means no lower bound (full history).
        page_size: Observations requested per SatNOGS API page.
        batch_size: Candidate observations accumulated across pages before a
            batch is dispatched to the worker pools. gr_satellites_kiss
            decodes in real time (--throttle), so a large batch is what lets
            hundreds of observations decode concurrently instead of the ~100
            in a single API page.
        limit: Cap the number of observations decoded this run (for testing).
        workers: Concurrency for the fast decoders (sso_rx_replay,
            gr_satellites --hexdump), which each finish in a few seconds.
        kiss_workers: Max concurrency for gr_satellites_kiss, which decodes
            in real time (--throttle) and so needs far more concurrent slots
            to get comparable throughput. New kiss jobs are started at most
            one per second (see KISS_DISPATCH_INTERVAL_SECONDS) so a big
            batch ramps up gradually instead of launching hundreds at once.
    """
    logger.info(f"Listing observations for NORAD {norad_id}")

    start_gt = _parse_start_filter(start) if start is not None else None
    if start_gt is not None:
        logger.info(
            f"  Filtering to observations starting after {start_gt.isoformat()}"
        )

    total_observations = 0
    total_decoded = 0
    since_checkpoint = 0
    reached_limit = False
    pending: list[dict[str, Any]] = []
    last_kiss_dispatch = 0.0

    with (
        db.connect(db_path) as con,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as fast_executor,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=kiss_workers
        ) as kiss_executor,
    ):
        done: set[tuple[int, str]] = db.already_decoded_pairs(con)

        def dispatch_fast(
            batch: list[dict[str, Any]],
        ) -> list[tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str]]]:
            """Run the fast decoders over a batch; write their rows, and
            return (obs, wav_path, tmp) for observations ready for a
            gr_satellites_kiss pass (tmp must be .cleanup()'d by the caller
            once that pass finishes)."""
            nonlocal total_decoded
            ready_for_kiss: list[
                tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str]]
            ] = []
            futures = {fast_executor.submit(_process_fast, obs): obs for obs in batch}
            for future in concurrent.futures.as_completed(futures):
                obs = futures[future]
                try:
                    rows, wav_path, tmp = future.result()
                except Exception:  # noqa: BLE001
                    logger.exception(f"Failed processing observation {obs['id']}")
                    rows, wav_path, tmp = [], None, None
                if rows:
                    db.append_packets(con, pl.DataFrame(rows, infer_schema_length=None))
                done.add((obs["id"], "sso_rx_replay"))
                done.add((obs["id"], "gr_satellites"))
                if wav_path is not None and tmp is not None:
                    ready_for_kiss.append((obs, wav_path, tmp))
                else:
                    # Nothing to feed gr_satellites_kiss; this observation is
                    # already fully processed.
                    done.add((obs["id"], "gr_satellites_kiss"))
                    total_decoded += 1
                    logger.info(f"[{total_decoded}] observation {obs['id']}")
            return ready_for_kiss

        def dispatch_kiss(
            ready: list[tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str]]],
        ) -> None:
            nonlocal total_decoded, last_kiss_dispatch
            kiss_futures: dict[
                concurrent.futures.Future[list[dict[str, Any]]],
                tuple[dict[str, Any], tempfile.TemporaryDirectory[str]],
            ] = {}
            for obs, wav_path, tmp in ready:
                elapsed = time.monotonic() - last_kiss_dispatch
                if elapsed < KISS_DISPATCH_INTERVAL_SECONDS:
                    time.sleep(KISS_DISPATCH_INTERVAL_SECONDS - elapsed)
                last_kiss_dispatch = time.monotonic()
                future = kiss_executor.submit(_process_kiss, obs, wav_path)
                kiss_futures[future] = (obs, tmp)

            for future in concurrent.futures.as_completed(kiss_futures):
                obs, tmp = kiss_futures[future]
                total_decoded += 1
                logger.info(f"[{total_decoded}] observation {obs['id']} (kiss)")
                try:
                    rows = future.result()
                except Exception:  # noqa: BLE001
                    logger.exception(f"Failed kiss-processing observation {obs['id']}")
                    rows = []
                if rows:
                    db.append_packets(con, pl.DataFrame(rows, infer_schema_length=None))
                done.add((obs["id"], "gr_satellites_kiss"))
                tmp.cleanup()

        def dispatch(batch: list[dict[str, Any]]) -> None:
            if not batch:
                return
            logger.info(f"Dispatching a batch of {len(batch)} observation(s)...")
            dispatch_kiss(dispatch_fast(batch))

        try:
            for page in fetch_all_observations(
                norad_id,
                start_gt_filter=start_gt,
                start_lt_filter=datetime.now(UTC),
                statuses=None,
                page_size=page_size,
            ):
                total_observations += len(page)
                since_checkpoint += len(page)
                observations_df = pl.DataFrame(page, infer_schema_length=None)
                db.upsert_observations(con, observations_df)

                if since_checkpoint >= CHECKPOINT_INTERVAL:
                    logger.debug(
                        f"Checkpointing after {total_observations} observation(s)"
                    )
                    con.execute("CHECKPOINT")
                    since_checkpoint = 0

                if reached_limit:
                    continue

                remaining = (
                    None if limit is None else limit - total_decoded - len(pending)
                )
                pending.extend(_select_candidates(page, done=done, remaining=remaining))

                if len(pending) >= batch_size:
                    dispatch(pending)
                    pending = []

                if limit is not None and total_decoded + len(pending) >= limit:
                    # A generator-backed fetch, so breaking here stops
                    # pulling further pages from the API rather than just
                    # skipping them.
                    dispatch(pending)
                    pending = []
                    reached_limit = True
                    break

            dispatch(pending)  # flush whatever's left once paging is exhausted
            pending = []

        except KeyboardInterrupt:
            # Worker threads block inside subprocess.run()-style calls, which
            # a signal can't interrupt from here — kill the children so those
            # threads unblock, then let each executor's own __exit__ join
            # them (fast now that nothing is left running) instead of
            # hanging (the kiss pool especially, mid multi-minute throttle).
            logger.warning("Interrupted; terminating in-flight downloads/decodes...")
            _subprocess_registry.terminate_all()
            fast_executor.shutdown(wait=True, cancel_futures=True)
            kiss_executor.shutdown(wait=True, cancel_futures=True)
            raise

    logger.info(
        f"Done. {total_observations} observation(s) listed, {total_decoded} decoded."
    )


@dataclass(frozen=True, slots=True)
class Args:
    """List SatNOGS observations, download audio, and decode packets to DuckDB."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    db_path: Path = DEFAULT_DB_PATH
    """DuckDB database file for the raw_observations / raw_packets tables."""

    start: str | None = None
    """Only pull observations starting after this point: a duration like
    '3 days' (relative to now) or an ISO 8601 date/datetime. Omit for full
    history."""

    page_size: int = 100
    """Observations requested per SatNOGS API page."""

    batch_size: int = 500
    """Candidate observations accumulated across pages before a batch is
    dispatched to the worker pools, so gr_satellites_kiss's real-time
    (--throttle) decoding can run hundreds concurrently instead of ~page_size
    at a time."""

    limit: int | None = None
    """Cap the number of observations decoded this run (for testing)."""

    workers: int = 4
    """Concurrency for the fast decoders (sso_rx_replay, gr_satellites
    --hexdump)."""

    kiss_workers: int = 200
    """Max concurrency for gr_satellites_kiss (real-time --throttle decode);
    new kiss jobs are started at most one per second regardless of this
    limit, so a big batch ramps up gradually."""

    debug: bool = False
    """Enable debug logging."""


def main() -> None:
    """Entry point: parse CLI and run the pipeline."""
    load_dotenv()  # picks up SATNOGS_NETWORK_API_KEY for higher API rate limits
    args = tyro.cli(Args)

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    try:
        run(
            norad_id=args.norad_id,
            db_path=args.db_path,
            start=args.start,
            page_size=args.page_size,
            batch_size=args.batch_size,
            limit=args.limit,
            workers=args.workers,
            kiss_workers=args.kiss_workers,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; exiting.")
        sys.exit(130)  # 128 + SIGINT, the conventional shell exit code


if __name__ == "__main__":
    main()
