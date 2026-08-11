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

import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import tyro
from dotenv import load_dotenv
from loguru import logger

from cts1_mo_tools.cts1_agenda_maker.satnogs_data import (
    OBSERVATION_STATUSES,
    fetch_all_observations,
)

from . import db
from .audio import convert_ogg_to_wav, download_audio
from .decode_gr_satellites import DEFAULT_SATCFG_PATH, run_gr_satellites
from .decode_sso_rx_replay import run_sso_rx_replay

DEFAULT_DB_PATH = Path("output/cts1_processing_pipeline.duckdb")


def _process_observation(obs: dict[str, Any], *, satcfg: Path) -> list[dict[str, Any]]:
    """Download one observation's audio and run both decoders on it."""
    obs_id = obs["id"]
    audio_url = obs["payload"]
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="cts1_pipeline_") as tmp_str:
        tmp_dir = Path(tmp_str)
        try:
            ogg_path = download_audio(audio_url, tmp_dir)
        except Exception:  # noqa: BLE001
            logger.exception(f"Failed to download audio for observation {obs_id}")
            return rows

        sso_rows = run_sso_rx_replay(ogg_path, report_filename=ogg_path.name)
        for row in sso_rows:
            rows.append(  # noqa: PERF401
                {
                    "observation_id": obs_id,
                    "decoder": "sso_rx_replay",
                    "audio_url": audio_url,
                    "ingested_at": datetime.now(UTC),
                    **row,
                }
            )

        try:
            wav_path = convert_ogg_to_wav(ogg_path)
            gr_rows = run_gr_satellites(wav_path, satcfg=satcfg)
        except Exception:  # noqa: BLE001
            logger.exception(f"gr_satellites decode failed for observation {obs_id}")
            gr_rows = []
        for row in gr_rows:
            rows.append(  # noqa: PERF401
                {
                    "observation_id": obs_id,
                    "decoder": "gr_satellites",
                    "audio_url": audio_url,
                    "ingested_at": datetime.now(UTC),
                    **row,
                }
            )

    return rows


def run(  # noqa: PLR0913
    *,
    norad_id: str = "69015",
    db_path: Path = DEFAULT_DB_PATH,
    satcfg: Path = DEFAULT_SATCFG_PATH,
    statuses: tuple[str, ...] = OBSERVATION_STATUSES,
    page_size: int = 100,
    limit: int | None = None,
    skip_packets: bool = False,
) -> None:
    """Run the pipeline once.

    Args:
        norad_id: NORAD catalog ID of the target satellite.
        db_path: DuckDB database file to write raw_observations/raw_packets to.
        satcfg: gr_satellites satellite YAML definition.
        statuses: SatNOGS observation statuses to list.
        page_size: Observations requested per API page.
        limit: Cap the number of observations decoded this run (for testing).
        skip_packets: Only sync raw_observations; skip audio download/decode.
    """
    logger.info(f"Listing observations for NORAD {norad_id} (statuses={statuses})")

    con = db.connect(db_path)
    done: set[tuple[int, str]] = (
        set() if skip_packets else db.already_decoded_pairs(con)
    )

    total_observations = 0
    total_decoded = 0
    reached_limit = False

    for page in fetch_all_observations(
        norad_id, statuses=statuses, page_size=page_size
    ):
        total_observations += len(page)
        db.upsert_observations(con, pl.DataFrame(page, infer_schema_length=None))

        if skip_packets or reached_limit:
            continue

        for obs in page:
            if not obs.get("payload"):
                continue
            key_sso, key_gr = (obs["id"], "sso_rx_replay"), (obs["id"], "gr_satellites")
            if key_sso in done and key_gr in done:
                continue

            total_decoded += 1
            logger.info(f"[{total_decoded}] observation {obs['id']}")
            rows = _process_observation(obs, satcfg=satcfg)
            if rows:
                db.append_packets(con, pl.DataFrame(rows, infer_schema_length=None))
            done.add(key_sso)
            done.add(key_gr)

            if limit is not None and total_decoded >= limit:
                # A generator-backed fetch, so breaking here stops pulling
                # further pages from the API rather than just skipping them.
                reached_limit = True
                break

        if reached_limit:
            break

    con.close()
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

    satcfg: Path = DEFAULT_SATCFG_PATH
    """gr_satellites satellite YAML definition."""

    page_size: int = 100
    """Observations requested per SatNOGS API page."""

    limit: int | None = None
    """Cap the number of observations decoded this run (for testing)."""

    skip_packets: bool = False
    """Only sync raw_observations; skip audio download and decoding."""

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

    run(
        norad_id=args.norad_id,
        db_path=args.db_path,
        satcfg=args.satcfg,
        page_size=args.page_size,
        limit=args.limit,
        skip_packets=args.skip_packets,
    )


if __name__ == "__main__":
    main()
