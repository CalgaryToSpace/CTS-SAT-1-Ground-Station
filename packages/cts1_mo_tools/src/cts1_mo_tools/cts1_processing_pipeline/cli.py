"""CTS-SAT-1 processing pipeline: top-level CLI.

Dispatches to the pipeline's steps as subcommands, with a handful of args
shared by every step (currently just `--db-path`/`--debug`) parsed ahead of
the subcommand.

Usage (uv):
    uv run cts1_processing_pipeline step_1
    uv run cts1_processing_pipeline --db-path output/cts1.duckdb step_1 --norad-id 69015
    uv run cts1_processing_pipeline step_1 --limit 5 --debug
    uv run cts1_processing_pipeline step_2
    uv run cts1_processing_pipeline step_3
"""

from __future__ import annotations

# tyro resolves dataclass field annotations at runtime (via get_type_hints),
# so imports used only in annotations below still need to be real imports.
# ruff: noqa: TC003

__all__ = ["Args", "main"]

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, assert_never

import tyro
from dotenv import load_dotenv
from loguru import logger
from tyro.conf import OmitSubcommandPrefixes

from .step_1_download_and_demodulate import pipeline as step_1_download_and_demodulate
from .step_2_deduplicate_packets import pipeline as step_2_deduplicate_packets
from .step_3_decode_packets import pipeline as step_3_decode_packets

DEFAULT_DB_PATH = step_1_download_and_demodulate.DEFAULT_DB_PATH


@dataclass(frozen=True, slots=True)
class Step1Args:
    """Step 1: list SatNOGS observations, download audio, and decode packets."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    start: str | None = None
    """Only pull observations starting after this point: a duration like
    '3 days' (relative to now) or an ISO 8601 date/datetime. Omit for full
    history."""

    limit: int | None = None
    """Cap the number of observations decoded this run (for testing)."""

    workers: int = 4
    """Concurrency for the decoders (sso_rx_replay, gr_satellites --hexdump,
    gr_satellites --kiss_out, satnogs_data_demod)."""

    temp_dir: Path | None = None
    """Directory to create per-observation temp dirs (audio
    downloads/WAV conversions) under. Defaults to the platform temp
    directory (see Python's `tempfile`)."""

    force_rerun_decoders: bool = False
    """By default, an observation/decoder pair already recorded in the
    decoder_runs table is skipped. Set this to rerun every decoder on every
    candidate observation regardless of that history."""


@dataclass(frozen=True, slots=True)
class Step2Args:
    """Step 2: dedupe raw_packets into distinct_packets_over_time."""


@dataclass(frozen=True, slots=True)
class Step3Args:
    """Step 3: decode distinct_packets_over_time into everything_decoded."""


# Add further steps as additional
# `Annotated[StepNArgs, tyro.conf.subcommand(name="step_n", prefix_name=False)]`
# members below.
Command = (
    Annotated[Step1Args, tyro.conf.subcommand(name="step_1", prefix_name=False)]
    | Annotated[Step2Args, tyro.conf.subcommand(name="step_2", prefix_name=False)]
    | Annotated[Step3Args, tyro.conf.subcommand(name="step_3", prefix_name=False)]
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Args:
    """CTS-SAT-1 processing pipeline."""

    db_path: Path = DEFAULT_DB_PATH
    """DuckDB database file step 1 appends observations/packets to. Later
    steps don't touch the database themselves -- they're parquet-in,
    parquet-out -- but still use this path's directory to find/write the
    parquet files step 1 exports next to it."""

    debug: bool = False
    """Enable debug logging."""

    command: Annotated[Command, OmitSubcommandPrefixes]
    """Which pipeline step to run."""


def main() -> None:
    """Entry point: parse CLI and dispatch to the selected step."""
    load_dotenv()  # picks up SATNOGS_NETWORK_API_KEY for higher API rate limits
    args = tyro.cli(Args)

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    try:
        if isinstance(args.command, Step1Args):
            step_1_download_and_demodulate.run(
                norad_id=args.command.norad_id,
                db_path=args.db_path,
                start=args.command.start,
                limit=args.command.limit,
                workers=args.command.workers,
                temp_dir=args.command.temp_dir,
                force_rerun=args.command.force_rerun_decoders,
            )
        elif isinstance(args.command, Step2Args):
            step_2_deduplicate_packets.run(output_dir=args.db_path.parent)
        elif isinstance(args.command, Step3Args):  # pyright: ignore[reportUnnecessaryIsInstance]
            step_3_decode_packets.run(output_dir=args.db_path.parent)
        else:
            assert_never(args.command)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; exiting.")
        sys.exit(130)  # 128 + SIGINT, the conventional shell exit code


if __name__ == "__main__":
    main()
