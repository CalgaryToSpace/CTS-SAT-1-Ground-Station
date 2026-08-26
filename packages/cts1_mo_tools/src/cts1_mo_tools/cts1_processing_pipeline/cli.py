"""CTS-SAT-1 processing pipeline: top-level CLI.

Dispatches to the pipeline's steps as subcommands, with a handful of args
shared by every step (currently just `--data-dir`/`--debug`) parsed ahead of
the subcommand. `--data-dir` is the one directory every step reads/writes
its DuckDB database and parquet files in -- see each step's own
`DEFAULT_DATA_DIR`/`OUTPUT_FILENAME` for the fixed filename it looks for.

Usage (uv):
    uv run cts1_processing_pipeline step_1
    uv run cts1_processing_pipeline --data-dir output step_1 --norad-id 69015
    uv run cts1_processing_pipeline step_1 --limit 5 --debug
    uv run cts1_processing_pipeline step_2
    uv run cts1_processing_pipeline step_3
    uv run cts1_processing_pipeline step_4
    uv run cts1_processing_pipeline daemon
    uv run cts1_processing_pipeline daemon --start "3 days" --interval 5
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

from . import daemon
from .step_1_download_and_demodulate import pipeline as step_1_download_and_demodulate
from .step_2_deduplicate_packets import pipeline as step_2_deduplicate_packets
from .step_3_decode_packets import pipeline as step_3_decode_packets
from .step_4_detect_satellite_events import pipeline as step_4_detect_satellite_events

DEFAULT_DATA_DIR = step_1_download_and_demodulate.DEFAULT_DATA_DIR


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

    tools: tuple[str, ...] | None = None
    """Which decoders to run, from {askew_demod_from_file, sso_rx_replay,
    gr_satellites_pdu, gr_satellites_kiss, satnogs_data_demod}. Omit to run
    all of them."""


@dataclass(frozen=True, slots=True)
class Step2Args:
    """Step 2: dedupe raw_packets into distinct_packets_over_time."""


@dataclass(frozen=True, slots=True)
class Step3Args:
    """Step 3: decode distinct_packets_over_time into everything_decoded."""


@dataclass(frozen=True, slots=True)
class Step4Args:
    """Step 4: detect satellite events (reboots, uplinks) from beacon
    counters into satellite_events_from_beacons."""


@dataclass(frozen=True, slots=True)
class DaemonArgs:
    """Daemon: run steps 1-4 continuously -- an initial backfill of
    `--start`, then a periodic requery + full steps 1-4 rerun every
    `--interval` minutes."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    start: str = "24 hours"
    """How far back the initial backfill reaches: a duration like '3 days'
    (relative to now) or an ISO 8601 date/datetime -- same syntax as step
    1's own --start."""

    interval: float = 15.0
    """Minutes between requeries. Each requery re-pulls observations
    starting in the trailing (interval + 30) minutes -- the 30-minute
    overlap catches a SatNOGS observation still uploading/being vetted
    during the previous poll -- then reruns steps 2 and 3."""

    limit: int | None = None
    """Cap the number of observations decoded per step-1 run (for testing)."""

    workers: int = 4
    """Concurrency for the decoders, same as step 1."""

    temp_dir: Path | None = None
    """Directory to create per-observation temp dirs under, same as step 1."""

    force_rerun_decoders: bool = False
    """Same as step 1's --force-rerun-decoders, applied to every requery."""

    tools: tuple[str, ...] | None = None
    """Same as step 1's --tools, applied to every requery."""


# Add further steps as additional
# `Annotated[StepNArgs, tyro.conf.subcommand(name="step_n", prefix_name=False)]`
# members below.
Command = (
    Annotated[Step1Args, tyro.conf.subcommand(name="step_1", prefix_name=False)]
    | Annotated[Step2Args, tyro.conf.subcommand(name="step_2", prefix_name=False)]
    | Annotated[Step3Args, tyro.conf.subcommand(name="step_3", prefix_name=False)]
    | Annotated[Step4Args, tyro.conf.subcommand(name="step_4", prefix_name=False)]
    | Annotated[DaemonArgs, tyro.conf.subcommand(name="daemon", prefix_name=False)]
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Args:
    """CTS-SAT-1 processing pipeline."""

    data_dir: Path = DEFAULT_DATA_DIR
    """Directory every step reads/writes its files in: step 1's DuckDB
    database (and the parquet files it checkpoints from that), and every
    later step's parquet-in-parquet-out file -- each step finds its own
    file(s) by a fixed filename inside this one directory."""

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
                data_dir=args.data_dir,
                start=args.command.start,
                limit=args.command.limit,
                workers=args.command.workers,
                temp_dir=args.command.temp_dir,
                force_rerun=args.command.force_rerun_decoders,
                tools=args.command.tools,
            )
        elif isinstance(args.command, Step2Args):
            step_2_deduplicate_packets.run(data_dir=args.data_dir)
        elif isinstance(args.command, Step3Args):
            step_3_decode_packets.run(data_dir=args.data_dir)
        elif isinstance(args.command, Step4Args):
            step_4_detect_satellite_events.run(data_dir=args.data_dir)
        elif isinstance(args.command, DaemonArgs):  # pyright: ignore[reportUnnecessaryIsInstance]
            daemon.run(
                norad_id=args.command.norad_id,
                data_dir=args.data_dir,
                start=args.command.start,
                interval=args.command.interval,
                limit=args.command.limit,
                workers=args.command.workers,
                temp_dir=args.command.temp_dir,
                force_rerun=args.command.force_rerun_decoders,
                tools=args.command.tools,
            )
        else:
            assert_never(args.command)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; exiting.")
        sys.exit(130)  # 128 + SIGINT, the conventional shell exit code


if __name__ == "__main__":
    main()
