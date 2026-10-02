"""CTS-SAT-1 processing pipeline: top-level CLI.

Dispatches to the pipeline's steps as subcommands, with a handful of args
shared by every step (currently just `--data-dir`/`--debug`) parsed ahead of
the subcommand. `--data-dir` is the one directory every step reads/writes
its DuckDB database and parquet files in -- see each step's own
`DEFAULT_DATA_DIR`/`OUTPUT_FILENAME` for the fixed filename it looks for.

Usage (uv):
    uv run cts1_processing_pipeline step_0
    uv run cts1_processing_pipeline step_0 --start "3 days" --refetch-all
    uv run cts1_processing_pipeline step_1
    uv run cts1_processing_pipeline --data-dir output step_1 --norad-id 69015
    uv run cts1_processing_pipeline step_1 --limit 5 --debug
    uv run cts1_processing_pipeline step_2
    uv run cts1_processing_pipeline step_3
    uv run cts1_processing_pipeline step_4
    uv run cts1_processing_pipeline step_5
    uv run cts1_processing_pipeline daemon
    uv run cts1_processing_pipeline daemon --start "3 days" --interval 5
"""

from __future__ import annotations

# tyro resolves dataclass field annotations at runtime (via get_type_hints),
# so imports used only in annotations below still need to be real imports.
# ruff: noqa: TC003

__all__ = ["Args", "main"]

import os
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, assert_never

import tyro
from dotenv import load_dotenv
from loguru import logger
from tyro.conf import OmitSubcommandPrefixes

from . import daemon, resource_limits
from .step_0_list_observations import pipeline as step_0_list_observations
from .step_1_download_and_demodulate import pipeline as step_1_download_and_demodulate
from .step_2_deduplicate_packets import pipeline as step_2_deduplicate_packets
from .step_3_decode_packets import pipeline as step_3_decode_packets
from .step_4_detect_satellite_events import pipeline as step_4_detect_satellite_events
from .step_5_reassemble_tcmd_responses import (
    pipeline as step_5_reassemble_tcmd_responses,
)

DEFAULT_DATA_DIR = step_0_list_observations.DEFAULT_DATA_DIR


@dataclass(frozen=True, slots=True)
class Step0Args:
    """Step 0: list SatNOGS observations into raw_observations, in 12h
    windows, skipping windows already listed for good."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    start: str | None = None
    """List observations starting after this point: a duration like
    '3 days' (relative to now) or an ISO 8601 date/datetime, rounded down to
    the start of its 12h window. Omit for full history (since
    2026-08-01)."""

    refetch_all: bool = False
    """By default, a window already listed after it settled (see the
    observation_listing_windows table) is skipped. Set this to re-list
    every window since --start regardless."""


@dataclass(frozen=True, slots=True)
class Step1Args:
    """Step 1: download audio and decode packets for the observations step
    0 listed."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    start: str | None = None
    """Only decode observations starting after this point: a duration like
    '3 days' (relative to now) or an ISO 8601 date/datetime. Omit for every
    observation step 0 has listed."""

    limit: int | None = None
    """Cap the number of observations decoded this run (for testing)."""

    workers: int = resource_limits.DEFAULT_DECODER_WORKERS
    """Concurrency for the decoders (sso_rx_replay, gr_satellites --hexdump,
    gr_satellites --kiss_out, satnogs_client_live_data). Defaults low so a run
    doesn't saturate a small box out from under the web UI -- see
    `resource_limits`."""

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
    gr_satellites_pdu, gr_satellites_kiss, satnogs_client_live_data}. Omit to run
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
class Step5Args:
    """Step 5: reassemble multi-packet telecommand responses from
    everything_decoded into reassembled_tcmd_responses."""


@dataclass(frozen=True, slots=True)
class DaemonArgs:
    """Daemon: run steps 0-5 continuously -- an initial backfill of
    `--start`, then a full steps 0-5 rerun every `--interval` minutes."""

    norad_id: Annotated[str, tyro.conf.Positional] = "69015"
    """NORAD catalog ID of the satellite (default: 69015, CTS-SAT-1)."""

    start: str = "24 hours"
    """How far back the initial backfill reaches: a duration like '3 days'
    (relative to now) or an ISO 8601 date/datetime -- same syntax as step
    0/1's own --start. Resolved once at startup; every later run covers
    the same span."""

    interval: float = 15.0
    """Minutes between runs. Each run re-lists only the tail of the SatNOGS
    listing windows that haven't settled yet (about the last hour of
    observations), then decodes whatever's new and reruns steps 2 through
    5."""

    limit: int | None = None
    """Cap the number of observations decoded per step-1 run (for testing)."""

    workers: int = resource_limits.DEFAULT_DECODER_WORKERS
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
    Annotated[Step0Args, tyro.conf.subcommand(name="step_0", prefix_name=False)]
    | Annotated[Step1Args, tyro.conf.subcommand(name="step_1", prefix_name=False)]
    | Annotated[Step2Args, tyro.conf.subcommand(name="step_2", prefix_name=False)]
    | Annotated[Step3Args, tyro.conf.subcommand(name="step_3", prefix_name=False)]
    | Annotated[Step4Args, tyro.conf.subcommand(name="step_4", prefix_name=False)]
    | Annotated[Step5Args, tyro.conf.subcommand(name="step_5", prefix_name=False)]
    | Annotated[DaemonArgs, tyro.conf.subcommand(name="daemon", prefix_name=False)]
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Args:
    """CTS-SAT-1 processing pipeline."""

    data_dir: Path = DEFAULT_DATA_DIR
    """Directory every step reads/writes its files in: steps 0/1's DuckDB
    database (and the parquet files they checkpoint from that), and every
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

    # A loud banner per process start, so a restart after a crash (e.g. the
    # daemon container coming back up) stands out when scrolling the logs.
    # The log format only carries the time of day, so the date goes here.
    logger.info("=" * 72)
    logger.info(
        f"STARTING NEW PROCESS: cts1_processing_pipeline (pid {os.getpid()}) "
        f"at {datetime.now(UTC).isoformat(timespec='seconds')}"
    )
    logger.info(f"  argv: {shlex.join(sys.argv[1:]) or '(none)'}")
    logger.info("=" * 72)

    # Nothing is waiting on a pipeline run, so it yields the CPU to
    # whatever is (the web UI, on the deployment box) -- and every decoder
    # subprocess inherits this. See `resource_limits`.
    resource_limits.lower_process_priority()

    try:
        if isinstance(args.command, Step0Args):
            step_0_list_observations.run(
                norad_id=args.command.norad_id,
                data_dir=args.data_dir,
                start=args.command.start,
                refetch_all=args.command.refetch_all,
            )
        elif isinstance(args.command, Step1Args):
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
        elif isinstance(args.command, Step5Args):
            step_5_reassemble_tcmd_responses.run(data_dir=args.data_dir)
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
