"""Step 1: download and demodulate.

Read the observations step 0 listed into DuckDB's `raw_observations` (this
step never queries the SatNOGS observation listing itself), then decode
every observation with audio and/or already-demodulated packets: download
the `.ogg` into a throwaway temp dir and run it through
`askew_demod_from_file`, `sso_rx_replay --forensics-report`, and
`gr_satellites` (via `sox`-converted WAV), download any `demoddata` packet
URLs directly, and land every decoded frame/PDU in DuckDB's `raw_packets`.

Invoked via the top-level CLI's `step_1` subcommand -- see
`cts1_mo_tools.cts1_processing_pipeline.cli`.
"""

from __future__ import annotations

from . import _subprocess_registry

__all__ = ["DB_FILENAME", "DEFAULT_DATA_DIR", "DEFAULT_DB_PATH", "run"]

import concurrent.futures
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import verify_csp_packet_crc32c
from cts1_mo_tools.cts1_processing_pipeline import landing_db, resource_limits
from cts1_mo_tools.cts1_processing_pipeline.common import parse_start_filter

from . import db
from .audio import convert_ogg_to_wav, download_audio
from .decode_askew_demod import run_askew_demod_from_file
from .decode_gr_satellites import DEFAULT_SATCFG_PATH, run_gr_satellites_pdu
from .decode_gr_satellites_kiss import run_gr_satellites_kiss
from .decode_satnogs_client_live_data import run_satnogs_client_live_data
from .decode_sso_rx_replay import run_sso_rx_replay

if TYPE_CHECKING:
    import duckdb

# See `landing_db` for why every step shares one `data_dir`.
DEFAULT_DATA_DIR = landing_db.DEFAULT_DATA_DIR
DB_FILENAME = landing_db.DB_FILENAME
DEFAULT_DB_PATH = landing_db.DEFAULT_DB_PATH
# Observations decoded per batch; a checkpoint is considered after each one.
CHECKPOINT_INTERVAL = 100
# Floor on the wall-clock gap between two mid-run checkpoint writeouts.
# `export_parquets` rewrites every table in full, which on a long run costs
# far more than the handful of new observations each writeout actually
# persists, so `CHECKPOINT_INTERVAL` alone isn't enough of a brake. The
# final writeout at the end of the run ignores this.
#
# On the deployment box (btrfs with compression, and swap it would rather
# not touch) a full rewrite is the single most disruptive thing the
# pipeline does to the web UI reading those same files, so the floor is
# generous: a crash loses at most this much decode work, which is cheap
# next to paying for the rewrite every few minutes. See
# `cts1_mo_tools/docs/resource-tuning.md`.
MIN_SECONDS_BETWEEN_CHECKPOINTS = 10 * 60.0
DEMOD_DOWNLOAD_WORKERS = resource_limits.DEFAULT_DEMOD_DOWNLOAD_WORKERS
DECODERS = (
    "askew_demod_from_file",
    "sso_rx_replay",
    "gr_satellites_pdu",
    "gr_satellites_kiss",
    "satnogs_client_live_data",
)
# Command each decoder's underlying tool is invoked as `<cmd> --version`
# with, for `decoder_runs.version`. None means the decoder has no versioned
# external tool (satnogs_client_live_data just downloads pre-decoded packet URLs
# from SatNOGS -- see _SATNOGS_CLIENT_LIVE_DATA_VERSION_HARDCODE below for its version).
_DECODER_VERSION_COMMAND = {
    "askew_demod_from_file": "askew_demod_from_file",
    "sso_rx_replay": "sso_rx_replay",
    "gr_satellites_pdu": "gr_satellites",
    "gr_satellites_kiss": "gr_satellites",
    "satnogs_client_live_data": None,
}
# satnogs_client_live_data has no local versioned tool -- it downloads packets
# already decoded by SatNOGS's own (continuously-deployed) infrastructure.
_SATNOGS_CLIENT_LIVE_DATA_VERSION_HARDCODE = "SatNOGS Rolling Release"


def _resolve_decoder_versions(tools: frozenset[str]) -> dict[str, str | None]:
    """Run `<tool> --version` once per distinct command, first line only.

    Only resolves versions for decoders in `tools`.
    """
    version_by_command: dict[str, str | None] = {}
    versions: dict[str, str | None] = {}
    for decoder, command in _DECODER_VERSION_COMMAND.items():
        if decoder not in tools:
            continue
        if decoder == "satnogs_client_live_data":
            versions[decoder] = _SATNOGS_CLIENT_LIVE_DATA_VERSION_HARDCODE
            continue
        if command is None:
            versions[decoder] = None
            continue
        if command not in version_by_command:
            proc = _subprocess_registry.run_tracked([command, "--version"], text=True)
            first_line = proc.stdout.splitlines()[0].strip() if proc.stdout else None
            version_by_command[command] = first_line
        versions[decoder] = version_by_command[command]
    return versions


def _csp_crc_valid(data_hex: str) -> bool:
    """Whether `data_hex` ends in a valid trailing CSP CRC-32C."""
    is_valid, _computed, _received = verify_csp_packet_crc32c(bytes.fromhex(data_hex))
    return is_valid


def _received_at(obs: dict[str, Any], *, time_in_file_ms: float | None) -> datetime:
    """Absolute UTC timestamp a packet was received.

    `time_in_file_ms` is an offset from the observation's start; decoders
    that can't determine a within-file position (gr_satellites_pdu, and
    sso_rx_replay/gr_satellites_kiss/satnogs_client_live_data on the rare row that
    lacks one) fall back to the observation's end time as the best available
    estimate.
    """
    if time_in_file_ms is not None:
        return obs["start"] + timedelta(milliseconds=time_in_file_ms)
    return obs["end"]


def _process_audio(  # noqa: C901, PLR0912
    obs: dict[str, Any],
    temp_dir: Path | None,
    tools: frozenset[str],
) -> list[dict[str, Any]]:
    """Download one observation's audio and run the enabled audio decoders on it.

    askew_demod_from_file, sso_rx_replay, gr_satellites --hexdump, and
    gr_satellites --kiss_out all now finish in a couple of seconds, so they
    all run at normal (small pool) concurrency, sharing one temp dir that's
    cleaned up before returning. Only the decoders named in `tools` run.
    """
    obs_id = obs["id"]
    audio_url = obs["payload"]
    rows: list[dict[str, Any]] = []
    needs_wav = "gr_satellites_pdu" in tools or "gr_satellites_kiss" in tools

    with tempfile.TemporaryDirectory(prefix="cts1_pipeline_", dir=temp_dir) as tmp_name:
        try:
            ogg_path = download_audio(audio_url, Path(tmp_name))
        except Exception:  # noqa: BLE001
            logger.exception(f"Failed to download audio for observation {obs_id}")
            return rows

        if "askew_demod_from_file" in tools:
            try:
                askew_rows = run_askew_demod_from_file(ogg_path)
            except Exception:  # noqa: BLE001
                logger.exception(
                    f"askew_demod_from_file decode failed for observation {obs_id}"
                )
                askew_rows = []
            for row in askew_rows:
                rows.append(  # noqa: PERF401
                    {
                        "observation_id": obs_id,
                        "decoder": "askew_demod_from_file",
                        "audio_url": audio_url,
                        "ingested_at": datetime.now(UTC),
                        "received_at": _received_at(
                            obs, time_in_file_ms=row["time_in_file_ms"]
                        ),
                        **row,
                    }
                )

        if "sso_rx_replay" in tools:
            sso_rows = run_sso_rx_replay(ogg_path, report_filename=ogg_path.name)
            for row in sso_rows:
                rows.append(  # noqa: PERF401
                    {
                        "observation_id": obs_id,
                        "decoder": "sso_rx_replay",
                        "audio_url": audio_url,
                        "ingested_at": datetime.now(UTC),
                        "received_at": _received_at(
                            obs, time_in_file_ms=row["time_in_file_ms"]
                        ),
                        **row,
                    }
                )

        if not needs_wav:
            return rows

        try:
            wav_path = convert_ogg_to_wav(ogg_path)
        except Exception:  # noqa: BLE001
            logger.exception(f"WAV conversion failed for observation {obs_id}")
            return rows

        if "gr_satellites_pdu" in tools:
            try:
                gr_rows = run_gr_satellites_pdu(wav_path, satcfg=DEFAULT_SATCFG_PATH)
            except Exception:  # noqa: BLE001
                logger.exception(
                    f"gr_satellites_pdu decode failed for observation {obs_id}"
                )
                gr_rows = []
            for row in gr_rows:
                rows.append(  # noqa: PERF401
                    {
                        "observation_id": obs_id,
                        "decoder": "gr_satellites_pdu",
                        "audio_url": audio_url,
                        "ingested_at": datetime.now(UTC),
                        "received_at": _received_at(obs, time_in_file_ms=None),
                        "rs_correctable": True,  # gr_satellites has no bad frames
                        **row,
                    }
                )

        if "gr_satellites_kiss" in tools:
            try:
                kiss_rows = run_gr_satellites_kiss(wav_path, satcfg=DEFAULT_SATCFG_PATH)
            except Exception:  # noqa: BLE001
                logger.exception(
                    f"gr_satellites_kiss decode failed for observation {obs_id}"
                )
                kiss_rows = []
            for row in kiss_rows:
                rows.append(  # noqa: PERF401
                    {
                        "observation_id": obs_id,
                        "decoder": "gr_satellites_kiss",
                        "audio_url": audio_url,
                        "ingested_at": datetime.now(UTC),
                        "received_at": _received_at(
                            obs, time_in_file_ms=row["time_in_file_ms"]
                        ),
                        "rs_correctable": True,  # gr_satellites has no bad frames
                        **row,
                    }
                )

    return rows


def _process_demod(obs: dict[str, Any], tools: frozenset[str]) -> list[dict[str, Any]]:
    """Download every already-demodulated packet SatNOGS has for this observation.

    Unlike the audio decoders, this needs no download of its own audio and
    no temp dir: each `demoddata` entry is already a standalone packet URL,
    downloaded via a thread pool nested inside this (already-pooled) call
    (see `run_satnogs_client_live_data`). A no-op unless "satnogs_client_live_data" is
    in `tools`.
    """
    if "satnogs_client_live_data" not in tools:
        return []

    obs_id = obs["id"]
    audio_url = obs.get("payload")
    rows: list[dict[str, Any]] = []

    try:
        demod_rows = run_satnogs_client_live_data(
            obs["demoddata"],
            observation_id=obs_id,
            max_workers=DEMOD_DOWNLOAD_WORKERS,
        )
    except Exception:  # noqa: BLE001
        logger.exception(f"satnogs_client_live_data failed for observation {obs_id}")
        demod_rows = []

    for row in demod_rows:
        time_in_file_ms = (row["received_at"] - obs["start"]).total_seconds() * 1000

        rows.append(
            {
                "observation_id": obs_id,
                "decoder": "satnogs_client_live_data",
                "audio_url": audio_url,
                "ingested_at": datetime.now(UTC),
                "time_in_file_ms": time_in_file_ms,
                "rs_correctable": True,  # no uncorrectable frames from this decoder
                **row,
            }
        )

    return rows


def _process_fast(
    obs: dict[str, Any],
    temp_dir: Path | None,
    tools: frozenset[str],
) -> list[dict[str, Any]]:
    """Run every enabled decoder for one observation: audio ones plus demod."""
    rows: list[dict[str, Any]] = []
    if obs.get("payload"):
        rows.extend(_process_audio(obs, temp_dir, tools))
    if obs.get("demoddata"):
        rows.extend(_process_demod(obs, tools))
    return rows


def _process_fast_timed(
    obs: dict[str, Any],
    temp_dir: Path | None,
    tools: frozenset[str],
) -> tuple[list[dict[str, Any]], int]:
    """Run `_process_fast` and report how long it took, in milliseconds."""
    started_at = time.monotonic()
    rows = _process_fast(obs, temp_dir, tools)
    runtime_ms = round((time.monotonic() - started_at) * 1000)
    return rows, runtime_ms


def _select_candidates(
    observations: list[dict[str, Any]],
    *,
    done: set[tuple[int, str]],
    limit: int | None,
    tools: frozenset[str],
) -> list[dict[str, Any]]:
    """Observations that have audio and/or demoddata and still need decoding.

    Stops once `limit` candidates have been picked (None = no cap). Only
    `tools` are considered when checking whether an observation is done.
    """
    candidates: list[dict[str, Any]] = []
    for obs in observations:
        if limit is not None and len(candidates) >= limit:
            break
        if not obs.get("payload") and not obs.get("demoddata"):
            continue
        if all((obs["id"], decoder) in done for decoder in tools):
            continue
        candidates.append(obs)
    return candidates


def run(  # noqa: C901, PLR0913, PLR0915
    *,
    norad_id: str = "69015",
    data_dir: Path = DEFAULT_DATA_DIR,
    start: str | None = None,
    limit: int | None = None,
    workers: int = resource_limits.DEFAULT_DECODER_WORKERS,
    temp_dir: Path | None = None,
    force_rerun: bool = False,
    tools: tuple[str, ...] | None = None,
) -> None:
    """Run the pipeline once.

    Args:
        norad_id: NORAD catalog ID of the target satellite -- only its
            observations in raw_observations are considered.
        data_dir: Directory to find the DuckDB database (`DB_FILENAME`) in
            -- raw_observations is read from there (step 0 writes it),
            raw_packets/decoder_runs land there, and their checkpoint
            parquet exports land alongside it.
        start: Only decode observations starting at/after this point: a
            duration like "3 days" (relative to now) or an ISO 8601
            date/datetime. None means every observation step 0 has listed.
        limit: Cap the number of observations decoded this run (for testing).
        workers: Concurrency for the decoders (askew_demod_from_file,
            sso_rx_replay, gr_satellites --hexdump, gr_satellites
            --kiss_out, satnogs_client_live_data), which each finish in a few
            seconds (satnogs_client_live_data spins up its own
            nested pool -- see DEMOD_DOWNLOAD_WORKERS -- for its own
            observation's packet downloads).
        temp_dir: Directory to create per-observation temp dirs (audio
            downloads/WAV conversions) under. None uses the platform default
            (see `tempfile`).
        force_rerun: Ignore the decoder_runs table's record of which
            observation/decoder pairs have already been run, and rerun every
            decoder on every candidate observation regardless. New runs are
            still recorded to decoder_runs as usual.
        tools: Which decoders to run, from DECODERS. None (the default) runs
            all of them.

    Raises:
        ValueError: If `tools` names something outside DECODERS.
    """
    if tools is None:
        enabled_tools = frozenset(DECODERS)
    else:
        unknown = set(tools) - set(DECODERS)
        if unknown:
            msg = f"Unknown tool(s) {sorted(unknown)}; choose from {DECODERS}"
            raise ValueError(msg)
        enabled_tools = frozenset(tools)

    db_path = data_dir / DB_FILENAME

    start_gte = parse_start_filter(start) if start is not None else None

    total_decoded = 0
    last_checkpoint_at = time.monotonic()

    def write_checkpoint(con: duckdb.DuckDBPyConnection) -> None:
        nonlocal last_checkpoint_at
        started_at = time.monotonic()
        logger.debug(
            f"Checkpointing after {total_decoded} observation(s) "
            f"({started_at - last_checkpoint_at:.1f}s since the last one)"
        )
        con.execute("CHECKPOINT")
        db.export_parquets(con, db_path)
        last_checkpoint_at = time.monotonic()
        logger.debug(f"Checkpoint written in {last_checkpoint_at - started_at:.1f}s")

    with (
        landing_db.connect(db_path) as con,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as fast_executor,
    ):
        done: set[tuple[int, str]] = (
            set() if force_rerun else db.already_decoded_pairs(con)
        )
        if force_rerun:
            logger.info("--force-rerun: ignoring decoder_runs history")

        observations = db.load_observations(con, norad_id=norad_id, start_gte=start_gte)
        candidates = _select_candidates(
            observations, done=done, limit=limit, tools=enabled_tools
        )
        logger.info(
            f"{len(candidates)} of {len(observations)} listed observation(s) for "
            f"NORAD {norad_id}"
            + (f" starting after {start_gte.isoformat()}" if start_gte else "")
            + " need decoding"
        )
        logger.info(f"  Using tools: {sorted(enabled_tools)}")
        if not candidates:
            return

        decoder_versions = _resolve_decoder_versions(enabled_tools)
        logger.info(f"Decoder versions: {decoder_versions}")

        def dispatch(batch: list[dict[str, Any]]) -> None:
            nonlocal total_decoded
            logger.info(f"Dispatching a batch of {len(batch)} observation(s)...")
            futures = {
                fast_executor.submit(
                    _process_fast_timed, obs, temp_dir, enabled_tools
                ): obs
                for obs in batch
            }
            for future in concurrent.futures.as_completed(futures):
                obs = futures[future]
                try:
                    rows, runtime_ms = future.result()
                except Exception:  # noqa: BLE001
                    # Every decoder already swallows its own failures, so this
                    # is something unexpected: leave the observation unrecorded
                    # so a later run retries it, rather than stamping a
                    # decoder_run with no runtime.
                    logger.exception(f"Failed processing observation {obs['id']}")
                    continue
                if rows:
                    df = pl.DataFrame(rows, infer_schema_length=None)
                    df = df.with_columns(
                        pl.col("data_hex")
                        .map_elements(_csp_crc_valid, return_dtype=pl.Boolean)
                        .alias("csp_crc_valid")
                    )
                    db.append_packets(con, df)
                db.record_decoder_runs(
                    con, obs["id"], decoder_versions, runtime_ms=runtime_ms
                )
                total_decoded += 1
                logger.info(
                    f"[{total_decoded}/{len(candidates)}] observation {obs['id']}"
                )

        try:
            for batch_start in range(0, len(candidates), CHECKPOINT_INTERVAL):
                dispatch(candidates[batch_start : batch_start + CHECKPOINT_INTERVAL])

                seconds_since_checkpoint = time.monotonic() - last_checkpoint_at
                if seconds_since_checkpoint >= MIN_SECONDS_BETWEEN_CHECKPOINTS:
                    write_checkpoint(con)
                else:
                    logger.debug(
                        "Skipping checkpoint: only "
                        f"{seconds_since_checkpoint:.1f}s since the last "
                        f"one (minimum {MIN_SECONDS_BETWEEN_CHECKPOINTS:.0f}s)"
                    )

        except KeyboardInterrupt:
            # Worker threads block inside subprocess.run()-style calls, which
            # a signal can't interrupt from here — kill the children so those
            # threads unblock, then let the executor's own __exit__ join them
            # (fast now that nothing is left running) instead of hanging.
            logger.warning("Interrupted; terminating in-flight downloads/decodes...")
            _subprocess_registry.terminate_all()
            fast_executor.shutdown(wait=True, cancel_futures=True)
            raise

        # Final writeout: unconditional, so whatever the rate limit held
        # back above still lands before the run ends.
        write_checkpoint(con)

    logger.info(f"Done. {total_decoded} observation(s) decoded.")
