"""Wrapper around the `sso_rx_replay --forensics-report` CLI tool.

`sso_rx_replay` decodes AX100/CSP frames straight out of a SatNOGS `.ogg`
recording (or WAV / headerless PCM). With `--forensics-report` it emits one
JSON object per line to stdout and never touches its own SQLite store, which
makes it a clean subprocess to shell out to per-observation:

  - one line per decoded frame: filename, time_in_file_ms, rssi, rs, data_base64
  - one line with only "filename" if the file decoded cleanly but had no frames
  - one line with "filename" + "error" if the file could not be read at all
"""

from __future__ import annotations

__all__ = ["parse_forensics_line", "run_sso_rx_replay"]

import base64
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from cts1_mo_tools.cts1_decode_satnogs_packets import verify_csp_packet_crc32c

from . import _subprocess_registry

if TYPE_CHECKING:
    from pathlib import Path

DECODER_NAME = "sso_rx_replay"


def _quality_tier(*, rs_correctable: bool, crc_pass: bool) -> str:
    """Tier for a frame, from its RS and CSP CRC-32C outcomes.

    RS-uncorrectable frames are "believable" whatever their CRC says: the
    payload is known-damaged, so a CRC that happens to pass is not meaningful.
    """
    if not rs_correctable:
        return "believable"
    return "good" if crc_pass else "rs_correctable_crc_fail"


def parse_forensics_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        logger.warning(f"sso_rx_replay: could not parse forensics line: {line!r}")
        return None

    assert isinstance(obj, dict)

    if set(obj.keys()) == {"filename"}:
        # SSO quirk where it reports the filename even if there are no frames.
        return None

    if "error" in obj:
        # The file couldn't be read/decoded at all (corrupt/truncated/empty
        # download, unsupported encoding, ...) -- sso_rx_replay reports this
        # per-file rather than failing its whole invocation, so this is a
        # normal "skip this one file" outcome, not something to raise on.
        logger.warning(f"sso_rx_replay: {obj.get('filename', '?')}: {obj['error']}")
        return None

    data_bytes = base64.b64decode(obj["data_base64"])
    rs = obj["rs"]
    # We run with --no-csp-crc32, so sso_rx_replay reports no CRC verdict of
    # its own and leaves the trailer in place -- check it here, otherwise an
    # RS-corrected frame with a broken CRC would be tiered as "good".
    crc_pass, _computed, _received = verify_csp_packet_crc32c(data_bytes)

    return {
        # Ignore as useless - "sso_filename": obj["filename"],
        "time_in_file_ms": obj["time_in_file_ms"],
        "rssi_db": obj["rssi"],
        # Negative means RS-uncorrectable -- no meaningful error count then.
        "rs_corrected_error_count": rs if rs >= 0 else None,
        "rs_correctable": rs >= 0,
        "data_hex": data_bytes.hex(),
        "data_length_bytes": len(data_bytes),
        "quality_tier": _quality_tier(rs_correctable=rs >= 0, crc_pass=crc_pass),
    }


def run_sso_rx_replay(
    audio_path: Path, *, report_filename: str
) -> list[dict[str, Any]]:
    """Run `sso_rx_replay --forensics-report` on an audio file.

    Args:
        audio_path: Path to a local .ogg/.wav/headerless-PCM recording.
        report_filename: Value to report as "filename" in the JSON output
            (e.g. the original SatNOGS filename, when audio_path is a
            temporary copy).

    Returns:
        One dict per forensics-report JSON line (frames, no-decode, or error).
    """
    proc = _subprocess_registry.run_tracked(
        [
            "sso_rx_replay",
            str(audio_path),
            "--forensics-report",
            "--no-csp-crc32",  # Do not validate nor strip CRC32.
            f"--report-filename={report_filename}",
        ],
        check=False,  # non-zero exit is expected on unreadable audio
        text=True,
    )
    if proc.returncode not in (0, 1):
        logger.warning(
            f"sso_rx_replay exited {proc.returncode} on {audio_path}: {proc.stderr}"
        )

    rows = [
        row for line in proc.stdout.splitlines() if (row := parse_forensics_line(line))
    ]
    logger.debug(f"sso_rx_replay: {len(rows)} report line(s) for {report_filename}")
    return rows
