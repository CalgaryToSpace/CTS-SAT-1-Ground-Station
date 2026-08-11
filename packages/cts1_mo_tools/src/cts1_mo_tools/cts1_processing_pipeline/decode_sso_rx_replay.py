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

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from . import _subprocess_registry

if TYPE_CHECKING:
    from pathlib import Path

DECODER_NAME = "sso_rx_replay"


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

    return {
        "sso_filename": obj["filename"],
        "sso_time_in_file_ms": obj["time_in_file_ms"],
        "sso_rssi": obj["rssi"],
        "sso_rs": obj["rs"],
        "sso_data_base64": obj["data_base64"],
        "sso_error": obj.get("error"),  # I don't think this actually really exists.
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
