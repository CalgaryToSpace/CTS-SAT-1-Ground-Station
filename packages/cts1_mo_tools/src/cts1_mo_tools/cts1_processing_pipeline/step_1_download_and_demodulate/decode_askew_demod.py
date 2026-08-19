"""Wrapper around the `askew_demod_from_file` CLI tool.

`askew_demod_from_file` decodes AX100 ASM+Golay (CSP) beacon frames straight
out of a SatNOGS `.ogg` recording, emitting one JSON object per decoded frame
to stdout. Its `time_in_file_ms` is trustworthy, same as sso_rx_replay's.
"""

from __future__ import annotations

__all__ = ["parse_askew_line", "run_askew_demod_from_file"]

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from . import _subprocess_registry

if TYPE_CHECKING:
    from pathlib import Path

DECODER_NAME = "askew_demod_from_file"


def parse_askew_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        logger.warning(f"askew_demod_from_file: could not parse line: {line!r}")
        return None

    assert isinstance(obj, dict)

    return {
        "time_in_file_ms": obj["time_in_file_ms"],
        "data_hex": obj["data_hex"],
        "data_length_bytes": obj["data_length_bytes"],
        "rs_corrected_error_count": obj["rs_corrected_error_count"],
        "rs_correctable": obj["rs_correctable"],
        # Skip including - We calculate it in here anyway - "crc_pass": obj["crc_pass"],
    }


def run_askew_demod_from_file(audio_path: Path) -> list[dict[str, Any]]:
    """Run `askew_demod_from_file --output-filter=all` on an audio file.

    Args:
        audio_path: Path to a local .ogg/.wav recording.

    Returns:
        One dict per decoded frame. stderr is ignored (not surfaced even on
        a non-zero exit -- a bad/unreadable file just yields no rows).
    """
    proc = _subprocess_registry.run_tracked(
        ["askew_demod_from_file", "--output-filter=all", str(audio_path)],
        check=False,
        text=True,
    )

    rows = [row for line in proc.stdout.splitlines() if (row := parse_askew_line(line))]
    logger.debug(f"askew_demod_from_file: {len(rows)} frame(s) for {audio_path}")
    return rows
