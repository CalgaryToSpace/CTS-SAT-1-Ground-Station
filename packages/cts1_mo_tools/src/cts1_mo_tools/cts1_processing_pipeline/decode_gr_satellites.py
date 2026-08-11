"""Wrapper around the `gr_satellites` CLI tool (CTS-SAT-1 / FrontierSat config).

`gr_satellites` does not know about CTS-SAT-1 out of the box, so it is always
invoked with the bundled `gr_satellites_frontiersat.yml` satellite
definition. Frames are pulled off its `--hexdump` stdout, which prints one
"VERBOSE PDU DEBUG PRINT" block per decoded PDU:

    ***** VERBOSE PDU DEBUG PRINT ******
    ((transmitter . 9k6 FSK downlink))
    pdu length =        208 bytes
    pdu vector contents =
    0000: c2 a2 8a 00 10 d0 6c 0e 00 08 22 08 5e 07 fa 07
    ...
    ************************************

`gr_satellites` reads WAV (not SatNOGS `.ogg` directly in all builds), so
callers should convert with `audio.convert_ogg_to_wav` first.
"""

from __future__ import annotations

__all__ = ["DEFAULT_SATCFG_PATH", "parse_hexdump_stdout", "run_gr_satellites"]

import re
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

DECODER_NAME = "gr_satellites"

DEFAULT_SATCFG_PATH = Path(__file__).parent / "gr_satellites_frontiersat.yml"

_PDU_BLOCK_RE = re.compile(
    r"\*+ VERBOSE PDU DEBUG PRINT \*+\s*\n"
    r"\(\(transmitter\s*\.\s*(?P<transmitter>[^)]+)\)\)\s*\n"
    r"pdu length =\s*(?P<length>\d+) bytes\s*\n"
    r"pdu vector contents =\s*\n"
    r"(?P<hexblock>(?:[0-9a-f]{4}:(?: [0-9a-f]{2})+\s*\n)+)",
)
_HEX_LINE_RE = re.compile(r"^[0-9a-f]{4}:((?: [0-9a-f]{2})+)", re.MULTILINE)


def parse_hexdump_stdout(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in _PDU_BLOCK_RE.finditer(stdout):
        length = int(m.group("length"))
        hex_bytes = "".join(
            byte
            for line_match in _HEX_LINE_RE.finditer(m.group("hexblock"))
            for byte in line_match.group(1).split()
        )
        rows.append(
            {
                "gr_transmitter": m.group("transmitter").strip(),
                "gr_pdu_length_bytes": length,
                "gr_pdu_hex": hex_bytes[: length * 2],
            }
        )
    return rows


def run_gr_satellites(
    wav_path: Path, *, satcfg: Path = DEFAULT_SATCFG_PATH
) -> list[dict[str, Any]]:
    """Run `gr_satellites` against a WAV file using the CTS-SAT-1 satcfg.

    Args:
        wav_path: Path to a local WAV recording (see `audio.convert_ogg_to_wav`).
        satcfg: Path to the gr_satellites satellite YAML definition.

    Returns:
        One dict per decoded PDU.
    """
    proc = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "gr_satellites",
            str(satcfg),
            "--hexdump",
            "--wavfile",
            str(wav_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(
            f"gr_satellites exited {proc.returncode} on {wav_path}: {proc.stderr}"
        )
        return []

    rows = parse_hexdump_stdout(proc.stdout)
    logger.debug(f"gr_satellites: {len(rows)} PDU(s) for {wav_path}")
    return rows
