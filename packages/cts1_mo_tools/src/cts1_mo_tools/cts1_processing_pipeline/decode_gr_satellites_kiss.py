"""Wrapper around `gr_satellites --kiss_out` (CTS-SAT-1 / FrontierSat config).

A separate decoding path from `decode_gr_satellites.py`'s `--hexdump` one:
KISS output is the only gr_satellites mode that carries a per-packet
timestamp. Per `pdu_to_kiss.py` in gr-satellites, every decoded PDU is
preceded by a KISS control frame (control byte 0x09) holding a big-endian
64-bit UTC Unix timestamp in milliseconds, derived from `--start_time` plus
the amount of the WAV consumed so far. Passing a fixed epoch `--start_time`
(see below) makes that timestamp file-relative without needing to decode at
real playback speed, so this runs just as fast as `--hexdump`.

`--hexdump` is still passed alongside `--kiss_out`: it's what makes the
yaml's `telemetry: hexdump` datasink valid at startup (gr_satellites raises
an `AttributeError` without it), even though that stdout output is not
parsed here — see `decode_gr_satellites.py` for the module that does.
"""

from __future__ import annotations

__all__ = ["parse_kiss_file", "run_gr_satellites_kiss"]

import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from . import _subprocess_registry
from .decode_gr_satellites import DEFAULT_SATCFG_PATH

DECODER_NAME = "gr_satellites_kiss"

# gr_satellites_frontiersat.yml defines exactly one transmitter; KISS output
# carries no transmitter label (unlike --hexdump's PDU debug print), so this
# is hardcoded to match the yaml rather than parsed at runtime.
GR_TRANSMITTER_NAME = "9k6 FSK downlink"

_FEND = 0xC0
_FESC = 0xDB
_TFEND = 0xDC
_TFESC = 0xDD
_TIMESTAMP_CONTROL_BYTE = 0x09
_DATA_CONTROL_BYTE = 0x00
_TIMESTAMP_PAYLOAD_LEN = 8


def _kiss_unescape(frame: bytes) -> bytes:
    """Reverse KISS FESC/TFEND/TFESC escaping within a single frame's bytes."""
    out = bytearray()
    it = iter(frame)
    for b in it:
        if b != _FESC:
            out.append(b)
            continue
        nxt = next(it, None)
        if nxt == _TFEND:
            out.append(_FEND)
        elif nxt == _TFESC:
            out.append(_FESC)
        elif nxt is not None:
            out.append(nxt)
    return bytes(out)


def parse_kiss_file(data: bytes, *, launch_time_ms: int) -> list[dict[str, Any]]:
    """Parse a gr_satellites `--kiss_out` file into one row per decoded PDU.

    Args:
        data: Raw bytes of the KISS output file.
        launch_time_ms: UTC Unix ms epoch matching the `--start_time` the
            gr_satellites subprocess was launched with; timestamps embedded
            in the KISS stream are relative to this to produce a
            file-relative `time_in_file_ms`.

    Returns:
        One dict per data frame, using the timestamp frame (if any) that
        immediately preceded it in the stream.
    """
    rows: list[dict[str, Any]] = []
    pending_timestamp_ms: int | None = None

    for raw_frame in data.split(bytes([_FEND])):
        if not raw_frame:
            continue
        frame = _kiss_unescape(raw_frame)
        if not frame:
            continue

        control, payload = frame[0], frame[1:]
        if control == _TIMESTAMP_CONTROL_BYTE:
            if len(payload) == _TIMESTAMP_PAYLOAD_LEN:
                pending_timestamp_ms = int.from_bytes(payload, "big")
            continue
        if control != _DATA_CONTROL_BYTE:
            continue  # some other KISS command (params, exit KISS mode, ...)

        time_in_file_ms = (
            pending_timestamp_ms - launch_time_ms
            if pending_timestamp_ms is not None
            else None
        )
        rows.append(
            {
                "transmitter_name": GR_TRANSMITTER_NAME,
                "data_length_bytes": len(payload),
                "data_hex": payload.hex(),
                "time_in_file_ms": time_in_file_ms,
            }
        )
        pending_timestamp_ms = None  # each timestamp frame covers one PDU

    return rows


def run_gr_satellites_kiss(
    wav_path: Path, *, satcfg: Path = DEFAULT_SATCFG_PATH
) -> list[dict[str, Any]]:
    """Run `gr_satellites --kiss_out` against a WAV file.

    Args:
        wav_path: Path to a local WAV recording (see `audio.convert_ogg_to_wav`).
        satcfg: Path to the gr_satellites satellite YAML definition.

    Returns:
        One dict per decoded PDU, with a file-relative `time_in_file_ms`.
    """
    with tempfile.NamedTemporaryFile(suffix=".kiss", delete=False) as tmp:
        kiss_path = Path(tmp.name)

    try:
        proc = _subprocess_registry.run_tracked(
            [
                "gr_satellites",
                str(satcfg),
                "--hexdump",
                "--kiss_out",
                str(kiss_path),
                # Note: Previously required --throttle on gr_satellites<5.10.0.
                "--wavfile",
                str(wav_path),
                # Fake start time so the timestamps are file-relative.
                "--start_time",
                "1970-01-01T00:00:00Z",
            ],
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning(
                f"gr_satellites_kiss exited {proc.returncode} on "
                f"{wav_path}: {proc.stderr}"
            )
            return []

        data = kiss_path.read_bytes()
    finally:
        kiss_path.unlink(missing_ok=True)

    rows = parse_kiss_file(data, launch_time_ms=0)
    logger.debug(f"gr_satellites_kiss: {len(rows)} PDU(s) for {wav_path}")
    return rows
