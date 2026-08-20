"""
cts1_picam_to_jpg.py
=====================
Convert an image in the PiCAM ASCII format to a JPG file.

Usage (uv):
    uv run cts1_picam_to_jpg <input_text_file> <output_jpg_file>
"""

import binascii
import sys
from pathlib import Path
from typing import Annotated

import tyro
from loguru import logger

PICAM_STRIPPED_LINE_LENGTH = 65  # @ + 64 hex chars


def parse_picam_ascii_to_jpg_bytes(text: str) -> bytes:
    """Parse PiCAM ASCII-format image data (as a string) into raw JPG bytes.

    Each line is expected to look like ``@SSSSTTTT<hex image data>``, where
    ``SSSS`` is the 4-hex-digit sentence number, ``TTTT`` is the 4-hex-digit
    total sentence count, and the remainder is hex-encoded JPG data. A
    sentence number of ``FACE`` marks the end-of-telemetry sentinel line.

    Malformed or wrong-length lines are logged and skipped.
    """
    jpg_data = bytearray()

    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        # CTS-SAT-1 firmware header string in file:
        if line == "START_CAM:":
            logger.debug(f"Line {line_num}: Start of image data.")
            continue

        if not line.startswith("@"):
            logger.error(f"Line {line_num} doesn't start with an @ sign. Skipping.")
            continue

        if len(line) != PICAM_STRIPPED_LINE_LENGTH:
            logger.warning(
                f"Line {line_num} is wrong length ({len(line)} chars). Skipping."
            )
            continue

        line = line[1:]  # remove '@' sign

        sentence_num_hex = line[0:4]
        total_sentences_hex = line[4:8]
        img_data_hex = line[8:].strip()

        if sentence_num_hex == "FACE":
            logger.info(f"Reached end telemetry seq on Line {line_num}.")
            # FIXME: Optionally parse this line per the datasheet's spec.
            continue

        logger.info(
            f"Line {line_num}: sentence_num={int(sentence_num_hex, 16)}, "
            f"total_sentences={int(total_sentences_hex, 16)}"
        )

        jpg_data += binascii.unhexlify(img_data_hex)

    return bytes(jpg_data)


def read_and_parse_image(text_file_in: Path, jpg_file_out: Path) -> None:
    """Read PiCAM ASCII data from `text_file_in` and write the decoded JPG to
    `jpg_file_out`."""
    text = text_file_in.read_text()
    jpg_data = parse_picam_ascii_to_jpg_bytes(text)
    jpg_file_out.write_bytes(jpg_data)


def run(
    text_file_in: Annotated[Path, tyro.conf.Positional],
    jpg_file_out: Annotated[Path, tyro.conf.Positional],
) -> None:
    """Convert an image in the PiCAM ASCII format to a JPG file.

    Args:
        text_file_in: Path to the input PiCAM ASCII text file.
        jpg_file_out: Path to write the decoded JPG file to.
    """
    if jpg_file_out.suffix.lower() != ".jpg":
        logger.error("Output file must be a .jpg file.")
        sys.exit(1)

    read_and_parse_image(text_file_in, jpg_file_out)
    logger.info(f"Done. Wrote: {jpg_file_out}")


def main() -> None:
    """Entry point."""
    tyro.cli(run)


if __name__ == "__main__":
    main()
