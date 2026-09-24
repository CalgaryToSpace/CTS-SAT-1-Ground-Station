"""Bootstrap local web UI development by downloading the deployed server's
pipeline output instead of running the whole pipeline locally.

Fetches every table in `export_tables.TABLE_SPECS` from the server's
`/raw-export/{table_key}` route (see `export_raw`) into `--data-dir`, under
the same fixed filenames the pipeline itself writes -- so afterwards
`cts1_data_web_ui` finds them exactly as if they'd been produced here.

Usage (uv):
    uv run cts1_bootstrap_dev_data
    uv run cts1_bootstrap_dev_data --data-dir output --only everything_decoded
"""

from __future__ import annotations

# tyro resolves dataclass field annotations at runtime (via get_type_hints),
# so imports used only in annotations below still need to be real imports.
# ruff: noqa: TC003

__all__ = ["main"]

import sys
from dataclasses import dataclass
from pathlib import Path

import requests
import tyro
from loguru import logger

from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate import (
    pipeline as step_1_pipeline,
)

from .export_raw import raw_export_url
from .export_tables import TABLE_SPECS, TableSpec

DEFAULT_BASE_URL = "https://frontiersat.mooo.com"

# Generous: the larger tables (raw_packets, everything_decoded) run to
# hundreds of MB, but this only bounds the wait between chunks, not the
# whole download.
_TIMEOUT_SECONDS = 60
_CHUNK_SIZE_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class Args:
    """Download the web UI's parquet files from a deployed server."""

    data_dir: Path = step_1_pipeline.DEFAULT_DATA_DIR
    """Directory to save the files in -- the same one `cts1_data_web_ui
    --data-dir` reads from."""

    base_url: str = DEFAULT_BASE_URL
    """Deployed web UI to download from."""

    only: tuple[str, ...] = ()
    """Table key(s) to download (default: all of them)."""


def _download_table(spec: TableSpec, base_url: str, data_dir: Path) -> bool:
    """Stream one table's parquet file into `data_dir`, returning whether it
    succeeded.

    Downloaded to a temp file and renamed into place only once complete, so
    an interrupted download never leaves a truncated parquet file behind for
    the web UI to choke on.
    """
    url = base_url.rstrip("/") + raw_export_url(spec.key)
    dest = data_dir / spec.filename
    tmp_path = dest.with_suffix(".tmp")
    logger.info(f"Downloading {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE_BYTES):
                    f.write(chunk)
    except requests.RequestException as e:
        logger.error(f"Failed to download {spec.key}: {e}")
        return False
    tmp_path.replace(dest)
    logger.info(f"Saved {spec.key} ({dest.stat().st_size / 1e6:,.1f} MB)")
    return True


def main() -> None:
    args = tyro.cli(Args)

    specs_by_key = {spec.key: spec for spec in TABLE_SPECS}
    unknown = [key for key in args.only if key not in specs_by_key]
    if unknown:
        logger.error(
            f"Unknown table key(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(specs_by_key)}."
        )
        sys.exit(2)
    specs = [specs_by_key[key] for key in args.only] if args.only else TABLE_SPECS

    args.data_dir.mkdir(parents=True, exist_ok=True)
    failed = [
        spec.key
        for spec in specs
        if not _download_table(spec, args.base_url, args.data_dir)
    ]
    if failed:
        logger.error(f"Failed to download: {', '.join(failed)}")
        sys.exit(1)
    logger.info("Done -- run `uv run cts1_data_web_ui` next.")


if __name__ == "__main__":
    main()
