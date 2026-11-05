"""Month-partitioned parquet outputs.

The pipeline's biggest outputs (`raw_packets`, `distinct_packets_over_time`,
`everything_decoded`) are each a *directory* under the name the single file
used to have (e.g. `everything_decoded.parquet/`), holding one parquet file
per UTC month of `received_at` (`2026-11.parquet`). Polars reads a directory
like that exactly as it reads a single file -- `pl.scan_parquet(path)` -- so
readers don't need to know, while writers only rewrite the months that
changed.

Polars only reads such a directory if:
  - it holds nothing but the partition files (no temp files: they're written
    beside it and renamed in),
  - every file has the identical schema, and
  - it isn't empty.
Everything here keeps to that.

Steps 2 and 3 run incrementally off this layout: each records, in a
`<output>.state.json` sidecar, a signature of every input month file it last
processed (see `changed_input_months`), so the next run knows which months
of its input changed since -- whether that's new data for this month or
months-old history that idle backfill just landed.
"""

from __future__ import annotations

__all__ = [
    "changed_input_months",
    "month_expr",
    "month_files",
    "replace_months",
    "signatures",
    "write_frame",
    "write_state",
]

import json
import shutil
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from pathlib import Path

PARTITION_COLUMN = "received_at"
# The partition rows with a null `received_at` go in.
NO_MONTH = "none"


def month_expr(column: str = PARTITION_COLUMN) -> pl.Expr:
    """The partition (`YYYY-MM`, UTC) each row of a frame belongs in."""
    return (
        pl.col(column)
        .dt.convert_time_zone("UTC")
        .dt.strftime("%Y-%m")
        .fill_null(NO_MONTH)
    )


def month_files(path: Path) -> dict[str, Path]:
    """Every partition file under `path`, by month. Empty if `path` isn't a
    partitioned directory (yet) -- e.g. a single file from before this
    layout, or nothing at all."""
    if not path.is_dir():
        return {}
    return {file.stem: file for file in sorted(path.glob("*.parquet"))}


def signatures(path: Path) -> dict[str, list[int]]:
    """A cheap fingerprint of every partition file under `path`: it changes
    whenever that month's file is rewritten."""
    return {
        month: [file.stat().st_mtime_ns, file.stat().st_size]
        for month, file in month_files(path).items()
    }


def _state_path(output: Path) -> Path:
    return output.with_name(output.name + ".state.json")


def write_state(output: Path, input_signatures: Mapping[str, list[int]]) -> None:
    """Record that `output` is now up to date with these input months."""
    tmp = _state_path(output).with_suffix(".tmp")
    tmp.write_text(json.dumps({"input_signatures": input_signatures}))
    tmp.replace(_state_path(output))


def changed_input_months(
    output: Path, input_signatures: Mapping[str, list[int]]
) -> set[str] | None:
    """Which input months changed since `output` was last brought up to date
    (`write_state`), or None if that can't be told and `output` must be
    rebuilt in full: there's no state yet, `output` isn't partitioned yet,
    or an input month has disappeared."""
    try:
        state = json.loads(_state_path(output).read_text())["input_signatures"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    if not month_files(output) or set(state) - set(input_signatures):
        return None
    return {
        month
        for month, signature in input_signatures.items()
        if state.get(month) != signature
    }


def replace_months(
    path: Path,
    writers: Mapping[str, Callable[[Path], None] | None],
    *,
    full: bool,
) -> None:
    """Rewrite the given months of the partitioned directory `path`.

    Each writer writes its month's file to the temp path it's given (`None`
    deletes that month instead). With `full=True`, `writers` is the whole
    output: it's built in a staging directory beside `path` and swapped in,
    replacing whatever `path` was (a partitioned directory, or a single file
    from before this layout). Otherwise each month is written beside `path`
    and renamed into it, leaving every other month alone.
    """
    if full:
        staging = path.with_name(path.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for month, write in writers.items():
            if write is not None:
                write(staging / f"{month}.parquet")
        old = path.with_name(path.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if path.is_dir():
            path.rename(old)
        elif path.exists():
            path.unlink()
        staging.rename(path)
        shutil.rmtree(old, ignore_errors=True)
        return

    path.mkdir(exist_ok=True)
    for month, write in writers.items():
        target = path / f"{month}.parquet"
        if write is None:
            target.unlink(missing_ok=True)
            continue
        tmp = path.with_name(f".{path.name}.{month}.tmp.parquet")
        write(tmp)
        tmp.replace(target)


def write_frame(
    df: pl.DataFrame,
    path: Path,
    *,
    months: Collection[str] | None = None,
    row_group_size: int | None = None,
) -> None:
    """Write `df` (sorted by `PARTITION_COLUMN`) into the partitioned
    directory `path`.

    `months=None` makes `df` the whole output. Otherwise only `months` are
    rewritten -- `df` must hold every row of each of them, and none of any
    other month -- and a month in `months` that `df` has no rows for is
    deleted.
    """
    keys = df.select(month_expr()).to_series()
    writers: dict[str, Callable[[Path], None] | None] = dict.fromkeys(months or ())
    offset = 0
    for run in keys.rle().to_list():
        month, length = run["value"], run["len"]
        assert month not in writers or writers[month] is None, "df isn't sorted"
        part = df.slice(offset, length)
        writers[month] = lambda tmp, part=part: part.write_parquet(
            tmp, row_group_size=row_group_size
        )
        offset += length
    if months is None and not writers:
        # An empty output still needs one (empty) file: polars can't read
        # an empty directory.
        writers[NO_MONTH] = df.write_parquet
    replace_months(path, writers, full=months is None)
