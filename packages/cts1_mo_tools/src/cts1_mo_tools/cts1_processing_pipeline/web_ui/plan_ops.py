"""Data layer for the web UI's Plan page: the collaborative command
plan (a SQLite file in the data dir, shared by every browser), its
name/password login, agenda generation, and the auto-populate generators.

The plan holds the same rows as the mission-ops spreadsheet
`cts1_spreadsheet_to_agenda` reads (Mode, tsexec Start/End, Interval,
Telecommand, ...), each assigned to the RAO uplink pass (by AOS) it goes up
on -- or to none yet, to be uplinked later. A pass's agenda is built by
`cts1_spreadsheet_to_agenda.build_agenda`, exactly as from a spreadsheet.

No NiceGUI here -- see `plan_ops_page`.
"""

from __future__ import annotations

__all__ = [
    "DB_FILENAME",
    "USERS_FILENAME",
    "PlanOpsStore",
    "build_pass_agenda",
    "check_command",
    "check_login",
    "dark_region_photo_rows",
    "load_incomplete_files",
    "load_satnogs_windows",
    "load_users",
    "missing_chunk_targets",
    "satnogs_downlink_rows",
    "satnogs_window_rows",
    "table_rows",
    "to_tsv",
    "to_xlsx",
]

import csv
import hmac
import io
import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import openpyxl
import polars as pl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from cts1_mo_tools import cts1_plan_mpi_ops as mpi_ops
from cts1_mo_tools.cts1_processing_pipeline.file_reassembly import ByteStatus
from cts1_mo_tools.cts1_processing_pipeline.step_11_reassemble_downlinked_files import (
    pipeline as step_11_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_12_fetch_satnogs_schedule import (
    pipeline as step_12_pipeline,
)
from cts1_mo_tools.cts1_satnogs_interval import build_coverage_windows
from cts1_mo_tools.cts1_spreadsheet_to_agenda import (
    build_agenda,
    build_summary_comment,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import satkit as sk

DB_FILENAME: Final = "plan_ops.sqlite"
USERS_FILENAME: Final = "plan_ops_users.json"

TIME_FORMAT: Final = "%Y-%m-%d %H:%M:%S"
"""How tsexec times are stored and edited: always UTC."""

# Columns a user may edit by name (they arrive from the browser), and their
# defaults -- a spreadsheet row's, as in the template.
FIELD_DEFAULTS: Final = {
    "mode": "Single",
    "tsexec_start": "",
    "tsexec_end": "",
    "interval": "",
    "telecommand": "",
    "resp_fname": "",
    "repeat": "1",
    "random": "5",
    "notes": "",
}
EDITABLE_FIELDS: Final = tuple(FIELD_DEFAULTS)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY,
    uplink_aos TEXT,  -- the RAO pass it goes up on (TIME_FORMAT); NULL = later
    mode TEXT NOT NULL DEFAULT 'Single',
    tsexec_start TEXT NOT NULL DEFAULT '',  -- TIME_FORMAT, UTC; '' = at once
    tsexec_end TEXT NOT NULL DEFAULT '',
    interval TEXT NOT NULL DEFAULT '',  -- like "2 min" or "30 sec"
    telecommand TEXT NOT NULL DEFAULT '',
    resp_fname TEXT NOT NULL DEFAULT '',
    repeat TEXT NOT NULL DEFAULT '1',
    random TEXT NOT NULL DEFAULT '5',
    notes TEXT NOT NULL DEFAULT '',
    updated_by TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
-- Bumped by every write, so each open page can tell when to reload.
CREATE TABLE IF NOT EXISTS revision (n INTEGER NOT NULL);
INSERT INTO revision SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM revision);
"""


def _now() -> str:
    return datetime.now(UTC).strftime(TIME_FORMAT)


def format_time(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(TIME_FORMAT)


def parse_stored_time(text: str) -> datetime:
    return datetime.strptime(text.strip(), TIME_FORMAT).replace(tzinfo=UTC)


class PlanOpsStore:
    """The shared command plan, in a SQLite file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, sql: str, params: Iterable[Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executemany(sql, params)
            conn.execute("UPDATE revision SET n = n + 1")

    def revision(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT n FROM revision").fetchone()[0]

    def list_commands(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM commands ORDER BY uplink_aos, tsexec_start, id"
            )
            return [dict(r) for r in rows]

    def add(self, rows: list[dict[str, Any]], user: str) -> None:
        now = _now()
        self._write(
            f"INSERT INTO commands (uplink_aos, {', '.join(EDITABLE_FIELDS)}, "  # noqa: S608
            f"updated_by, updated_at) VALUES ({', '.join('?' * 12)})",
            (
                (
                    row.get("uplink_aos"),
                    *(str(row.get(f, d)) for f, d in FIELD_DEFAULTS.items()),
                    user,
                    now,
                )
                for row in rows
            ),
        )

    def update_fields(self, changes: list[tuple[int, str, str]], user: str) -> None:
        """Set each `(command_id, field, value)`, in one write (e.g. a paste)."""
        # The field names come from the browser.
        if bad := {field for _, field, _ in changes} - set(EDITABLE_FIELDS):
            msg = f"Not editable field(s): {sorted(bad)!r}"
            raise ValueError(msg)
        now = _now()
        with closing(self._connect()) as conn, conn:
            for command_id, field, value in changes:
                conn.execute(
                    f"UPDATE commands SET {field} = ?, updated_by = ?, "  # noqa: S608
                    "updated_at = ? WHERE id = ?",
                    (value, user, now, command_id),
                )
            conn.execute("UPDATE revision SET n = n + 1")

    def move(self, ids: list[int], uplink_aos: str | None, user: str) -> None:
        now = _now()
        self._write(
            "UPDATE commands SET uplink_aos = ?, updated_by = ?, updated_at = ? "
            "WHERE id = ?",
            ((uplink_aos, user, now, i) for i in ids),
        )

    def delete(self, ids: list[int]) -> None:
        self._write("DELETE FROM commands WHERE id = ?", ((i,) for i in ids))


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------


def load_users(path: Path) -> dict[str, str]:
    """The `{"name": "password", ...}` JSON file; empty if there's none."""
    if not path.exists():
        return {}
    users = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(users, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v
        for k, v in users.items()  # pyright: ignore[reportUnknownVariableType]
    ):
        msg = f'{path} must be a JSON object of "name": "password" pairs.'
        raise ValueError(msg)
    return users  # pyright: ignore[reportUnknownVariableType]


def check_login(path: Path, name: str, password: str) -> bool:
    expected = load_users(path).get(name)
    return expected is not None and hmac.compare_digest(
        expected.encode(), password.encode()
    )


# ---------------------------------------------------------------------
# Agenda
# ---------------------------------------------------------------------


def _sheet_time(text: str, pass_start: datetime, label: str) -> str:
    """A stored tsexec as `cts1_spreadsheet_to_agenda` takes it: a time of
    day within 24 h after the pass start, or "0" for "at once".
    """
    if not text.strip() or text.strip() in {"0", "past"}:
        return "0"
    dt = parse_stored_time(text)
    # The spreadsheet format only has times of day, rolled over past midnight.
    if not pass_start <= dt < pass_start + timedelta(days=1):
        msg = f"{label} {text} isn't within 24 h after the uplink pass's AOS."
        raise ValueError(msg)
    return dt.strftime("%H:%M:%S")


def check_command(command: dict[str, Any], uplink_los: datetime | None) -> str:
    """A warning for a command that would run before its uplink pass's LOS
    (so before it's even been uplinked), or with an unreadable tsexec;
    "" if none. A blank or "0" tsexec (run at once) is deliberate.
    """
    text = command["tsexec_start"].strip()
    if text in {"", "0", "past"}:
        return ""
    try:
        tsexec = parse_stored_time(text)
    except ValueError:
        return "⚠ unreadable tsexec start"
    if uplink_los is not None and tsexec < uplink_los:
        return f"⚠ runs before LOS ({uplink_los:%H:%M:%S}Z)"
    return ""


def build_pass_agenda(commands: list[dict[str, Any]], pass_aos: datetime) -> str:
    """The agenda file text for one pass's commands, as
    `cts1_spreadsheet_to_agenda` would write it from the same spreadsheet.

    Raises:
        ValueError: On a command the agenda can't be built with.
    """
    rows = [
        {
            "Mode": c["mode"],
            "tsexec Start (UTC)": _sheet_time(
                c["tsexec_start"], pass_aos, f"{c['telecommand']}: tsexec start"
            ),
            "tsexec End (UTC)": c["tsexec_end"]
            and _sheet_time(
                c["tsexec_end"], pass_aos, f"{c['telecommand']}: tsexec end"
            ),
            "Interval": c["interval"],
            "Telecommand": c["telecommand"],
            "resp_fname": c["resp_fname"],
            "Repeat": c["repeat"],
            "Random": c["random"],
            "TCMD Notes": c["notes"],
        }
        for c in commands
    ]
    if not rows:
        return ""
    agenda = build_agenda(pass_aos.strftime("%Y-%m-%d"), pass_aos, rows)
    return build_summary_comment(rows) + "\n\n" + "\n".join(agenda) + "\n"


# ---------------------------------------------------------------------
# Copy/download as a spreadsheet
# ---------------------------------------------------------------------

TABLE_COLUMNS: Final = {
    "uplink": "Uplink pass",
    "mode": "Mode",
    "tsexec_start": "tsexec Start (UTC)",
    "tsexec_end": "tsexec End (UTC)",
    "interval": "Interval",
    "telecommand": "Telecommand",
    "resp_fname": "resp_fname",
    "repeat": "Repeat",
    "random": "Random",
    "notes": "TCMD Notes",
}
"""Each command's field -> its spreadsheet column header."""


def table_rows(commands: list[dict[str, Any]]) -> list[list[str]]:
    """`commands` (each with an `uplink` label) as a header row + rows."""
    return [list(TABLE_COLUMNS.values())] + [
        [str(c.get(k) or "") for k in TABLE_COLUMNS] for c in commands
    ]


def to_tsv(rows: list[list[str]]) -> str:
    """Tab-separated, quoted as Excel/Google Sheets expect on paste."""
    out = io.StringIO()
    csv.writer(out, dialect="excel-tab", lineterminator="\n").writerows(rows)
    return out.getvalue()


def to_xlsx(rows: list[list[str]]) -> bytes:
    """An .xlsx of `rows` (the first one a bold header), in a monospaced font."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Plan"
    for row in rows:
        sheet.append(row)
    for cell in (c for row in sheet.iter_rows() for c in row):
        cell.font = Font(name="Consolas", bold=cell.row == 1)
    for i, column in enumerate(zip(*rows, strict=True), start=1):
        width = min(max(len(v) for v in column) + 2, 100)
        sheet.column_dimensions[get_column_letter(i)].width = width
    sheet.freeze_panes = "A2"
    out = io.BytesIO()
    workbook.save(out)
    return out.getvalue()


# ---------------------------------------------------------------------
# Auto-populate generators
# ---------------------------------------------------------------------

BULK_DOWNLINK_COMMAND: Final = (
    "CTS1+exec_blob_from_fs(blobs/bulk_downlink_start_v3.blob,0,{target})"
)
"""`target` is "<path>;<start_offset>;<length>"."""

DEFAULT_CHUNK_BYTES: Final = 30_000
"""How much a missing-chunk downlink sends, unless reconfigured."""

DOWNLINK_BYTES_PER_SEC: Final = 750
"""Bulk downlink rate: a downlink's repeat interval is its length over this."""

CAMERA_SETUP_COMMAND: Final = "CTS1+camera_setup()"
CAMERA_CAPTURE_COMMAND: Final = (
    "CTS1+exec_blob_from_fs(blobs/camera_capture_v2.blob,0,{location};d)"
)
CAMERA_FILENAME_FORMAT: Final = "camera/%Y-%m/%Y-%m-%d_%H%M%SZ.img"


def load_incomplete_files(data_dir: Path) -> pl.DataFrame:
    """Step 11's partly downlinked files, most recently downlinked first."""
    path = data_dir / step_11_pipeline.DOWNLINKED_FILES_OUTPUT_FILENAME
    schema = {
        "file_path": pl.String,
        "byte_length": pl.Int64,
        "missing_bytes": pl.Int64,
        "segments": pl.String,
        "last_downlinked_at": pl.Datetime("us", "UTC"),
    }
    if not path.exists():
        return pl.DataFrame(schema=schema)
    return (
        pl.scan_parquet(path)
        .select(list(schema))
        .filter(pl.col("missing_bytes") > 0)
        .sort("last_downlinked_at", descending=True)
        .collect()
    )


def downlink_ranges(
    missing: list[tuple[int, int]], file_size: int, chunk_bytes: int
) -> list[tuple[int, int]]:
    """`(start, length)` chunks of `chunk_bytes` (or the whole file, if it's
    smaller) covering every `(start, end)` missing range -- so neighbouring
    small gaps share a chunk, and big ones are split into several.
    """
    if chunk_bytes <= 0:
        msg = f"Chunk size must be positive, not {chunk_bytes}."
        raise ValueError(msg)
    chunks: list[tuple[int, int]] = []
    covered = 0  # Everything before this is in a chunk already.
    for gap_start, gap_end in sorted(missing):
        gap_end = min(gap_end, file_size)  # noqa: PLW2901 -- else it never ends.
        pos = max(gap_start, covered)
        while pos < gap_end:
            start = max(0, min(pos, file_size - chunk_bytes))  # Fit inside the file.
            chunks.append((start, min(chunk_bytes, file_size)))
            covered = pos = start + chunk_bytes
    return chunks


def missing_chunk_targets(files: pl.DataFrame, chunk_bytes: int) -> list[str]:
    """Bulk downlink targets ("<path>;<start>;<length>") covering each
    (partly downlinked) file's missing bytes, from step 11's
    `downlinked_files` rows, in row order.
    """
    targets: list[str] = []
    for row in files.filter(pl.col("missing_bytes") > 0).iter_rows(named=True):
        missing = [
            (s.start, s.end)
            for s in step_11_pipeline.decode_segments(row["segments"])
            if s.status is ByteStatus.MISSING
        ]
        targets.extend(
            f"{row['file_path']};{start};{length}"
            for start, length in downlink_ranges(
                missing, row["byte_length"], chunk_bytes
            )
        )
    return targets


def load_satnogs_windows(
    data_dir: Path, start: datetime, end: datetime
) -> list[tuple[datetime, datetime, str]]:
    """`(start, end, note)` of each time SatNOGS stations are scheduled to
    listen between `start` and `end`, clipped to them, from step 12's
    schedule.

    Raises:
        LookupError: If step 12 hasn't fetched the schedule yet.
    """
    path = data_dir / step_12_pipeline.OUTPUT_FILENAME
    if not path.exists():
        msg = f"No SatNOGS schedule yet -- run step 12 (it writes {path.name})."
        raise LookupError(msg)
    observations = [
        {
            "start": r["start"].isoformat(),
            "end": r["end"].isoformat(),
            "ground_station": r["ground_station_id"],
            "id": r["observation_id"],
        }
        for r in pl.read_parquet(path).iter_rows(named=True)
    ]
    return [
        (max(w.start, start), min(w.end, end), f"{w.station_count} station(s)")
        for w in build_coverage_windows(observations)
        if w.end > start and w.start < end
    ]


def satnogs_window_rows(
    windows: list[tuple[datetime, datetime, str]],
) -> list[dict[str, Any]]:
    """An Interval row per SatNOGS coverage window `(start, end, note)`, with
    only its times filled in -- the telecommand is the operator's to write.
    """
    return [
        {
            "mode": "Interval",
            "tsexec_start": format_time(start),
            "tsexec_end": format_time(end),
            "repeat": "",
            "random": "",
            "notes": f"SatNOGS window ({note})",
        }
        for start, end, note in windows
    ]


def satnogs_downlink_rows(
    windows: list[tuple[datetime, datetime, str]], target: str
) -> list[dict[str, Any]]:
    """`satnogs_window_rows`, each downlinking `target` ("<path>;<start>;
    <length>") over and over (for SatNOGS's lossy reception), as often as
    `DOWNLINK_BYTES_PER_SEC` lets one downlink finish before the next.
    """
    length = int(target.rsplit(";", 1)[1])
    interval = f"{math.ceil(length / DOWNLINK_BYTES_PER_SEC)} sec"
    return [
        row
        | {
            "interval": interval,
            "telecommand": BULK_DOWNLINK_COMMAND.format(target=target),
            "notes": f"Downlink {target} during {row['notes']}",
        }
        for row in satnogs_window_rows(windows)
    ]


def dark_region_photo_rows(  # noqa: PLR0913
    tle: sk.TLE,
    start: datetime,
    end: datetime,
    *,
    latitude_range: str | None,
    longitude_range: str | None,
    every_sec: float,
    eclipse_start: mpi_ops.EclipseStart = "penumbra",
    setup_lead_sec: float = 30,
) -> list[dict[str, Any]]:
    """A capture every `every_sec` while the satellite's over the region in
    Earth's shadow (as `cts1_plan_mpi_ops` finds it), between `start` and
    `end` -- each with its own `camera_setup`, `setup_lead_sec` before it.
    """
    region = mpi_ops.Region(
        "photo",
        mpi_ops._parse_range(latitude_range),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        mpi_ops._parse_range(longitude_range),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    step = timedelta(seconds=mpi_ops.TIME_STEP_SEC)
    times = [start + i * step for i in range(int((end - start) / step) + 1)]
    eph = mpi_ops.propagate(tle, times)
    mask = np.asarray(
        mpi_ops._in_region_mask(region, eph)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        & mpi_ops._in_shadow_mask(eclipse_start, eph)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    rows: list[dict[str, Any]] = []
    for enter, exit_ in mpi_ops._mask_to_intervals(mask, times):  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        t = enter
        while t <= exit_:
            location = t.strftime(CAMERA_FILENAME_FORMAT)
            rows.extend(
                (
                    {
                        "mode": "Single",
                        "tsexec_start": format_time(
                            t - timedelta(seconds=setup_lead_sec)
                        ),
                        "telecommand": CAMERA_SETUP_COMMAND,
                        "notes": f"Set up camera for {location}",
                    },
                    {
                        "mode": "Single",
                        "tsexec_start": format_time(t),
                        "telecommand": CAMERA_CAPTURE_COMMAND.format(location=location),
                        "notes": "Picture over dark region",
                    },
                )
            )
            t += timedelta(seconds=every_sec)
    return rows
