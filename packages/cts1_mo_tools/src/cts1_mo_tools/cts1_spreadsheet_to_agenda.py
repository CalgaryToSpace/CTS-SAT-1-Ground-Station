import contextlib
import csv
import random
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import openpyxl
import polars as pl
import tyro

import logging

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

_TIME_FORMATS = ("%H:%M:%S", "%H:%M")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _parse_time_of_day(value: str) -> time:
    for fmt in _TIME_FORMATS:
        with contextlib.suppress(ValueError):
            return datetime.strptime(value, fmt).replace(tzinfo=UTC).time()

    msg = f"Invalid time format: {value}"
    raise ValueError(msg)


def parse_time(
    mission_date: str,
    mission_start: datetime,
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    if value.lower() == "past":
        year = int(mission_date.split("-", maxsplit=1)[0])
        return datetime(1970, 1, 1, tzinfo=UTC)

    try:
        if float(value) == 0:
            return datetime(1970, 1, 1, tzinfo=UTC)
    except ValueError:
        pass

    parsed_time = _parse_time_of_day(value)
    year, month, day = map(int, mission_date.split("-"))
    candidate = datetime(
        year,
        month,
        day,
        parsed_time.hour,
        parsed_time.minute,
        parsed_time.second,
        tzinfo=UTC,
    )

    if candidate < mission_start:
        candidate += timedelta(days=1)

    return candidate


def epoch_ms(dt: datetime | None) -> int:
    if dt is None:
        return 0
    return int(dt.timestamp() * 1000)


def parse_interval(text: str | None) -> timedelta | None:
    if not text:
        return None

    number, units = str(text).strip().lower().split()
    value = float(number)

    if units.startswith("sec"):
        return timedelta(seconds=value)

    if units.startswith("min"):
        return timedelta(minutes=value)

    msg = f"Unknown interval format: {text}"
    raise ValueError(msg)

def parse_repeat_random(value: str | None, cmd: str) -> int:
    # Empty value check
    if value is None or value.strip() == "":
        logger.warning(f"Empty repeat or random value for command {cmd}. Default: 0.")

    # Negative value check
    elif float(value.strip()) < 0:
        logger.warning(f"Negative repeat or random value for command {cmd}. Default: 0.")

    # Non-integer value check
    elif not float(value.strip()).is_integer():
        logger.warning(f"Non-integer repeat or random value for command {cmd} was truncated.")

    return to_int(value)

def to_int(value: str | None, default: int = 0) -> int:
    text = "" if value is None else str(value).strip()

    if not text or text.lower() == "nan":
        return default

    return int(float(text))


def substitute(text: str | None, mission_date: str) -> str:
    return "" if text is None else str(text).replace("{DATE}", mission_date)


TSSENT_RE = re.compile(r"@tssent=(\d+)")
TSEXEC_RE = re.compile(r"@tsexec=(\d+)")


def format_epoch_ms(ms_text: str) -> str:
    dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=int(ms_text))
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def annotate_readable(line: str) -> str:
    """Prefix a command line with human-readable UTC times for @tssent=/@tsexec="""
    tssent_match = TSSENT_RE.search(line)
    tsexec_match = TSEXEC_RE.search(line)

    if not tssent_match and not tsexec_match:
        return line

    parts: list[str] = []

    if tssent_match:
        parts.append(f"tssent: {format_epoch_ms(tssent_match.group(1))}")

    if tsexec_match:
        parts.append(f"tsexec: {format_epoch_ms(tsexec_match.group(1))}")

    return " | ".join(parts) + " | " + line


def format_command(
    command: str,
    tssent: datetime,
    tsexec: datetime,
    resp: str,
) -> str:
    command = command.strip("!")

    line = f"{command}@tssent={epoch_ms(tssent)}@tsexec={epoch_ms(tsexec)}"

    if resp:
        line += f"@resp_fname={resp}"

    return line + "!"


# ---------------------------------------------------------------------
# Spreadsheet Loader
# ---------------------------------------------------------------------


def extract_date(value: str | None) -> str | None:
    """Pull a YYYY-MM-DD date out of a cell that may hold extra text
    (e.g. "2026-07-05T07:03:00" or "2026-07-05 00:00:00").
    """
    if not value:
        return None

    match = DATE_RE.search(str(value))
    return match.group(1) if match else None


def _parse_mission_start(mission_date: str, start_utc_value: str | None) -> datetime:
    """Resolve the mission's exact start datetime, used as the anchor for
    the midnight-rollover logic in parse_time.
    """
    if start_utc_value:
        try:
            parsed = datetime.fromisoformat(start_utc_value.strip())
        except ValueError as exc:
            msg = f"Could not parse Start UTC value: {start_utc_value!r}"
            raise ValueError(msg) from exc

        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    year, month, day = map(int, mission_date.split("-"))
    return datetime(year, month, day, tzinfo=UTC)


def _normalize_header(text: str) -> str:
    return " ".join(text.split())


def _cell_to_str(value: object) -> str:
    """Stringify a raw openpyxl cell value."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _read_raw_rows(path: Path) -> list[list[str]]:
    if path.suffix.lower() == ".xlsx":
        workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
        worksheet = workbook.active

        if worksheet is None:
            return []

        return [
            [_cell_to_str(cell) for cell in row]
            for row in worksheet.iter_rows(values_only=True)
        ]

    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


@dataclass
class _HeaderInfo:
    row: int
    col: int
    date_value: str | None
    start_utc_value: str | None


def _find_header(rows: list[list[str]]) -> _HeaderInfo:
    """Scan rows for the "Date"/"Start UTC" labels and the "Mode" header"""
    date_value: str | None = None
    start_utc_value: str | None = None

    for i, row in enumerate(rows):
        if not row:
            continue

        for j, cell in enumerate(row):
            label = _normalize_header(str(cell).strip())

            if label.lower() == "date" and date_value is None and j + 1 < len(row):
                date_value = row[j + 1]
            elif (
                label.lower() == "start utc"
                and start_utc_value is None
                and j + 1 < len(row)
            ):
                start_utc_value = row[j + 1]
            elif label == "Mode":
                return _HeaderInfo(i, j, date_value, start_utc_value)

    msg = "Could not locate command table."
    raise ValueError(msg)


def _extract_commands(
    rows: list[list[str]],
    header: _HeaderInfo,
) -> list[dict[str, str]]:
    headers = [_normalize_header(c.strip()) for c in rows[header.row][header.col :]]
    commands: list[dict[str, str]] = []

    for row in rows[header.row + 1 :]:
        sliced = row[header.col :]

        if not any(str(x).strip() for x in sliced):
            continue

        sliced = sliced + [""] * (len(headers) - len(sliced))
        commands.append(dict(zip(headers, sliced, strict=False)))

    return commands


def load_sheet(path: Path) -> tuple[str, datetime, list[dict[str, str]]]:
    rows = _read_raw_rows(path)
    header = _find_header(rows)

    mission_date = (
        extract_date(header.date_value) or extract_date(header.start_utc_value) or ""
    )

    if not mission_date:
        msg = "Mission date not found."
        raise ValueError(msg)

    mission_start = _parse_mission_start(mission_date, header.start_utc_value)

    return mission_date, mission_start, _extract_commands(rows, header)


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------


@dataclass
class _RowContext:
    row: dict[str, str]
    mission_date: str
    mission_start: datetime
    cmd: str
    resp: str


def _build_single_entries(
    ctx: _RowContext,
    repeat: int,
    random_repeat: int,
) -> tuple[list[tuple[int, str]], list[str]]:
    tssent = parse_time(ctx.mission_date, ctx.mission_start, ctx.row["tssent (UTC)"])
    if tssent is None:
        msg = f"Missing tssent for {ctx.cmd}"
        raise ValueError(msg)
    tsexec = parse_time(
        ctx.mission_date, ctx.mission_start, ctx.row["tsexec Start (UTC)"]
    )
    if tsexec is None:
        msg = f"Missing tsexec for {ctx.cmd}"
        raise ValueError(msg)

    command = format_command(ctx.cmd, tssent, tsexec, ctx.resp)

    entries = [(epoch_ms(tssent), command) for _ in range(repeat)]
    random_entries = [command for _ in range(random_repeat)]

    return entries, random_entries


def _build_interval_entries(ctx: _RowContext) -> list[tuple[int, str]]:
    start = parse_time(
        ctx.mission_date, ctx.mission_start, ctx.row["tsexec Start (UTC)"]
    )
    end = parse_time(ctx.mission_date, ctx.mission_start, ctx.row["tsexec End (UTC)"])
    interval = parse_interval(ctx.row["Interval"])

    if start is None or end is None or interval is None:
        msg = f"Interval missing for {ctx.cmd}"
        raise ValueError(msg)

    entries: list[tuple[int, str]] = []
    current = start

    while current <= end:
        command = format_command(ctx.cmd, current, current, ctx.resp)
        entries.append((epoch_ms(current), command))
        current += interval

    return entries


def build_agenda(
    mission_date: str,
    mission_start: datetime,
    rows: list[dict[str, str]],
) -> list[str]:

    agenda: list[
        tuple[int, str]
    ] = []  # Tuple of (epoch_ms of tssent, the formatted command line)
    random_commands: list[str] = []

    for row in rows:
        mode = row["Mode"].strip().lower()
        cmd = substitute(row["Telecommand"], mission_date)
        resp = substitute(row["resp_fname"], mission_date)
        ctx = _RowContext(
            row=row,
            mission_date=mission_date,
            mission_start=mission_start,
            cmd=cmd,
            resp=resp,
        )

        if mode == "single":
            repeat = parse_repeat_random(row["Repeat"], cmd)
            random_repeat = parse_repeat_random(row["Random"], cmd)
            entries, random_entries = _build_single_entries(ctx, repeat, random_repeat)
            agenda.extend(entries)
            random_commands.extend(random_entries)

        elif mode == "interval":
            agenda.extend(_build_interval_entries(ctx))
            if (row["Repeat"].strip() != "") or (row["Random"].strip() != ""):
                logger.warning(
                    f"Repeat/Random values are ignored for Interval mode for command {cmd}."
                )
        else:
            msg = "Invalid Mode (Options: Single or Interval)"
            raise ValueError(msg)

    agenda.sort(key=lambda x: x[0])

    lines = [cmd for _, cmd in agenda]

    for cmd in random_commands:
        lines.insert(random.randint(0, len(lines)), cmd)  # noqa: S311

    return lines


def build_summary_comment(rows: list[dict[str, Any]]) -> str:
    """Render a dataframe preview of the input file's commands, commented out
    with a leading "#" on every line so it can be prepended to the agenda.
    """
    df = pl.DataFrame(rows)

    df = df.drop([col for col in df.columns if len(col.strip()) <= 1])

    df = df.with_columns(
        # Shorten the Telecommand column.
        pl.col("Telecommand").str.strip_prefix("CTS1+").str.strip_suffix("!")
    )

    with pl.Config(
        ascii_tables=True,
        tbl_rows=-1,  # Show all rows.
        tbl_cols=-1,  # Show all columns.
        tbl_width_chars=400,  # Allow table to be extra wide.
        tbl_hide_dataframe_shape=True,  # Hide the row/column count prefix.
        fmt_str_lengths=180,  # Allow each column up to 180 characters.
    ):
        table_text = str(df)

    output_comment = "\n".join(f"# {line}" for line in table_text.splitlines())

    output_comment += "\n\n"

    return output_comment


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def spreadsheet_file_to_agenda_file(
    input_file: Path,
    output_file: Path = Path("agenda_output.txt"),
    seed: int | None = None,
    *,
    readable: bool = False,
    summary_table: bool = True,
) -> None:
    """Generate an agenda file from a spreadsheet.

    Parameters
    ----------
    input_file : Path
        The path to the spreadsheet file.
    output_file : Path, default=Path("agenda_output.txt")
        The path to the output file.
    seed : int | None, default=None
        The random seed to use. If None, a random seed will be used.
    readable : bool, default=False
        Whether to generate a readable version of the agenda, in addition to the normal
        agenda.
    summary_table : bool
        Whether to include a summary table of the commands in the output file as a
        comment.
    """
    if seed is not None:
        random.seed(seed)

    mission_date, mission_start, rows = load_sheet(input_file)

    agenda = build_agenda(mission_date, mission_start, rows)

    summary_comment = build_summary_comment(rows) + "\n\n" if summary_table else ""

    output_file.write_text(
        summary_comment + "\n".join(agenda) + "\n",
        encoding="utf-8",
    )

    print(f"Generated {len(agenda)} commands → {output_file}")  # noqa: T201

    if readable:
        readable_path = output_file.with_stem(f"{output_file.stem}_readable")
        readable_lines = [annotate_readable(line) for line in agenda]
        readable_path.write_text(
            summary_comment + "\n".join(readable_lines) + "\n",
            encoding="utf-8",
        )

        print(f"Saved readable version → {readable_path}")  # noqa: T201


def main() -> None:
    tyro.cli(spreadsheet_file_to_agenda_file)


if __name__ == "__main__":
    main()
