from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import tyro

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Number of ":"-separated parts in a "HH:MM" / "HH:MM:SS" time string.
_TIME_PARTS_HOUR_MINUTE = 2
_TIME_PARTS_HOUR_MINUTE_SECOND = 3


@dataclass
class Args:
    input_file: Path
    output_file: Path = Path("agenda_output.txt")
    seed: int | None = None
    readable: bool = False
    """Also write a human-readable copy with tssent/tsexec decoded to UTC."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_time(date: str, value: str | None) -> datetime | None:
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    year, month, day = map(int, date.split("-"))

    if value.lower() == "past":
        return datetime(year, 1, 1, tzinfo=UTC)

    try:
        if float(value) == 0:
            # Sentinel: 0 means "no real timestamp" and should be emitted
            # literally as epoch (0 ms), not resolved to the current time.
            return datetime(1970, 1, 1, tzinfo=UTC)
    except ValueError:
        pass

    parts = value.split(":")

    if len(parts) == _TIME_PARTS_HOUR_MINUTE:
        hour, minute = map(int, parts)
        second = 0
    elif len(parts) == _TIME_PARTS_HOUR_MINUTE_SECOND:
        hour, minute, second = map(int, parts)
    else:
        msg = f"Invalid time format: {value}"
        raise ValueError(msg)

    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def epoch_ms(dt: datetime | None) -> int:
    if dt is None:
        return 0
    return int(dt.timestamp() * 1000)


def parse_interval(text: str | None) -> timedelta | None:
    if not text:
        return None

    number, units = str(text).strip().lower().split()
    value = int(float(number))

    if units.startswith("sec"):
        return timedelta(seconds=value)

    if units.startswith("min"):
        return timedelta(minutes=value)

    msg = f"Unknown interval format: {text}"
    raise ValueError(msg)


def to_int(value: str | None, default: int = 0) -> int:
    """Parse an int from a cell that may be blank or a float-like string.

    xlsx cells read back as strings (e.g. via dtype=str) frequently come
    through as "5.0" instead of "5", which plain int() cannot parse.
    """
    text = "" if value is None else str(value).strip()

    if not text or text.lower() == "nan":
        return default

    return int(float(text))


def substitute(text: str | None, mission_date: str) -> str:
    return "" if text is None else str(text).replace("{DATE}", mission_date)


TSSENT_RE = re.compile(r"@tssent=(\d+)")
TSEXEC_RE = re.compile(r"@tsexec=(\d+)")


def format_epoch_ms(ms_text: str) -> str:
    dt = datetime.fromtimestamp(int(ms_text) / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def annotate_readable(line: str) -> str:
    """Prefix a command line with human-readable UTC times for @tssent=/@tsexec="""
    tssent_match = TSSENT_RE.search(line)
    tsexec_match = TSEXEC_RE.search(line)

    if not tssent_match and not tsexec_match:
        return line

    parts: list[str] = []

    if tssent_match:
        parts.append(f"tssent: {format_epoch_ms(tssent_match.group(1))} UTC")

    if tsexec_match:
        parts.append(f"tsexec: {format_epoch_ms(tsexec_match.group(1))} UTC")

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


def _read_raw_rows(path: Path) -> list[list[str]]:
    if path.suffix.lower() == ".xlsx":
        df = pl.read_excel(path, has_header=False, engine="openpyxl")
        return [list(row) for row in df.fill_null("").rows()]

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
            label = str(cell).strip()

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
    headers = [c.strip() for c in rows[header.row][header.col :]]
    commands: list[dict[str, str]] = []

    for row in rows[header.row + 1 :]:
        sliced = row[header.col :]

        if not any(str(x).strip() for x in sliced):
            continue

        sliced = sliced + [""] * (len(headers) - len(sliced))
        commands.append(dict(zip(headers, sliced, strict=False)))

    return commands


def load_sheet(path: Path) -> tuple[str, list[dict[str, str]]]:
    rows = _read_raw_rows(path)
    header = _find_header(rows)

    mission_date = (
        extract_date(header.date_value) or extract_date(header.start_utc_value) or ""
    )

    return mission_date, _extract_commands(rows, header)


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------


@dataclass
class _RowContext:
    row: dict[str, str]
    mission_date: str
    cmd: str
    resp: str


def _build_single_entries(
    ctx: _RowContext,
    repeat: int,
    random_repeat: int,
) -> tuple[list[tuple[int, str]], list[str]]:
    tssent = parse_time(ctx.mission_date, ctx.row["tssent (UTC)"])
    if tssent is None:
        msg = f"Missing tssent for {ctx.cmd}"
        raise ValueError(msg)

    tsexec = parse_time(ctx.mission_date, ctx.row["tsexec (UTC)"])
    if tsexec is None:
        msg = f"Missing tsexec for {ctx.cmd}"
        raise ValueError(msg)

    command = format_command(ctx.cmd, tssent, tsexec, ctx.resp)

    entries = [(epoch_ms(tssent), command) for _ in range(repeat)]
    random_entries = [command for _ in range(random_repeat)]

    return entries, random_entries


def _build_interval_entries(ctx: _RowContext) -> list[tuple[int, str]]:
    start = parse_time(ctx.mission_date, ctx.row["Window Start"])
    end = parse_time(ctx.mission_date, ctx.row["Window End"])
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
    rows: list[dict[str, str]],
) -> list[str]:
    agenda: list[tuple[int, str]] = []
    random_commands: list[str] = []

    for row in rows:
        mode = row["Mode"].strip().lower()
        cmd = substitute(row["Telecommand"], mission_date)
        resp = substitute(row["resp_fname"], mission_date)
        ctx = _RowContext(row=row, mission_date=mission_date, cmd=cmd, resp=resp)

        if mode == "single":
            repeat = to_int(row["Repeat"])
            random_repeat = to_int(row["Random"])
            entries, random_entries = _build_single_entries(ctx, repeat, random_repeat)
            agenda.extend(entries)
            random_commands.extend(random_entries)

        elif mode == "interval":
            agenda.extend(_build_interval_entries(ctx))

    agenda.sort(key=lambda x: x[0])

    lines = [cmd for _, cmd in agenda]

    for cmd in random_commands:
        lines.insert(random.randint(0, len(lines)), cmd)  # noqa: S311

    return lines


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = tyro.cli(Args)

    if args.seed is not None:
        random.seed(args.seed)

    mission_date, rows = load_sheet(args.input_file)

    if not mission_date:
        msg = "Mission date not found."
        raise ValueError(msg)

    agenda = build_agenda(mission_date, rows)

    args.output_file.write_text("\n".join(agenda), encoding="utf-8")

    print(f"Generated {len(agenda)} commands → {args.output_file}")  # noqa: T201

    if args.readable:
        readable_path = args.output_file.with_name(
            f"{args.output_file.stem}_readable{args.output_file.suffix}"
        )
        readable_lines = [annotate_readable(line) for line in agenda]
        readable_path.write_text("\n".join(readable_lines), encoding="utf-8")

        print(f"Saved readable version → {readable_path}")  # noqa: T201


if __name__ == "__main__":
    main()
