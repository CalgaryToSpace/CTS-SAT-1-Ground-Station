from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import tyro

UTC = timezone.utc


@dataclass
class Args:
    input_file: Path
    output_file: Path = Path("agenda_output.txt")
    seed: int | None = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def epoch_ms(dt: datetime | None) -> int:
    if dt is None:
        return 0
    return int(dt.timestamp() * 1000)


def parse_time(date: str, value: str | None) -> datetime | None:
    if value is None:
        return None

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return None

    if value.lower() == "past":
        return datetime(1970, 1, 1, tzinfo=UTC)

    return datetime.fromisoformat(f"{date}").replace(tzinfo=UTC)


def parse_interval(text: str | None) -> timedelta | None:
    if text is None:
        return None

    text = str(text).strip()

    if text == "":
        return None

    number, units = text.lower().split()

    value = int(number)

    if units.startswith("sec"):
        return timedelta(seconds=value)

    if units.startswith("min"):
        return timedelta(minutes=value)

    raise ValueError(f"Unknown interval: {text}")


def substitute(text: str | None, mission_date: str) -> str:
    if text is None:
        return ""

    return str(text).replace("{DATE}", mission_date)


def random_times(
    start: datetime,
    end: datetime,
    count: int,
) -> list[datetime]:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    return sorted(
        datetime.fromtimestamp(
            random.randint(start_ts, end_ts),
            tz=UTC,
        )
        for _ in range(count)
    )


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

    if not line.endswith("!"):
        line += "!"

    return line


# ---------------------------------------------------------------------
# Spreadsheet Loader
# ---------------------------------------------------------------------


def load_sheet(path: Path) -> tuple[str, list[dict[str, str]]]:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, header=None, dtype=str).fillna("")
        rows = df.values.tolist()

    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))

    mission_date = ""

    header_row = None

    for i, row in enumerate(rows):
        if not row:
            continue

        first = row[0].strip()

        if first == "Start UTC":
            mission_date = row[1].strip()

        if first == "Mode":
            header_row = i
            break

    if header_row is None:
        msg = "Could not locate command table."
        raise ValueError(msg)

    headers = [c.strip() for c in rows[header_row]]

    commands: list[dict[str, str]] = []

    for row in rows[header_row + 1 :]:
        if not any(str(x).strip() for x in row):
            continue

        row += [""] * (len(headers) - len(row))

        commands.append(dict(zip(headers, row, strict=False)))

    return mission_date, commands


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------


def build_agenda(
    mission_date: str,
    rows: list[dict[str, str]],
) -> list[str]:
    agenda: list[tuple[int, str]] = []

    for row in rows:
        mode = row["Mode"].strip().lower()

        cmd = substitute(row["Telecommand"], mission_date)

        resp = substitute(row["resp_fname"], mission_date)

        repeat = int(row["Repeat"] or 0)
        random_repeat = int(row["Random"] or 0)

        if mode == "single":
            tssent = parse_time(mission_date, row["tssent (UTC)"])
            tsexec = parse_time(mission_date, row["tsexec (UTC)"])

            if tssent is None or tsexec is None:
                raise ValueError(f"Missing timestamp for {cmd}")

            for _ in range(repeat):
                agenda.append(
                    (
                        epoch_ms(tssent),
                        format_command(cmd, tssent, tsexec, resp),
                    )
                )

            if random_repeat:
                window_start = parse_time(
                    mission_date,
                    row["Window Start"],
                )

                window_end = parse_time(
                    mission_date,
                    row["Window End"],
                )

                if window_start is None or window_end is None:
                    raise ValueError(f"Random window missing for {cmd}")

                for dt in random_times(
                    window_start,
                    window_end,
                    random_repeat,
                ):
                    agenda.append(
                        (
                            epoch_ms(dt),
                            format_command(cmd, dt, dt, resp),
                        )
                    )

        elif mode == "interval":
            start = parse_time(
                mission_date,
                row["Window Start"],
            )

            end = parse_time(
                mission_date,
                row["Window End"],
            )

            interval = parse_interval(row["Interval"])

            if start is None or end is None or interval is None:
                raise ValueError(f"Interval missing for {cmd}")

            current = start

            while current <= end:
                agenda.append(
                    (
                        epoch_ms(current),
                        format_command(
                            cmd,
                            current,
                            current,
                            resp,
                        ),
                    )
                )

                current += interval

    agenda.sort(key=lambda x: x[0])

    return [x[1] for x in agenda]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = tyro.cli(Args)

    if args.seed is not None:
        random.seed(args.seed)

    mission_date, rows = load_sheet(args.input_file)

    if not mission_date:
        raise ValueError("Mission date not found.")

    agenda = build_agenda(mission_date, rows)

    args.output_file.write_text(
        "\n".join(agenda),
        encoding="utf-8",
    )

    print(
        f"Generated {len(agenda)} commands → {args.output_file}",
    )


if __name__ == "__main__":
    main()
