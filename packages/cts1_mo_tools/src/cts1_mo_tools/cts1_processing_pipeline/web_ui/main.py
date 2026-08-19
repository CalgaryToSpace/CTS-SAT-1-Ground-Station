"""CTS-SAT-1 Processing Pipeline Web UI.

A NiceGUI dashboard over `everything_decoded.parquet` (step 3's output):
recent beacon(s) at a glance, plus line/scatter charts for every field in
the basic and extended beacon packets, grouped by subsystem.

Usage (uv):
    uv run cts1_data_web_ui
    uv run cts1_data_web_ui --parquet-path output/everything_decoded.parquet
    uv run cts1_data_web_ui --hours 6
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

# tyro resolves dataclass field annotations at runtime (via get_type_hints),
# so imports used only in annotations below still need to be real imports.

__all__ = ["main"]

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 -- tyro needs this at runtime, see below
from typing import TYPE_CHECKING

import tyro
from nicegui import ui

from . import data as beacon_data
from .charts import BEACON_CHART_GROUPS, OTHER_CHART_GROUPS, chart_option
from .file_reassembly import (
    COVERAGE_ROW_WIDTH_BYTES,
    BulkHeaderCandidate,
    ReassemblyResult,
    find_header_candidates,
    reassemble_bulk_chunks,
    render_coverage_png,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from nicegui import events

    from .charts import ChartSpec

REFRESH_INTERVAL_SEC = 30.0

# label -> hours; "All time" (None) skips the `received_at` filter entirely.
WINDOW_CHOICES: dict[str, float | None] = {
    "Last 1h": 1.0,
    "Last 6h": 6.0,
    "Last 24h": 24.0,
    "Last 7d": 24.0 * 7,
    "All time": None,
}


def _age_str(received_at: datetime) -> str:
    delta = datetime.now(UTC) - received_at.replace(tzinfo=UTC)
    total_sec = int(delta.total_seconds())
    if total_sec < 60:  # noqa: PLR2004
        return f"{total_sec}s ago"
    if total_sec < 3600:  # noqa: PLR2004
        return f"{total_sec // 60}m ago"
    return f"{total_sec // 3600}h {(total_sec % 3600) // 60}m ago"


def _latest_beacon_card(path: Path) -> None:
    latest = beacon_data.latest_beacons(path, n=1)
    with ui.card().classes("w-full"):
        if latest.is_empty():
            ui.label("No beacon packets decoded yet.").classes("text-lg")
            return

        row = latest.to_dicts()[0]
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(f"Latest beacon -- {row['packet_type']}").classes(
                    "text-xl font-bold"
                )
                received_at = row["received_at"]
                ui.label(
                    f"{received_at:%Y-%m-%d %H:%M:%S} UTC ({_age_str(received_at)})"
                ).classes("text-caption text-grey")
            ui.label(row.get("friendly_message") or "").classes("text-sm italic")

        stats = [
            ("Satellite", row.get("satellite_name")),
            ("Uptime", f"{row.get('uptime_sec', '?')} s"),
            ("Battery", f"{row.get('eps_battery_percent', '?')}%"),
            ("Battery Voltage", f"{row.get('eps_battery_voltage_V', '?')} V"),
            ("OBC Temp", f"{row.get('obc_temperature_C', '?')} °C"),
            ("EPS Mode", row.get("eps_mode")),
            ("CTS1 State", row.get("cts1_operation_state")),
            ("RSSI", f"{row.get('rssi_db', '?')} dB"),
        ]
        with ui.row().classes("w-full gap-8 flex-wrap mt-2"):
            for label, value in stats:
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-caption text-grey text-uppercase")
                    ui.label(str(value)).classes("text-base font-medium")


def _recent_beacons_table(path: Path) -> None:
    recent = beacon_data.latest_beacons(path, n=10)
    with ui.card().classes("w-full"):
        ui.label("Recent Beacons").classes("text-lg font-bold")
        if recent.is_empty():
            ui.label("Nothing to show yet.")
            return

        columns = [
            {"name": "received_at", "label": "Received", "field": "received_at"},
            {"name": "packet_type", "label": "Type", "field": "packet_type"},
            {"name": "uptime_sec", "label": "Uptime (s)", "field": "uptime_sec"},
            {
                "name": "eps_battery_percent",
                "label": "Battery %",
                "field": "eps_battery_percent",
            },
            {
                "name": "obc_temperature_C",
                "label": "OBC Temp (°C)",
                "field": "obc_temperature_C",
            },
            {"name": "rssi_db", "label": "RSSI (dB)", "field": "rssi_db"},
        ]
        rows = [
            {
                "received_at": f"{r['received_at']:%Y-%m-%d %H:%M:%S}",
                "packet_type": r["packet_type"],
                "uptime_sec": r.get("uptime_sec"),
                "eps_battery_percent": r.get("eps_battery_percent"),
                "obc_temperature_C": r.get("obc_temperature_C"),
                "rssi_db": r.get("rssi_db"),
            }
            for r in recent.to_dicts()
        ]
        ui.table(columns=columns, rows=rows, row_key="received_at").classes("w-full")


def _render_chart_group(title: str, specs: list[ChartSpec], df: pl.DataFrame) -> None:
    """One collapsible chart group, with charts mounted lazily on first expand.

    An `ui.echart` is a real ECharts instance in the browser (its own canvas,
    its own render loop) -- mounting all ~47 of them up front, even behind a
    collapsed (but still-present) panel, is what makes the page laggy. Only
    building a group's charts the first time its panel is actually opened
    keeps the initial page (and every later re-render) down to whatever the
    user currently has expanded.
    """
    options = [
        option for spec in specs if (option := chart_option(spec, df)) is not None
    ]
    if not options:
        return

    built = False

    def _on_toggle(e: events.ValueChangeEventArguments) -> None:
        nonlocal built
        if not e.value or built:
            return
        built = True
        with (
            container,
            ui.grid(columns="repeat(auto-fit, minmax(420px, 1fr))").classes(
                "w-full gap-4 p-2"
            ),
        ):
            for option in options:
                ui.echart(option).classes("h-72")

    with ui.expansion(title, value=False, on_value_change=_on_toggle).classes(
        "w-full border rounded"
    ):
        container = ui.column().classes("w-full")


def _chart_groups(path: Path, *, since: datetime | None) -> None:
    beacons = beacon_data.load_beacon_window(path, since=since)
    all_packets = beacon_data.load_packet_window(path, since=since)

    for title, specs in BEACON_CHART_GROUPS:
        _render_chart_group(title, specs, beacons)

    for title, specs in OTHER_CHART_GROUPS:
        _render_chart_group(title, specs, all_packets)


@dataclass(frozen=True, slots=True)
class Args:
    """CTS-SAT-1 processing pipeline web UI."""

    parquet_path: Path = beacon_data.DEFAULT_PARQUET_PATH
    """Path to `everything_decoded.parquet` (step 3's output)."""

    hours: float = 24.0
    """Default chart time window, in hours. Adjustable in the UI afterward."""

    port: int = 8089
    """Port to serve the dashboard on."""


def _window_label_for_hours(hours: float | None) -> str:
    for label, value in WINDOW_CHOICES.items():
        if value == hours:
            return label
    return "Last 24h"


def _page_shell() -> ui.column:
    """The column every page's content sits in: centered, capped width."""
    return ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4")


def _build_home_page(args: Args) -> None:
    state: dict[str, float | None] = {"hours": args.hours}

    # Split in two so the 30s auto-refresh only ever touches the cheap,
    # DOM-only status widgets -- the chart section (the expensive part,
    # once panels are open) is left alone unless the user explicitly asks
    # for it via the window selector or the "Refresh charts" button. Tearing
    # down and remounting ~47 ECharts instances every 30s regardless of
    # what's open was the main source of lag.
    @ui.refreshable
    def live_status() -> None:
        latest = beacon_data.latest_beacons(args.parquet_path, n=1)
        if latest.is_empty():
            ui.label(
                f"No beacon packets found at {args.parquet_path}. Run the "
                "pipeline (step_1, step_2, step_3) first."
            ).classes("text-lg text-warning")
            return
        _latest_beacon_card(args.parquet_path)
        _recent_beacons_table(args.parquet_path)

    @ui.refreshable
    def chart_section() -> None:
        since = (
            datetime.now(UTC) - timedelta(hours=state["hours"])
            if state["hours"] is not None
            else None
        )
        _chart_groups(args.parquet_path, since=since)

    def _on_window_change(label: str) -> None:
        state["hours"] = WINDOW_CHOICES[label]
        chart_section.refresh()

    with _page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("CTS-SAT-1 Ground Station").classes("text-2xl font-bold")
            with ui.row().classes("items-center gap-4"):
                ui.select(
                    list(WINDOW_CHOICES),
                    value=_window_label_for_hours(state["hours"]),
                    label="Chart window",
                    on_change=lambda e: _on_window_change(e.value),
                ).classes("w-40")
                ui.button(
                    "Refresh charts", icon="refresh", on_click=chart_section.refresh
                )
        live_status()
        chart_section()

    ui.timer(REFRESH_INTERVAL_SEC, live_status.refresh)


@dataclass
class _TimeRangeRow:
    """One (mutable) start/end pair the user is editing in the range list."""

    start: str | None = None
    end: str | None = None


RANGE_INPUT_MASK = "####-##-## ##:##:##"
RANGE_INPUT_PLACEHOLDER = "YYYY-MM-DD HH:MM:SS"


def _parse_range_input(value: str | None) -> datetime | None:
    """Parse a `RANGE_INPUT_MASK`-shaped value ("YYYY-MM-DD HH:MM:SS"),
    treated as UTC (like every other timestamp in this dashboard).

    A plain masked text input rather than `<input type=datetime-local>` is
    deliberate: that native picker's on-screen *display* follows the
    browser/OS locale (day-first vs. month-first, 12h vs. 24h), even though
    its underlying value is always ISO 8601 -- ambiguous exactly where a
    ground-station operator can least afford it. The mask forces
    YYYY-MM-DD/24h ordering no matter whose browser this runs in.

    Returns None for anything that doesn't (yet) parse -- an in-progress or
    invalid edit is just treated the same as an empty row rather than
    raising, since this runs on every keystroke.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _valid_ranges(rows: list[_TimeRangeRow]) -> list[tuple[datetime, datetime]]:
    """Every row with both a start and an end, start <= end. Rows the user
    hasn't finished filling in (or got backwards) are silently skipped
    rather than erroring -- they just don't contribute to the selection yet.
    """
    ranges = []
    for row in rows:
        start = _parse_range_input(row.start)
        end = _parse_range_input(row.end)
        if start is not None and end is not None and start <= end:
            ranges.append((start, end))
    return ranges


def _byte_range_str(start: int, end: int) -> str:
    return f"{start:,}-{end:,} ({end - start:,} bytes)"


def _duplicates_status(result: ReassemblyResult) -> None:
    """The pass/fail line above the chunks table -- the table itself always
    renders (see `_build_chunks_table`) regardless of which of these
    applies, so it's still there to inspect even mid-download, before every gap is
    filled in.
    """
    if not result.duplicates:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("Every offset appears exactly once.").classes("text-positive")
    elif result.has_conflicts:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label(
                f"{len(result.conflicts)} offset(s) have CONFLICTING "
                "content -- narrow the time range(s) until these agree "
                "before trusting the reassembled bytes below (the other "
                f"{len(result.duplicates) - len(result.conflicts)} "
                "duplicated offset(s) agree and aren't a problem)."
            ).classes("text-negative")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.icon("info", color="grey")
            ui.label(
                f"{len(result.duplicates)} offset(s) were received more "
                "than once (e.g. a retransmission), but every copy agrees "
                "-- not a problem."
            ).classes("text-caption text-grey")


CHUNKS_TABLE_PAGE_SIZE = 100


def _build_chunks_table(result: ReassemblyResult) -> Callable[[], None]:
    """Build a paginated, independently-refreshable chunks table.

    A file's chunks (`result.offsets`) are already fully computed
    server-side by this point -- reassembling the bytes needs every one of
    them regardless -- but a 20+ MB file is 100,000+ 195-byte chunks, and
    shipping that many rows to the browser at once (even paginated
    client-side by Quasar, which still means the whole row set travels over
    the websocket first) would be its own problem. Paging *here*, before
    anything is handed to `ui.table`, keeps each page turn down to
    `CHUNKS_TABLE_PAGE_SIZE` rows -- flipping pages only re-renders this one
    component (via the returned callable's `.refresh()`), not the whole
    results section, so it never re-fetches or re-reassembles anything.
    """
    page_state = {"index": 0}

    @ui.refreshable
    def chunks_table() -> None:
        total = len(result.offsets)
        if total == 0:
            return
        total_pages = -(-total // CHUNKS_TABLE_PAGE_SIZE)  # ceil div
        page_state["index"] = max(0, min(page_state["index"], total_pages - 1))
        start = page_state["index"] * CHUNKS_TABLE_PAGE_SIZE
        page_offsets = result.offsets[start : start + CHUNKS_TABLE_PAGE_SIZE]

        columns = [
            {"name": "offset", "label": "Offset", "field": "offset"},
            {"name": "length", "label": "Length", "field": "length"},
            {"name": "count", "label": "Copies", "field": "count"},
            {
                "name": "distinct_contents_count",
                "label": "Distinct Contents Count",
                "field": "distinct_contents_count",
            },
            {"name": "consistent", "label": "Agree?", "field": "consistent"},
        ]
        rows = [
            {
                "offset": o.offset,
                "length": ", ".join(str(n) for n in o.lengths),
                "count": o.count,
                "distinct_contents_count": o.distinct_contents_count,
                "consistent": "yes" if o.consistent else "CONFLICTING BYTES",
            }
            for o in page_offsets
        ]
        ui.table(columns=columns, rows=rows, row_key="offset").classes("w-full")

        if total_pages == 1:
            ui.label(f"{total:,} distinct offset(s).").classes("text-caption text-grey")
            return

        def _turn_page(delta: int) -> None:
            page_state["index"] += delta
            chunks_table.refresh()

        with ui.row().classes("w-full items-center justify-between mt-2"):
            ui.label(
                f"Showing {start + 1:,}-{min(start + CHUNKS_TABLE_PAGE_SIZE, total):,} "
                f"of {total:,} distinct offset(s)."
            ).classes("text-caption text-grey")
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="chevron_left", on_click=lambda: _turn_page(-1)).props(
                    "flat dense round"
                ).set_enabled(page_state["index"] > 0)
                ui.label(f"Page {page_state['index'] + 1} / {total_pages}")
                ui.button(icon="chevron_right", on_click=lambda: _turn_page(1)).props(
                    "flat dense round"
                ).set_enabled(page_state["index"] < total_pages - 1)

    return chunks_table


COVERAGE_BLOCK_PX = 3  # on-screen size of one byte's block; may change later


def _coverage_map(result: ReassemblyResult) -> None:
    """A byte-coverage map: one block per byte, green if covered, red if
    still a gap, `COVERAGE_ROW_WIDTH_BYTES` blocks per row.

    `render_coverage_png` encodes one *pixel* per byte -- small and cheap to
    ship even for a 20+ MB file -- and the `COVERAGE_BLOCK_PX`-per-byte
    on-screen size is applied here purely with CSS
    (`image-rendering: pixelated` keeps the block edges crisp instead of
    blurring the upscale).
    """
    if result.span_bytes == 0:
        return
    data_uri = "data:image/png;base64," + base64.b64encode(
        render_coverage_png(result)
    ).decode("ascii")
    width_px = COVERAGE_ROW_WIDTH_BYTES * COVERAGE_BLOCK_PX
    height_px = -(-result.span_bytes // COVERAGE_ROW_WIDTH_BYTES) * COVERAGE_BLOCK_PX
    with ui.row().classes("w-full overflow-x-auto"):
        ui.image(data_uri).style(
            f"width: {width_px}px; height: {height_px}px; image-rendering: pixelated;"
        )


def _gaps_section(result: ReassemblyResult) -> None:
    if result.is_gapless:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("No gaps -- every byte in range is covered.").classes(
                "text-positive"
            )
        return

    with ui.row().classes("items-center gap-2"):
        ui.icon("warning", color="warning")
        ui.label(
            f"{len(result.gaps)} missing byte range(s) -- these need to be "
            "re-downlinked:"
        ).classes("text-warning")
    for start, end in result.gaps[:50]:
        ui.label(_byte_range_str(start, end)).classes("font-mono text-sm")
    if len(result.gaps) > 50:  # noqa: PLR2004
        ui.label(f"...and {len(result.gaps) - 50} more.").classes(
            "text-caption text-grey"
        )


def _sha256_comparison(result: ReassemblyResult, expected: str | None) -> None:
    ui.label(f"SHA-256 of assembled bytes: {result.sha256}").classes(
        "font-mono text-sm"
    )
    if not expected:
        ui.label("No SHA-256 to compare against -- pick a header below.").classes(
            "text-caption text-grey"
        )
    elif len(expected) < 64:  # noqa: PLR2004
        ui.label(
            f"Header's SHA-256 was truncated ({len(expected)}/64 hex chars) -- "
            "can't verify."
        ).classes("text-caption text-grey")
    elif expected == result.sha256:
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle", color="positive")
            ui.label("Matches the header's SHA-256.").classes("text-positive")
    else:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label("Does NOT match the header's SHA-256.").classes("text-negative")


def _best_named_candidate(
    candidates: list[BulkHeaderCandidate],
) -> BulkHeaderCandidate | None:
    """Whichever candidate looks most trustworthy: prefer one with a full
    (untruncated, 64-hex-char) SHA-256, then one with a known file_size,
    tie-broken by most recently received.
    """
    named = [c for c in candidates if c.file]
    if not named:
        return None

    def _score(c: BulkHeaderCandidate) -> tuple[int, int, datetime]:
        has_full_sha256 = c.sha256 is not None and len(c.sha256) == 64  # noqa: PLR2004
        return (int(has_full_sha256), int(c.file_size is not None), c.received_at)

    return max(named, key=_score)


def _header_candidates_table(candidates: list[BulkHeaderCandidate]) -> str | None:
    """Render the found header candidates and return the SHA-256 (possibly
    None/truncated) of whichever one looks most trustworthy -- see
    `_best_named_candidate`.
    """
    if not candidates:
        ui.label(
            "No TCMD_RESPONSE header found in the selected range(s) -- that's "
            "fine, headers are sometimes missing entirely; just cross-check "
            "the file name/size/hash some other way."
        ).classes("text-caption text-grey")
        return None

    columns = [
        {"name": "received_at", "label": "Received", "field": "received_at"},
        {"name": "action", "label": "Action", "field": "action"},
        {"name": "file", "label": "File", "field": "file"},
        {"name": "file_size", "label": "Size (bytes)", "field": "file_size"},
        {"name": "crc16", "label": "CRC-16", "field": "crc16"},
        {
            "name": "sha256",
            "label": "SHA-256",
            "field": "sha256",
            # The full value, not manually truncated -- the browser ellipses
            # it if the column is too narrow to fit (see Quasar's `.ellipsis`
            # utility class), rather than us guessing how much to cut.
            "classes": "ellipsis",
            "headerClasses": "ellipsis",
            "style": "max-width: 220px",
        },
    ]
    rows = [
        {
            "received_at": f"{c.received_at:%Y-%m-%d %H:%M:%S}",
            "action": c.action or "",
            "file": c.file or "",
            "file_size": f"{c.file_size:,}" if c.file_size is not None else "",
            "crc16": c.crc16 or "",
            "sha256": c.sha256 or "",
        }
        for c in candidates
    ]
    ui.table(columns=columns, rows=rows, row_key="received_at").classes("w-full")

    best = _best_named_candidate(candidates)
    return best.sha256 if best is not None else None


def _reassembler_results(path: Path, ranges: list[tuple[datetime, datetime]]) -> None:
    if not ranges:
        ui.label(
            "No time range selected yet -- fill in a start and end above "
            "(an empty selection intentionally shows nothing, rather than "
            "every BULK_FILE_DOWNLINK packet ever decoded)."
        ).classes("text-caption text-grey")
        return

    chunks = beacon_data.load_bulk_file_downlink_packets(path, ranges=ranges)
    tcmd_responses = beacon_data.load_tcmd_response_packets(path, ranges=ranges)
    candidates = find_header_candidates(tcmd_responses)

    with ui.card().classes("w-full"):
        ui.label("Header candidates").classes("text-lg font-bold")
        expected_sha256 = _header_candidates_table(candidates)

    with ui.card().classes("w-full"):
        ui.label("BULK_FILE_DOWNLINK chunks").classes("text-lg font-bold")
        if chunks.is_empty():
            ui.label("No BULK_FILE_DOWNLINK packets in the selected range(s).").classes(
                "text-caption text-grey"
            )
            return

        result = reassemble_bulk_chunks(chunks)
        ui.label(
            f"{result.total_chunks:,} packet(s), {result.unique_offsets:,} "
            f"distinct offset(s), spanning {result.span_bytes:,} bytes."
        ).classes("text-caption text-grey")

        _duplicates_status(result)
        _coverage_map(result)
        _build_chunks_table(result)()
        _gaps_section(result)
        _sha256_comparison(result, expected_sha256)

        best = _best_named_candidate(candidates)
        # The full path (e.g. "ADCS/log_b51a.TLM"), not just the basename --
        # slashes become underscores since the browser would otherwise treat
        # them as directory separators in the downloaded filename.
        filename = (
            best.file.replace("/", "_") if best is not None and best.file else None
        )
        ui.button(
            "Download reassembled bytes",
            icon="download",
            on_click=lambda: ui.download.content(
                result.data, filename=filename or "reassembled_file.bin"
            ),
        )
        if result.has_conflicts or not result.is_gapless:
            ui.label(
                "Conflicting offsets and/or gaps are present -- the download "
                "will still work (missing bytes are filled with 0x00), but "
                "isn't a verified, complete copy of the file yet."
            ).classes("text-caption text-grey")


def _build_file_reassembler_page(args: Args) -> None:
    rows: list[_TimeRangeRow] = [_TimeRangeRow()]

    @ui.refreshable
    def range_editor() -> None:
        for i, row in enumerate(rows):
            with ui.row().classes("w-full items-center gap-2"):

                def _set_start(e: events.ValueChangeEventArguments, row=row) -> None:  # noqa: ANN001
                    row.start = e.value

                def _set_end(e: events.ValueChangeEventArguments, row=row) -> None:  # noqa: ANN001
                    row.end = e.value

                ui.input(
                    "Start (UTC)",
                    value=row.start or "",
                    placeholder=RANGE_INPUT_PLACEHOLDER,
                    on_change=_set_start,
                ).props(f'mask="{RANGE_INPUT_MASK}"').classes("w-56")
                ui.input(
                    "End (UTC)",
                    value=row.end or "",
                    placeholder=RANGE_INPUT_PLACEHOLDER,
                    on_change=_set_end,
                ).props(f'mask="{RANGE_INPUT_MASK}"').classes("w-56")
                if len(rows) > 1:

                    def _remove(i: int = i) -> None:
                        rows.pop(i)
                        range_editor.refresh()

                    ui.button(icon="close", on_click=_remove).props("flat round dense")
        ui.button(
            "Add time range",
            icon="add",
            on_click=lambda: (rows.append(_TimeRangeRow()), range_editor.refresh()),
        ).props("flat")

    @ui.refreshable
    def results() -> None:
        _reassembler_results(args.parquet_path, _valid_ranges(rows))

    with _page_shell():
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("File Reassembler").classes("text-2xl font-bold")
            ui.button("Analyze", icon="search", on_click=results.refresh)
        ui.label(
            "Pick one or more UTC time ranges covering a single bulk file "
            "download. Missing time boundaries on a row mean that row is "
            "ignored, so an empty first row (the default) selects nothing "
            "until you fill one in."
        ).classes("text-caption text-grey")
        with ui.card().classes("w-full"):
            range_editor()
        results()


def _build_pages(args: Args) -> None:
    def _drawer() -> None:
        with ui.left_drawer(value=True):
            ui.link("Home", "/").classes("text-lg p-2")
            ui.link("File Reassembler", "/file-reassembler").classes("text-lg p-2")

    @ui.page("/")
    def home_page() -> None:
        _drawer()
        _build_home_page(args)

    @ui.page("/file-reassembler")
    def file_reassembler_page() -> None:
        _drawer()
        _build_file_reassembler_page(args)


def main() -> None:
    """Entry point: parse CLI args, build pages, and start the NiceGUI server."""
    args = tyro.cli(Args)
    _build_pages(args)
    ui.run(title="CTS-SAT-1 Ground Station", dark=True, port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
