"""The main screen GUI, which allows sending commands, and viewing the RX/TX log.

Tabs (left pane):
  1. Telecommand Input – single-command sender; serial port + display options live here
  2. SatNOGS Passes    – fetch / review upcoming observation windows
  3. Command Groups    – loop + priority commands with per-group timing controls
  4. Generate Agenda   – build, preview, and save the command agenda

Right pane: live RX/TX log; after agenda generation, lines are streamed here in the
format `tssent: <UTC> | tsexec: <UTC> | <command>` for easy copy-paste with timing context.
"""

import argparse
import functools
import json
import random
import re
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_split_pane
from dash import ALL, MATCH, callback, dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger
from sortedcontainers import SortedDict

from cts1_telecommand_input.paths import clone_firmware_repo
from cts1_telecommand_input.serial_util import list_serial_ports
from cts1_telecommand_input.telecommand_array_parser import parse_telecommand_list_from_repo
from cts1_telecommand_input.telecommand_preview import generate_telecommand_preview
from cts1_telecommand_input.telecommand_types import TelecommandDefinition
from cts1_telecommand_input.file_util import save_command, parse_datetime_to_timestamp_ms
from cts1_telecommand_input.terminal_app.app_config import MAX_ARGS_PER_TELECOMMAND
from cts1_telecommand_input.terminal_app.app_store import app_store
from cts1_telecommand_input.terminal_app.app_types import (
    UART_PORT_NAME_DISCONNECTED,
    RxTxLogEntry,
)
from cts1_telecommand_input.terminal_app.serial_thread import start_uart_listener

from cts1_mo_tools.cts1_agenda_maker.main import (
    parse_iso,
    dt_to_local_str,
    _lat_lon_to_country,
)
from cts1_mo_tools.cts1_agenda_maker.satnogs_data import iter_future_observation_pages

UART_PORT_OPTION_LABEL_DISCONNECTED = "⛔ Disconnected ⛔"
AGENDA_DIR = Path("agenda")

# ---------------------------------------------------------------------------
# SatNOGS fetch thread state
# ---------------------------------------------------------------------------
_fetch_stop = threading.Event()
_fetch_results: list[dict] = []
_fetch_done = threading.Event()
_fetch_status_msg = ""

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_TSSENT_RE = re.compile(r"@tssent=(\d+)")
_TSEXEC_RE = re.compile(r"@tsexec=(\d+)")

COMMAND_PREFIX = "CTS1+"
COMMAND_SUFFIX = "!"

# ---------------------------------------------------------------------------
# Telecommand repo helpers
# ---------------------------------------------------------------------------


@functools.lru_cache
def get_telecommand_list_from_repo_cached(repo_path: Path | None) -> list[TelecommandDefinition]:
    if repo_path is None:
        return []
    return parse_telecommand_list_from_repo(repo_path)


def get_telecommand_list_from_repo() -> list[TelecommandDefinition]:
    return get_telecommand_list_from_repo_cached(app_store.firmware_repo_path)


def get_telecommand_name_list() -> list[str]:
    return [tcmd.name for tcmd in get_telecommand_list_from_repo()]


def get_telecommand_by_name(name: str) -> TelecommandDefinition:
    tcmds = get_telecommand_list_from_repo()
    tcmd = next((t for t in tcmds if t.name == name), None)
    if not tcmd:
        raise ValueError(f"Telecommand not found: {name}")
    return tcmd


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _now_local_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _parse_dt_flexible(s: str) -> datetime | None:
    """Parse ISO datetime or bare HH:MM (assumes UTC today). Returns None on failure."""
    if not s:
        return None
    s = s.strip()
    try:
        return parse_iso(s)
    except Exception:
        pass
    try:
        now = datetime.now(tz=timezone.utc)
        t = datetime.strptime(s, "%H:%M").replace(
            tzinfo=timezone.utc, year=now.year, month=now.month, day=now.day
        )
        return t
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Agenda file helpers
# ---------------------------------------------------------------------------


def _agenda_filename(uplink_start_str: str, suffix: str = "") -> Path:
    try:
        dt = parse_iso(uplink_start_str)
        compact = dt.strftime("%Y-%m-%dT%H%M")
    except Exception:
        compact = datetime.now().strftime("%Y-%m-%dT%H%M")
    AGENDA_DIR.mkdir(parents=True, exist_ok=True)
    return AGENDA_DIR / f"{compact}_agenda{suffix}.txt"


def _make_clean_lines(raw_lines: list[str]) -> list[str]:
    """Strip comment lines and inline comments — pure command strings only."""
    out = []
    for line in raw_lines:
        if line.lstrip().startswith("#"):
            continue
        line = line.split(" #", 1)[0].rstrip()
        if line:
            out.append(line)
    return out


def _make_timed_display_lines(raw_lines: list[str]) -> list[str]:
    """
    Build right-pane display lines in the format:
        tssent: 2026-06-18 20:41:10 UTC | tsexec: 2026-06-18 20:41:10 UTC | CTS1+cmd()@tssent=...!
    Comment-only lines are dropped; command lines without timestamps are passed through as-is.
    """
    out = []
    for line in raw_lines:
        if line.lstrip().startswith("#"):
            continue
        cmd = line.split(" #", 1)[0].rstrip()
        if not cmd:
            continue
        tssent_m = _TSSENT_RE.search(cmd)
        tsexec_m = _TSEXEC_RE.search(cmd)
        if not tssent_m and not tsexec_m:
            out.append(cmd)
            continue
        parts = []
        if tssent_m:
            ms = int(tssent_m.group(1))
            human = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            parts.append(f"tssent: {human} UTC")
        if tsexec_m:
            ms = int(tsexec_m.group(1))
            human = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            parts.append(f"tsexec: {human} UTC")
        out.append(" | ".join(parts) + " | " + cmd)
    return out


def _write_agenda_file(lines: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pass-range merging helper (SatNOGS summary)
# ---------------------------------------------------------------------------


def _merge_obs_time_ranges(
    observations: list[dict], selected_ids: list
) -> list[tuple[datetime, datetime, list]]:
    """Merge overlapping/touching selected observation windows into combined coverage ranges.

    Returns a sorted list of (merged_start, merged_end, [obs_id, ...]) tuples.
    """
    sel = [o for o in observations if o.get("id") in selected_ids]
    intervals: list[tuple[datetime, datetime, list]] = []
    for obs in sel:
        try:
            s = parse_iso(obs["start"]).astimezone(UTC)
            e = parse_iso(obs["end"]).astimezone(UTC)
        except Exception:
            continue
        intervals.append((s, e, [obs.get("id")]))

    if not intervals:
        return []

    intervals.sort(key=lambda iv: iv[0])
    merged: list[tuple[datetime, datetime, list]] = [intervals[0]]
    for s, e, ids in intervals[1:]:
        last_s, last_e, last_ids = merged[-1]
        if s <= last_e:  # overlapping or touching → merge
            merged[-1] = (last_s, max(last_e, e), last_ids + ids)
        else:
            merged.append((s, e, ids))
    return merged


def _format_merged_window(s: datetime, e: datetime) -> str:
    """e.g. '2026-06-19T16:37:37Z → 16:48:49Z' (shortened end when same UTC date)."""
    s_str = s.strftime("%Y-%m-%dT%H:%M:%SZ")
    e_str = e.strftime("%H:%M:%SZ") if s.date() == e.date() else e.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{s_str} → {e_str}"


# ---------------------------------------------------------------------------
# Agenda group builder
# ---------------------------------------------------------------------------


def _repeat_mode_visibility(repeat_mode: str) -> dict[str, dict]:
    """Return per-field CSS visibility dict for the chosen timing mode."""
    show: dict = {}
    hide: dict = {"display": "none"}
    return {
        "block_interval": show if repeat_mode == "interval" else hide,
        "repeat_count": show if repeat_mode == "count" else hide,
        "duration_window": show if repeat_mode == "duration" else hide,
    }


def _resolve_group_window(
    group: dict,
    pass_start: datetime,
    pass_end: datetime,
) -> tuple[datetime, datetime]:
    start_mode = group.get("start_mode", "offset")
    start_value = str(group.get("start_value", "0")).strip()
    repeat_mode = group.get("repeat_mode", "interval")

    if start_mode == "offset":
        try:
            offset_sec = float(start_value)
        except ValueError:
            offset_sec = 0.0
        win_start = pass_start + timedelta(seconds=offset_sec)
    elif start_mode == "absolute":
        dt = _parse_dt_flexible(start_value)
        win_start = dt if dt else pass_start
    else:  # fixed_tssent — tsexec window starts at AOS
        win_start = pass_start

    if repeat_mode == "duration":
        ws = _parse_dt_flexible(str(group.get("window_start", "")).strip())
        we = _parse_dt_flexible(str(group.get("window_end", "")).strip())
        win_start = ws if ws else win_start
        win_end = we if we else pass_end
    elif repeat_mode == "once":
        win_end = win_start + timedelta(seconds=0.1)
    elif repeat_mode == "count":
        block_interval = float(group.get("block_interval", 20.0))
        repeat_count = int(group.get("repeat_count", 5))
        cmd_count = len([c for c in (group.get("cmds") or "").splitlines() if c.strip()])
        cmd_interval = float(group.get("cmd_interval", 1.0))
        block_dur = max(cmd_count * cmd_interval, 1.0)
        win_end = win_start + timedelta(seconds=repeat_count * block_interval + block_dur)
    else:  # interval
        win_end = pass_end

    win_start = max(win_start, pass_start)
    win_end = min(win_end, pass_end)
    return win_start, win_end


def _build_group_lines(
    group: dict,
    pass_start: datetime,
    pass_end: datetime,
    base_tssent: datetime,
) -> tuple[list[str], datetime]:
    """
    Build command lines for one group within one pass.

    tssent advances after EVERY COMMAND (not per block) — the satellite
    de-duplicates on tssent regardless of command name, so two commands
    in the same block with the same tssent would cause the second to be
    silently dropped.

    Returns (lines, updated_base_tssent).
    """
    raw_cmds = [c.strip() for c in (group.get("cmds") or "").splitlines() if c.strip()]
    if not raw_cmds:
        return [], base_tssent

    resp_fname = (group.get("resp_fname") or "").strip()
    cmds = []
    for c in raw_cmds:
        c = c.removeprefix(COMMAND_PREFIX).removesuffix(COMMAND_SUFFIX)
        if resp_fname and "@resp_fname=" not in c:
            c = f"{c}@resp_fname={resp_fname}"
        cmds.append(c)

    start_mode = group.get("start_mode", "offset")
    repeat_mode = group.get("repeat_mode", "interval")
    block_interval_sec = float(group.get("block_interval", 20.0))
    cmd_interval_sec = float(group.get("cmd_interval", 1.0))
    tssent_spacing_sec = float(group.get("tssent_spacing", 1.0))
    sched_count = int(group.get("sched_count", 5))
    random_count = int(group.get("random_count", 0))
    random_seed_raw = str(group.get("random_seed", "")).strip()

    # Fixed-tssent mode pins all tssent values to a past timestamp
    fixed_tssent_dt: datetime | None = None
    if start_mode == "fixed_tssent":
        try:
            fixed_tssent_dt = parse_iso(str(group.get("start_value", "")).strip())
        except Exception:
            pass

    win_start, win_end = _resolve_group_window(group, pass_start, pass_end)
    if win_end <= win_start:
        return [], base_tssent

    # Build tsexec slots for scheduled (regular) blocks
    if repeat_mode == "once":
        tsexec_slots = [win_start]
    elif repeat_mode == "count":
        n = int(group.get("repeat_count", 5))
        tsexec_slots = [win_start + timedelta(seconds=i * block_interval_sec) for i in range(n)]
    else:  # interval or duration
        tsexec_slots = []
        t = win_start
        while t < win_end:
            tsexec_slots.append(t)
            t += timedelta(seconds=block_interval_sec)

    # Cap scheduled slots
    if sched_count > 0:
        tsexec_slots = tsexec_slots[:sched_count]

    # Random injection slots
    random_slots: list[datetime] = []
    if random_count > 0 and win_end > win_start:
        rng = random.Random(int(random_seed_raw) if random_seed_raw.isdigit() else None)
        total_sec = (win_end - win_start).total_seconds()
        for _ in range(random_count):
            random_slots.append(
                win_start + timedelta(seconds=rng.uniform(0, max(total_sec - 1, 1)))
            )

    all_slots = sorted(tsexec_slots + random_slots)

    lines: list[str] = []
    current_tssent = base_tssent

    for slot_dt in all_slots:
        cmd_tsexec_dt = slot_dt
        for cmd in cmds:
            tssent_dt = fixed_tssent_dt if fixed_tssent_dt else current_tssent
            tssent_ms = int(tssent_dt.timestamp() * 1000)
            tsexec_ms = int(cmd_tsexec_dt.timestamp() * 1000)

            out = f"{COMMAND_PREFIX}{cmd}@tssent={tssent_ms}@tsexec={tsexec_ms}{COMMAND_SUFFIX}"
            tssent_utc = tssent_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            tsexec_utc = cmd_tsexec_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"{out}  # tssent={tssent_utc} tsexec={tsexec_utc}")

            cmd_tsexec_dt += timedelta(seconds=cmd_interval_sec)

            # Advance tssent after EVERY command, not just per block
            if fixed_tssent_dt is None:
                current_tssent += timedelta(seconds=tssent_spacing_sec)

    return lines, current_tssent


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _empty_obs_table() -> html.Div:
    return html.Div(
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("✓", style={"width": "40px"}),
                            html.Th("Obs ID"),
                            html.Th("GS ID"),
                            html.Th("Country"),
                            html.Th("Start (UTC)"),
                            html.Th("End (UTC)"),
                            html.Th("Duration"),
                            html.Th("Start (Local)"),
                            html.Th("Wait (uplink LOS → AOS)"),
                        ]
                    )
                ),
                html.Tbody(
                    id="obs-table-body",
                    children=[
                        html.Tr(
                            html.Td(
                                "No observations loaded.",
                                colSpan=9,
                                className="text-center text-muted",
                            )
                        )
                    ],
                ),
            ],
            bordered=True,
            hover=True,
            striped=True,
            responsive=True,
            style={"fontFamily": "monospace", "fontSize": "0.85rem"},
        ),
        style={"maxHeight": "350px", "overflowY": "auto"},
    )


def _obs_table_rows(
    observations: list[dict],
    selected_ids: list,
    uplink_end_dt: datetime | None,
) -> list:
    if not observations:
        return [
            html.Tr(
                html.Td("No observations found.", colSpan=9, className="text-center text-muted")
            )
        ]
    rows = []
    for obs in sorted(observations, key=lambda o: o.get("start", "")):
        obs_id = obs.get("id", "?")
        gs = obs.get("ground_station", "?")
        start_dt = end_dt = None
        try:
            start_dt = parse_iso(obs["start"])
        except Exception:
            pass
        try:
            end_dt = parse_iso(obs["end"])
        except Exception:
            pass

        start_utc = start_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if start_dt else "?"
        end_utc = end_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if end_dt else "?"
        start_local = dt_to_local_str(start_dt) if start_dt else "?"
        dur_str = "?"
        if start_dt and end_dt:
            dur_sec = int((end_dt - start_dt).total_seconds())
            dur_str = f"{dur_sec // 60}m {dur_sec % 60}s"
        country = "?"
        try:
            country = _lat_lon_to_country(obs["station_lat"], obs["station_lng"]) or "?"
        except Exception:
            pass
        wait_str = "N/A"
        if uplink_end_dt and start_dt:
            delta = start_dt - uplink_end_dt
            wait_str = f"{-delta} ago" if delta.total_seconds() < 0 else str(delta)

        rows.append(
            html.Tr(
                [
                    html.Td(
                        dbc.Checkbox(
                            id={"type": "obs-checkbox", "index": obs_id},
                            value=(obs_id in selected_ids),
                        ),
                        style={"width": "40px"},
                    ),
                    html.Td(str(obs_id)),
                    html.Td(str(gs)),
                    html.Td(country),
                    html.Td(start_utc),
                    html.Td(end_utc),
                    html.Td(dur_str),
                    html.Td(start_local),
                    html.Td(wait_str),
                ]
            )
        )
    return rows


def _obs_summary(observations: list[dict], selected_ids: list) -> html.Div:
    """Show combined (overlap-merged) coverage windows for the selected passes."""
    merged = _merge_obs_time_ranges(observations, selected_ids)
    if not merged:
        return html.Div()
    items = []
    for s, e, ids in merged:
        dur_sec = int((e - s).total_seconds())
        dur_str = f"{dur_sec // 60}m {dur_sec % 60}s"
        window_str = _format_merged_window(s, e)
        ids_str = ", ".join(str(i) for i in ids)
        items.append(
            html.Span(
                f"{window_str}  [{dur_str}]  (Obs: {ids_str})",
                className="badge bg-secondary me-1 mb-1",
                style={"fontFamily": "monospace", "fontSize": "0.8rem"},
            )
        )
    return html.Div(
        [
            html.Small(
                "Combined uplink coverage windows (overlapping passes merged, UTC):",
                className="text-muted d-block mb-1",
            ),
            html.Div(items),
        ]
    )


def _tab_satnogs() -> dbc.Tab:
    return dbc.Tab(
        label="SatNOGS Passes",
        children=[
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Satellite NORAD ID"),
                            dbc.Input(
                                id="sat-id-input",
                                value="69015",
                                placeholder="e.g. 69015",
                                style={"fontFamily": "monospace"},
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Start of Uplink Pass (ISO with timezone)"),
                            dbc.Input(
                                id="uplink-start-input",
                                value=_now_local_iso(),
                                placeholder="2024-05-01T12:00:00-07:00",
                                style={"fontFamily": "monospace"},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Uplink Duration (min)"),
                            dbc.Input(
                                id="uplink-dur-input",
                                type="number",
                                value=15,
                                min=0.1,
                                max=60,
                                style={"fontFamily": "monospace"},
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Fetch next N hours"),
                            dbc.Input(
                                id="next-hours-input",
                                type="number",
                                value=6,
                                min=0.1,
                                max=720,
                                style={"fontFamily": "monospace"},
                            ),
                        ],
                        width=2,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Fetch Observations 🛰️",
                                id="fetch-obs-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Stop ⏹️",
                                id="stop-fetch-btn",
                                color="warning",
                                className="me-2",
                                style={"display": "none"},
                            ),
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(
                            id="obs-fetch-status",
                            className="text-info ms-3",
                            style={"fontFamily": "monospace"},
                        )
                    ),
                ],
                align="center",
                className="mb-3",
            ),
            dcc.Interval(id="fetch-poll-interval", interval=500, n_intervals=0, disabled=True),
            dcc.Store(id="observations-store", data=[]),
            dcc.Store(id="selected-obs-store", data=[]),
            _empty_obs_table(),
            html.Div(id="obs-summary-container", className="mt-2 mb-1"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(id="obs-count-text", className="text-secondary"), width="auto"
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Select All",
                            id="select-all-btn",
                            size="sm",
                            color="secondary",
                            className="me-2",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Deselect All", id="deselect-all-btn", size="sm", color="secondary"
                        ),
                        width="auto",
                    ),
                ],
                align="center",
            ),
            html.Br(),
        ],
    )


def _command_group_card(group_idx: int, gd: dict | None = None) -> dbc.Card:
    gd = gd or {}
    vis = _repeat_mode_visibility(gd.get("repeat_mode", "interval"))

    timing_options = [
        {"label": "Interval (every N sec during window)", "value": "interval"},
        {"label": "Fixed repeat count", "value": "count"},
        {"label": "Once only", "value": "once"},
        {"label": "Duration window (start–end times)", "value": "duration"},
    ]
    start_mode_options = [
        {"label": "Offset after AOS (s)", "value": "offset"},
        {"label": "Absolute UTC time (HH:MM)", "value": "absolute"},
        {"label": "Fixed tssent (past ISO timestamp)", "value": "fixed_tssent"},
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                id={"type": "cg-name", "index": group_idx},
                                value=gd.get("name", f"Group {group_idx + 1}"),
                                placeholder="Group name…",
                                style={"fontFamily": "monospace", "fontWeight": "bold"},
                            ),
                            width=9,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "✕ Remove",
                                id={"type": "cg-remove-btn", "index": group_idx},
                                color="danger",
                                size="sm",
                                outline=True,
                            ),
                            width="auto",
                            className="ms-auto",
                        ),
                    ],
                    align="center",
                )
            ),
            dbc.CardBody(
                [
                    dbc.Label("Commands (one per line — bare name or full CTS1+…! form)"),
                    dbc.Textarea(
                        id={"type": "cg-cmds", "index": group_idx},
                        value=gd.get("cmds", ""),
                        placeholder="core_system_stats()\nget_all_system_thermal_info()",
                        rows=4,
                        style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("@resp_fname (optional)"),
                                    dbc.Input(
                                        id={"type": "cg-resp-fname", "index": group_idx},
                                        value=gd.get("resp_fname", ""),
                                        placeholder="e.g. adcs_data/2026-06-15_control.run",
                                        style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                                    ),
                                    dbc.FormText("Appended to every command in this group."),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    html.Hr(style={"borderColor": "#555"}),
                    # Start-time controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Start time mode"),
                                    dcc.Dropdown(
                                        id={"type": "cg-start-mode", "index": group_idx},
                                        options=start_mode_options,
                                        value=gd.get("start_mode", "offset"),
                                        clearable=False,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Offset (s) / Absolute time (HH:MM UTC) / Fixed tssent (ISO)"
                                    ),
                                    dbc.Input(
                                        id={"type": "cg-start-value", "index": group_idx},
                                        value=str(gd.get("start_value", "0")),
                                        placeholder="0  or  21:31:00  or  2026-01-01T00:00:00Z",
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=8,
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Timing mode + mode-specific fields
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Timing mode (tsexec scheduling)"),
                                    dcc.Dropdown(
                                        id={"type": "cg-repeat-mode", "index": group_idx},
                                        options=timing_options,
                                        value=gd.get("repeat_mode", "interval"),
                                        clearable=False,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Block interval (s)"),
                                    dbc.Input(
                                        id={"type": "cg-block-interval", "index": group_idx},
                                        type="number",
                                        value=gd.get("block_interval", 20.0),
                                        min=1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                id={"type": "cg-block-interval-col", "index": group_idx},
                                md=4,
                                style=vis["block_interval"],
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Repeat count"),
                                    dbc.Input(
                                        id={"type": "cg-repeat-count", "index": group_idx},
                                        type="number",
                                        value=gd.get("repeat_count", 5),
                                        min=1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                id={"type": "cg-repeat-count-col", "index": group_idx},
                                md=4,
                                style=vis["repeat_count"],
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Duration window row (shown only for "duration" mode)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Window start (HH:MM UTC / ISO)"),
                                    dbc.Input(
                                        id={"type": "cg-window-start", "index": group_idx},
                                        value=gd.get("window_start", ""),
                                        placeholder="21:32:30  or  2026-06-15T21:32:30Z",
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Window end (HH:MM UTC / ISO)"),
                                    dbc.Input(
                                        id={"type": "cg-window-end", "index": group_idx},
                                        value=gd.get("window_end", ""),
                                        placeholder="21:42:30",
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        id={"type": "cg-duration-window-row", "index": group_idx},
                        className="mb-2",
                        style=vis["duration_window"],
                    ),
                    # Spacing controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Command spacing within block (s)"),
                                    dbc.Input(
                                        id={"type": "cg-cmd-interval", "index": group_idx},
                                        type="number",
                                        value=gd.get("cmd_interval", 1.0),
                                        min=0.1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                    dbc.FormText(
                                        "tsexec gap between consecutive commands in one block."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("tssent spacing between commands (s)"),
                                    dbc.Input(
                                        id={"type": "cg-tssent-spacing", "index": group_idx},
                                        type="number",
                                        value=gd.get("tssent_spacing", 1.0),
                                        min=0.1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                    dbc.FormText(
                                        "Advances after every command — tssent must be unique per command "
                                        "or the satellite's de-dup logic will silently drop duplicates."
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Hr(style={"borderColor": "#555"}),
                    # Random injection
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Scheduled block count"),
                                    dbc.Input(
                                        id={"type": "cg-sched-count", "index": group_idx},
                                        type="number",
                                        value=gd.get("sched_count", 5),
                                        min=0,
                                        style={"fontFamily": "monospace"},
                                    ),
                                    dbc.FormText(
                                        "Max regularly-spaced blocks (0 = unlimited for interval mode)."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Random injection count"),
                                    dbc.Input(
                                        id={"type": "cg-random-count", "index": group_idx},
                                        type="number",
                                        value=gd.get("random_count", 0),
                                        min=0,
                                        style={"fontFamily": "monospace"},
                                    ),
                                    dbc.FormText(
                                        "Extra blocks at random times within the window."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Random seed (blank = new each run)"),
                                    dbc.Input(
                                        id={"type": "cg-random-seed", "index": group_idx},
                                        value=str(gd.get("random_seed", "")),
                                        placeholder="e.g. 42",
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=4,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-3",
        style={"border": "1px solid #555"},
    )


def _tab_command_groups() -> dbc.Tab:
    default_groups = [
        {
            "name": "Telemetry Loop",
            "cmds": "CTS1+core_system_stats()!\nCTS1+get_all_system_thermal_info()!",
            "resp_fname": "",
            "start_mode": "offset",
            "start_value": "0",
            "cmd_interval": 1.0,
            "tssent_spacing": 1.0,
            "block_interval": 20.0,
            "repeat_mode": "interval",
            "repeat_count": 5,
            "window_start": "",
            "window_end": "",
            "sched_count": 5,
            "random_count": 0,
            "random_seed": "",
        }
    ]
    return dbc.Tab(
        label="Command Groups",
        children=[
            html.Hr(),
            # Priority commands card
            dbc.Card(
                [
                    dbc.CardHeader(html.Strong("⭐ Priority Commands")),
                    dbc.CardBody(
                        [
                            dbc.Label(
                                "Injected with a fixed tssent (past timestamp) so the satellite "
                                "de-duplicates on tssent. @tsexec defaults to 0 (immediate)."
                            ),
                            dbc.Textarea(
                                id="priority-cmds-input",
                                value=(
                                    "CTS1+config_set_int_var(TCMD_require_unique_tssent,1)!\n"
                                    "CTS1+obc_set_stm32_sysclk_to_hse()!"
                                ),
                                rows=3,
                                style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Fixed tssent (past ISO timestamp)"),
                                            dbc.Input(
                                                id="priority-fixed-tssent",
                                                value="2026-01-01T00:00:00Z",
                                                placeholder="2026-01-01T00:00:00Z",
                                                style={"fontFamily": "monospace"},
                                            ),
                                            dbc.FormText(
                                                "Use a past date — satellite de-dups on this."
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("@tsexec (ms, 0 = immediate)"),
                                            dbc.Input(
                                                id="priority-tsexec-input",
                                                value="0",
                                                placeholder="0",
                                                style={"fontFamily": "monospace"},
                                            ),
                                            dbc.FormText("Applied to the upfront priority batch."),
                                        ],
                                        md=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Inject every N loop commands"),
                                            dbc.Input(
                                                id="priority-interval-input",
                                                type="number",
                                                value=50,
                                                min=1,
                                                style={"fontFamily": "monospace"},
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Scheduled count (per pass)"),
                                            dbc.Input(
                                                id="priority-sched-count",
                                                type="number",
                                                value=5,
                                                min=0,
                                                style={"fontFamily": "monospace"},
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Random count (per pass)"),
                                            dbc.Input(
                                                id="priority-random-count",
                                                type="number",
                                                value=10,
                                                min=0,
                                                style={"fontFamily": "monospace"},
                                            ),
                                        ],
                                        md=2,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4",
                color="warning",
                outline=True,
            ),
            html.H5("Loop Command Groups", className="mb-1"),
            html.P(
                "Each group runs its commands as a block. Timing controls set when "
                "and how often the block repeats within each satellite pass.",
                className="text-muted mb-3",
            ),
            html.Div(
                id="command-groups-container",
                children=[_command_group_card(i, g) for i, g in enumerate(default_groups)],
            ),
            dbc.Button(
                "＋ Add Command Group",
                id="add-group-btn",
                color="success",
                outline=True,
                className="mb-3",
            ),
            dcc.Store(id="command-groups-store", data=default_groups),
        ],
    )


def _tab_telecommand_input(*, selected_command_name: str, enable_advanced: bool) -> dbc.Tab:
    return dbc.Tab(
        label="Telecommand Input",
        children=_generate_left_pane_send_commands(
            selected_command_name=selected_command_name,
            enable_advanced=enable_advanced,
        ),
    )


def _tab_generate() -> dbc.Tab:
    return dbc.Tab(
        label="Generate Agenda",
        children=[
            html.Hr(),
            html.P(
                "Uses the passes selected in 'SatNOGS Passes' and the groups defined "
                "in 'Command Groups' to produce a time-stamped command agenda.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "⚡ Generate Command Agenda",
                            id="generate-agenda-btn",
                            color="primary",
                            size="lg",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(
                            id="generate-status",
                            className="text-success ms-3",
                            style={"fontFamily": "monospace"},
                        )
                    ),
                ],
                align="center",
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "💾 Save Agenda (with comments)",
                            id="save-agenda-btn",
                            color="secondary",
                            disabled=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "🧹 Save Clean Agenda (commands only)",
                            id="save-clean-agenda-btn",
                            color="secondary",
                            outline=True,
                            disabled=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(
                            id="save-agenda-status",
                            className="text-info ms-2",
                            style={"fontFamily": "monospace"},
                        )
                    ),
                ],
                align="center",
                className="mb-3",
            ),
            html.Hr(),
            html.H6("Preview (with comments):", className="text-muted"),
            dbc.Textarea(
                id="agenda-preview",
                value="(generate agenda to see preview)",
                readOnly=True,
                rows=30,
                style={
                    "fontFamily": "monospace",
                    "fontSize": "0.78rem",
                    "backgroundColor": "#0d1117",
                    "color": "#c9d1d9",
                },
            ),
            dcc.Store(id="agenda-lines-store", data=[]),
            dcc.Store(id="agenda-uplink-start-store", data=""),
        ],
    )


def generate_left_pane(*, selected_command_name: str, enable_advanced: bool) -> list:
    return [
        html.H1("CTS-SAT-1 Telecommand Input Terminal", className="text-center"),
        dbc.Tabs(
            id="left-pane-tabs",
            children=[
                _tab_telecommand_input(
                    selected_command_name=selected_command_name,
                    enable_advanced=enable_advanced,
                ),
                _tab_satnogs(),
                _tab_command_groups(),
                _tab_generate(),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Callbacks – Serial port
# ---------------------------------------------------------------------------


def handle_uart_port_change(uart_port_name: str) -> None:
    last = app_store.uart_port_name
    if uart_port_name != last:
        if uart_port_name == UART_PORT_NAME_DISCONNECTED:
            msg = "Serial port disconnected."
        elif last == UART_PORT_NAME_DISCONNECTED:
            msg = f"Serial port connected: {uart_port_name}"
        else:
            msg = f"Serial port changed from {last} to {uart_port_name}"
        logger.info(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "notice"))
    app_store.uart_port_name = uart_port_name


@callback(
    Output("uart-port-dropdown", "options"),
    Input("uart-port-dropdown", "value"),
    Input("uart-port-dropdown-interval-component", "n_intervals"),
)
def update_uart_port_dropdown_options(uart_port_name, _n_intervals):
    if uart_port_name is None:
        uart_port_name = UART_PORT_NAME_DISCONNECTED
    handle_uart_port_change(uart_port_name)
    port_name_list = list_serial_ports()
    if app_store.uart_port_name not in ([*port_name_list, UART_PORT_NAME_DISCONNECTED]):
        msg = f"Serial port no longer available: {app_store.uart_port_name}"
        logger.warning(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "error"))
        app_store.uart_port_name = UART_PORT_NAME_DISCONNECTED
    return [
        {"label": UART_PORT_OPTION_LABEL_DISCONNECTED, "value": UART_PORT_NAME_DISCONNECTED}
    ] + [{"label": p, "value": p} for p in port_name_list]


# ---------------------------------------------------------------------------
# Callbacks – SatNOGS Passes
# ---------------------------------------------------------------------------


@callback(
    Output("observations-store", "data"),
    Output("obs-fetch-status", "children"),
    Output("fetch-poll-interval", "disabled"),
    Output("fetch-obs-btn", "disabled"),
    Output("stop-fetch-btn", "style"),
    Input("fetch-obs-btn", "n_clicks"),
    State("sat-id-input", "value"),
    State("uplink-start-input", "value"),
    State("uplink-dur-input", "value"),
    State("next-hours-input", "value"),
    prevent_initial_call=True,
)
def start_fetch_observations(n_clicks, sat_id, uplink_start_str, uplink_dur, next_hours):
    global _fetch_results, _fetch_status_msg
    if not sat_id:
        return (
            dash.no_update,
            "⚠️ Enter a satellite NORAD ID first.",
            True,
            False,
            {"display": "none"},
        )
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
    except Exception:
        return dash.no_update, "⚠️ Invalid uplink start datetime.", True, False, {"display": "none"}

    uplink_end_dt = uplink_start_dt + timedelta(minutes=float(uplink_dur or 15))
    start_lt_filter = uplink_start_dt + timedelta(hours=float(next_hours or 6))

    _fetch_results = []
    _fetch_stop.clear()
    _fetch_done.clear()
    _fetch_status_msg = "Fetching…"

    def _run():
        global _fetch_status_msg
        try:
            for page in iter_future_observation_pages(
                sat_id,
                start_gt_filter=uplink_end_dt,
                start_lt_filter=start_lt_filter,
            ):
                _fetch_results.extend(page)
                _fetch_status_msg = f"Fetching… {len(_fetch_results)} so far"
                if _fetch_stop.is_set():
                    _fetch_status_msg = f"⏹️ Stopped. {len(_fetch_results)} loaded."
                    break
            else:
                _fetch_status_msg = f"✅ Loaded {len(_fetch_results)} observations."
        except Exception as exc:
            _fetch_status_msg = f"❌ Error: {exc}"
        finally:
            _fetch_done.set()

    threading.Thread(target=_run, daemon=True).start()
    return [], "Fetching…", False, True, {"display": "inline-block"}


@callback(
    Output("observations-store", "data", allow_duplicate=True),
    Output("obs-fetch-status", "children", allow_duplicate=True),
    Output("fetch-poll-interval", "disabled", allow_duplicate=True),
    Output("fetch-obs-btn", "disabled", allow_duplicate=True),
    Output("stop-fetch-btn", "style", allow_duplicate=True),
    Input("fetch-poll-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_fetch_progress(_n):
    done = _fetch_done.is_set()
    return (
        list(_fetch_results),
        _fetch_status_msg,
        done,
        not done,
        {"display": "none"} if done else {"display": "inline-block"},
    )


@callback(Input("stop-fetch-btn", "n_clicks"), prevent_initial_call=True)
def stop_fetch(_n_clicks):
    _fetch_stop.set()


@callback(
    Output("obs-table-body", "children"),
    Output("selected-obs-store", "data"),
    Output("obs-count-text", "children"),
    Output("obs-summary-container", "children"),
    Input("observations-store", "data"),
    Input("select-all-btn", "n_clicks"),
    Input("deselect-all-btn", "n_clicks"),
    State("uplink-start-input", "value"),
    State("uplink-dur-input", "value"),
    State("selected-obs-store", "data"),
    prevent_initial_call=False,
)
def update_obs_table(
    observations, _sel_all, _desel_all, uplink_start_str, uplink_dur, current_selected
):
    from dash import ctx

    triggered = ctx.triggered_id if ctx.triggered_id else ""
    obs_list = observations or []

    if triggered == "select-all-btn":
        selected_ids = [o.get("id") for o in obs_list]
    elif triggered == "deselect-all-btn":
        selected_ids = []
    else:
        existing = set(current_selected or [])
        selected_ids = list(existing | {o.get("id") for o in obs_list})

    uplink_end_dt = None
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
        uplink_end_dt = uplink_start_dt + timedelta(minutes=float(uplink_dur or 15))
    except Exception:
        pass

    rows = _obs_table_rows(obs_list, selected_ids, uplink_end_dt)
    valid_sel = [i for i in selected_ids if i in {o.get("id") for o in obs_list}]
    count_text = f"{len(obs_list)} fetched, {len(valid_sel)} selected"
    summary = _obs_summary(obs_list, selected_ids)
    return rows, selected_ids, count_text, summary


@callback(
    Output("selected-obs-store", "data", allow_duplicate=True),
    Output("obs-count-text", "children", allow_duplicate=True),
    Output("obs-summary-container", "children", allow_duplicate=True),
    Input({"type": "obs-checkbox", "index": ALL}, "value"),
    State({"type": "obs-checkbox", "index": ALL}, "id"),
    State("observations-store", "data"),
    prevent_initial_call=True,
)
def obs_checkbox_changed(values, ids, observations):
    selected_ids = [id_dict["index"] for id_dict, val in zip(ids, values) if val]
    total = len(observations or [])
    count_text = f"{total} fetched, {len(selected_ids)} selected"
    summary = _obs_summary(observations or [], selected_ids)
    return selected_ids, count_text, summary


# ---------------------------------------------------------------------------
# Callbacks – Command Groups
# ---------------------------------------------------------------------------

_CG_STATE_KEYS = [
    "cg-name",
    "cg-cmds",
    "cg-resp-fname",
    "cg-start-mode",
    "cg-start-value",
    "cg-repeat-mode",
    "cg-block-interval",
    "cg-repeat-count",
    "cg-window-start",
    "cg-window-end",
    "cg-cmd-interval",
    "cg-tssent-spacing",
    "cg-sched-count",
    "cg-random-count",
    "cg-random-seed",
]
_CG_DEFAULTS = {
    "cg-name": "Group",
    "cg-cmds": "",
    "cg-resp-fname": "",
    "cg-start-mode": "offset",
    "cg-start-value": "0",
    "cg-repeat-mode": "interval",
    "cg-block-interval": 20.0,
    "cg-repeat-count": 5,
    "cg-window-start": "",
    "cg-window-end": "",
    "cg-cmd-interval": 1.0,
    "cg-tssent-spacing": 1.0,
    "cg-sched-count": 5,
    "cg-random-count": 0,
    "cg-random-seed": "",
}
_UI_TO_STORE = {k: k.replace("cg-", "").replace("-", "_") for k in _CG_STATE_KEYS}


@callback(
    Output("command-groups-store", "data"),
    Output("command-groups-container", "children"),
    Input("add-group-btn", "n_clicks"),
    Input({"type": "cg-remove-btn", "index": ALL}, "n_clicks"),
    State("command-groups-store", "data"),
    *[State({"type": k, "index": ALL}, "value") for k in _CG_STATE_KEYS],
    prevent_initial_call=True,
)
def manage_command_groups(add_clicks, remove_clicks, stored_groups, *all_field_values):
    from dash import ctx

    n = len(stored_groups)
    field_lists = list(all_field_values)
    current_groups = []
    for i in range(n):
        g = {}
        for fi, key in enumerate(_CG_STATE_KEYS):
            store_key = _UI_TO_STORE[key]
            vals = field_lists[fi]
            g[store_key] = vals[i] if i < len(vals) else _CG_DEFAULTS[key]
        current_groups.append(g)

    triggered = ctx.triggered_id
    if triggered == "add-group-btn":
        new_g = {_UI_TO_STORE[k]: _CG_DEFAULTS[k] for k in _CG_STATE_KEYS}
        new_g["name"] = f"Group {len(current_groups) + 1}"
        current_groups.append(new_g)
    elif isinstance(triggered, dict) and triggered.get("type") == "cg-remove-btn":
        idx = triggered["index"]
        if 0 <= idx < len(current_groups):
            current_groups.pop(idx)

    cards = [_command_group_card(i, g) for i, g in enumerate(current_groups)]
    return current_groups, cards


@callback(
    Output({"type": "cg-block-interval-col", "index": MATCH}, "style"),
    Output({"type": "cg-repeat-count-col", "index": MATCH}, "style"),
    Output({"type": "cg-duration-window-row", "index": MATCH}, "style"),
    Input({"type": "cg-repeat-mode", "index": MATCH}, "value"),
)
def toggle_group_repeat_mode_fields(repeat_mode):
    """Show only the field(s) relevant to the selected timing mode."""
    vis = _repeat_mode_visibility(repeat_mode)
    return vis["block_interval"], vis["repeat_count"], vis["duration_window"]


# ---------------------------------------------------------------------------
# Callbacks – Telecommand Input
# ---------------------------------------------------------------------------


@callback(
    Output("argument-inputs-container", "children"),
    Input("telecommand-dropdown", "value"),
)
def update_argument_inputs(selected_command_name: str):
    selected_tcmd = get_telecommand_by_name(selected_command_name)
    arg_inputs = []
    for arg_num in range(MAX_ARGS_PER_TELECOMMAND):
        if selected_tcmd.argument_descriptions and arg_num < len(
            selected_tcmd.argument_descriptions
        ):
            label = f"Arg {arg_num}: {selected_tcmd.argument_descriptions[arg_num]}"
        else:
            label = f"Arg {arg_num}"
        this_id = f"arg-input-{arg_num}"
        arg_inputs.append(
            dbc.FormFloating(
                [
                    dbc.Input(
                        type="text",
                        id=this_id,
                        placeholder=label,
                        disabled=(arg_num >= selected_tcmd.number_of_args),
                        style={"fontFamily": "monospace"},
                    ),
                    dbc.Label(label, html_for=this_id),
                ],
                className="mb-3",
                style=({"display": "none"} if arg_num >= selected_tcmd.number_of_args else {}),
            )
        )
    return arg_inputs


@callback(
    Output("stored-command-preview", "data"),
    Input("telecommand-dropdown", "value"),
    Input("suffix-tags-checklist", "value"),
    Input("input-tsexec-suffix-tag", "value"),
    Input("input-tssent-datetime", "value"),
    Input("input-resp_fname-suffix-tag", "value"),
    Input("extra-suffix-tags-input", "value"),
    Input("uart-update-interval-component", "n_intervals"),
    *[Input(f"arg-input-{i}", "value") for i in range(MAX_ARGS_PER_TELECOMMAND)],
    prevent_initial_call=True,
)
def update_stored_command_preview(
    selected_command_name,
    suffix_tags_checklist,
    tsexec_suffix_tag,
    tssent_datetime_input,
    resp_fname_suffix_tag,
    extra_suffix_tags_input,
    _n_intervals,
    *every_arg_value,
):
    if suffix_tags_checklist is None:
        suffix_tags_checklist = []
    if tsexec_suffix_tag == "":
        tsexec_suffix_tag = None
    if resp_fname_suffix_tag == "":
        resp_fname_suffix_tag = None

    tssent_timestamp_ms = tsexec_timestamp_ms = None
    if tssent_datetime_input:
        tssent_timestamp_ms = parse_datetime_to_timestamp_ms(tssent_datetime_input)
        tsexec_timestamp_ms = parse_datetime_to_timestamp_ms(tssent_datetime_input)

    selected_command = get_telecommand_by_name(selected_command_name)
    arg_vals = [
        str(every_arg_value[i]) if every_arg_value[i] is not None else ""
        for i in range(selected_command.number_of_args)
    ]

    enable_tssent_suffix = "enable_tssent_tag" in suffix_tags_checklist
    if tssent_timestamp_ms is not None:
        enable_tssent_suffix = False

    extra_suffix_tags: dict = {}
    if extra_suffix_tags_input:
        try:
            parsed = json.loads(extra_suffix_tags_input)
            if isinstance(parsed, dict):
                extra_suffix_tags.update(parsed)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")

    if tssent_timestamp_ms is not None:
        extra_suffix_tags["tssent"] = str(tssent_timestamp_ms)
    if tsexec_timestamp_ms is not None:
        extra_suffix_tags["tsexec"] = str(tsexec_timestamp_ms)

    return generate_telecommand_preview(
        tcmd_name=selected_command_name,
        arg_list=arg_vals,
        enable_tssent_suffix=enable_tssent_suffix,
        tsexec_suffix_value=tsexec_suffix_tag,
        resp_fname_suffix_value=resp_fname_suffix_tag,
        extra_suffix_tags=extra_suffix_tags.copy(),
    )


@callback(
    Output("command-preview-container", "children"),
    Input("stored-command-preview", "data"),
)
def update_command_preview_render(command_preview: str):
    return [
        html.H4("Command Preview", className="text-center"),
        html.Pre(command_preview, id="command-preview", className="mb-3"),
    ]


def send_command_to_device(command_text: str) -> None:
    app_store.last_tx_timestamp_sec = time.time()
    app_store.tx_queue.append(command_text.encode("ascii"))


@callback(
    Input("send-button", "n_clicks"),
    State("telecommand-dropdown", "value"),
    State("stored-command-preview", "data"),
    *[State(f"arg-input-{i}", "value") for i in range(MAX_ARGS_PER_TELECOMMAND)],
    prevent_initial_call=True,
)
def send_button_callback(n_clicks, selected_command_name, command_preview, *every_arg_value):
    if selected_command_name is None:
        app_store.append_to_rxtx_log(RxTxLogEntry(b"No command selected.", "error"))
        return
    args = [
        every_arg_value[i]
        for i in range(get_telecommand_by_name(selected_command_name).number_of_args)
    ]
    if any(a is None or a == "" for a in args):
        app_store.append_to_rxtx_log(
            RxTxLogEntry(
                f"Not all arguments filled in for {selected_command_name}.".encode(), "error"
            )
        )
        return
    if app_store.uart_port_name == UART_PORT_NAME_DISCONNECTED:
        app_store.append_to_rxtx_log(
            RxTxLogEntry(b"Can't send command when disconnected.", "error")
        )
        return
    send_command_to_device(command_preview)


@callback(Input("clear-log-button", "n_clicks"), prevent_initial_call=True)
def clear_log_button_callback(n_clicks: int):
    max_idx = app_store.rxtx_log.keys()[-1]
    app_store.rxtx_log = SortedDict({max_idx + 1: RxTxLogEntry(b"Log Reset", "notice")}).copy()


@callback(
    Output("stored-rxtx-log-pause-limits", "data"),
    Output("pause-button", "children"),
    Output("pause-button", "color"),
    Input("pause-button", "n_clicks"),
)
def pause_button_callback(n_clicks: int):
    if n_clicks % 2 == 0:
        return {"paused": False, "pause_min_idx": None, "pause_max_idx": None}, "Pause ⏸️", "danger"
    pause_min_idx = app_store.rxtx_log.keys()[0]
    pause_max_idx = app_store.rxtx_log.keys()[-1]
    return (
        {"paused": True, "pause_min_idx": pause_min_idx, "pause_max_idx": pause_max_idx},
        "Resume ▶️",
        "success",
    )


@callback(
    Output("selected-tcmd-info-container", "children"),
    Input("telecommand-dropdown", "value"),
)
def update_selected_tcmd_info(selected_command_name: str):
    selected_command = get_telecommand_by_name(selected_command_name)
    docstring = selected_command.full_docstring or f"No docstring for {selected_command.tcmd_func}"
    table_fields = selected_command.to_dict_table_fields()
    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Field"), html.Th("Value")])),
            html.Tbody(
                [
                    html.Tr([html.Td(k), html.Td(v, style={"fontFamily": "monospace"})])
                    for k, v in table_fields.items()
                ]
            ),
        ],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
    )
    return [
        html.H4("Command Info", className="text-center"),
        table,
        html.Hr(),
        html.H4("Command Docstring", className="text-center"),
        html.Pre(docstring, id="selected-tcmd-info", className="mb-3"),
    ]


@callback(
    Output("suffix-tags-checklist", "value"),
    Input("input-tssent-datetime", "value"),
    State("suffix-tags-checklist", "value"),
)
def disable_checkbox_when_datetime_present(dt_value, checklist_values):
    if dt_value:
        return [v for v in (checklist_values or []) if v != "enable_tssent_tag"]
    return checklist_values


@callback(
    Input("save-button", "n_clicks"),
    State("stored-command-preview", "data"),
    State("filename-input", "value"),
    prevent_initial_call=True,
)
def save_button_callback(n_clicks, command_preview, filename):
    if not command_preview:
        app_store.append_to_rxtx_log(RxTxLogEntry(b"No command to save.", "error"))
        return
    try:
        filepath = save_command(command_preview, filename)
        app_store.append_to_rxtx_log(RxTxLogEntry(f"Saved → {filepath.name}".encode(), "input"))
        app_store.append_to_rxtx_log(
            RxTxLogEntry(f"Telecommand → {command_preview}".encode(), "input")
        )
    except OSError as e:
        app_store.append_to_rxtx_log(RxTxLogEntry(str(e).encode(), "error"))


# ---------------------------------------------------------------------------
# Callbacks – Generate Agenda
# ---------------------------------------------------------------------------


@callback(
    Output("agenda-preview", "value"),
    Output("generate-status", "children"),
    Output("agenda-lines-store", "data"),
    Output("agenda-uplink-start-store", "data"),
    Output("save-agenda-btn", "disabled"),
    Output("save-clean-agenda-btn", "disabled"),
    Input("generate-agenda-btn", "n_clicks"),
    State("observations-store", "data"),
    State("selected-obs-store", "data"),
    State("uplink-start-input", "value"),
    State("uplink-dur-input", "value"),
    State("next-hours-input", "value"),
    State("sat-id-input", "value"),
    State("priority-cmds-input", "value"),
    State("priority-fixed-tssent", "value"),
    State("priority-tsexec-input", "value"),
    State("priority-interval-input", "value"),
    State("priority-sched-count", "value"),
    State("priority-random-count", "value"),
    State("command-groups-store", "data"),
    prevent_initial_call=True,
)
def generate_agenda(
    n_clicks,
    observations,
    selected_ids,
    uplink_start_str,
    uplink_dur,
    next_hours,
    sat_id,
    priority_cmds_raw,
    priority_fixed_tssent_str,
    priority_tsexec_input,
    priority_interval,
    priority_sched_count,
    priority_random_count,
    command_groups,
):
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
    except Exception:
        return dash.no_update, "❌ Invalid uplink start datetime.", [], "", True, True

    selected_obs = [o for o in (observations or []) if o.get("id") in (selected_ids or [])]
    if not selected_obs:
        return dash.no_update, "❌ No observations selected.", [], "", True, True

    selected_obs_sorted = sorted(selected_obs, key=lambda o: o.get("start", ""))
    priority_cmds = [c.strip() for c in (priority_cmds_raw or "").splitlines() if c.strip()]

    priority_fixed_tssent_dt: datetime | None = None
    try:
        priority_fixed_tssent_dt = parse_iso((priority_fixed_tssent_str or "").strip())
    except Exception:
        pass

    try:
        priority_tsexec_ms = int(str(priority_tsexec_input or "0").strip())
    except ValueError:
        priority_tsexec_ms = 0

    all_output_lines: list[str] = [
        "# CTS-SAT-1 Command Agenda",
        f"# Generated: {datetime.now(tz=UTC).isoformat()}",
        f"# Satellite NORAD ID: {sat_id}",
        f"# Uplink start: {uplink_start_str}",
        f"# Uplink duration: {uplink_dur} min",
        "",
    ]

    total_cmd_count = 0
    group_errors: list[str] = []
    tssent_cursor = uplink_start_dt

    # ── Priority commands — upfront batch
    if priority_cmds and priority_fixed_tssent_dt:
        all_output_lines.append("# ── Priority Commands (upfront)")
        p_tssent_ms = int(priority_fixed_tssent_dt.timestamp() * 1000)
        tssent_utc = priority_fixed_tssent_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        tsexec_display = (
            "immediate"
            if priority_tsexec_ms == 0
            else datetime.fromtimestamp(priority_tsexec_ms / 1000, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        for pcmd in priority_cmds:
            pcmd_clean = pcmd.removeprefix(COMMAND_PREFIX).removesuffix(COMMAND_SUFFIX)
            out = (
                f"{COMMAND_PREFIX}{pcmd_clean}"
                f"@tssent={p_tssent_ms}@tsexec={priority_tsexec_ms}{COMMAND_SUFFIX}"
            )
            all_output_lines.append(f"{out}  # tssent={tssent_utc} tsexec={tsexec_display}")
            total_cmd_count += 1
        all_output_lines.append("")

    # ── Per-pass loop
    for obs in selected_obs_sorted:
        try:
            pass_start = parse_iso(obs["start"])
            pass_end = parse_iso(obs["end"])
        except Exception:
            continue

        obs_id = obs.get("id", "?")
        gs = obs.get("ground_station", "?")
        dur_sec = int((pass_end - pass_start).total_seconds())
        pass_start_utc = pass_start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        pass_end_utc = pass_end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        all_output_lines.append(
            f"# ══ Obs {obs_id} | GS {gs} | {pass_start_utc} → {pass_end_utc} "
            f"({dur_sec // 60}m {dur_sec % 60}s)"
        )

        # Priority injections within pass
        if priority_cmds and priority_fixed_tssent_dt:
            p_sched = int(priority_sched_count or 0)
            p_rand = int(priority_random_count or 0)
            p_tssent_ms = int(priority_fixed_tssent_dt.timestamp() * 1000)
            p_slots: list[datetime] = []
            if p_sched > 0:
                interval_sec = dur_sec / max(p_sched, 1)
                p_slots += [
                    pass_start + timedelta(seconds=i * interval_sec) for i in range(p_sched)
                ]
            if p_rand > 0:
                rng = random.Random()
                for _ in range(p_rand):
                    p_slots.append(pass_start + timedelta(seconds=rng.uniform(0, dur_sec)))
            p_slots.sort()
            if p_slots:
                all_output_lines.append("# ── Priority injections")
                for slot in p_slots:
                    tsexec_ms = int(slot.timestamp() * 1000)
                    tsexec_utc = slot.strftime("%Y-%m-%dT%H:%M:%SZ")
                    tssent_utc_inner = priority_fixed_tssent_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    for pcmd in priority_cmds:
                        pcmd_clean = pcmd.removeprefix(COMMAND_PREFIX).removesuffix(COMMAND_SUFFIX)
                        out = (
                            f"{COMMAND_PREFIX}{pcmd_clean}"
                            f"@tssent={p_tssent_ms}@tsexec={tsexec_ms}{COMMAND_SUFFIX}"
                        )
                        all_output_lines.append(
                            f"{out}  # tssent={tssent_utc_inner} tsexec={tsexec_utc}"
                        )
                        total_cmd_count += 1

        # Loop command groups
        for group in command_groups or []:
            group_name = group.get("name", "Group")
            try:
                g_lines, tssent_cursor = _build_group_lines(
                    group, pass_start, pass_end, tssent_cursor
                )
            except Exception as exc:
                group_errors.append(f"Obs {obs_id} / {group_name}: {exc}")
                logger.exception(f"Error building group lines: {exc}")
                continue
            if g_lines:
                all_output_lines.append(f"# ── {group_name}")
                all_output_lines.extend(g_lines)
                total_cmd_count += len(g_lines)

        all_output_lines.append("")

    all_output_lines.append(f"# Total commands: {total_cmd_count}")
    if group_errors:
        all_output_lines.append("# WARNINGS:")
        for err in group_errors:
            all_output_lines.append(f"#   {err}")

    status = f"✅ {total_cmd_count} commands, {len(selected_obs_sorted)} pass(es)."
    if group_errors:
        status += f" ⚠️ {len(group_errors)} error(s)."

    # Push timed display lines to right-pane RX/TX log
    timed_lines = _make_timed_display_lines(all_output_lines)
    app_store.append_to_rxtx_log(
        RxTxLogEntry(b"=== Generated Agenda (tssent/tsexec + command) ===", "notice")
    )
    for tl in timed_lines:
        app_store.append_to_rxtx_log(RxTxLogEntry(tl.encode("ascii", errors="replace"), "input"))

    return (
        "\n".join(all_output_lines),
        status,
        all_output_lines,
        uplink_start_str or "",
        False,
        False,
    )


@callback(
    Output("save-agenda-status", "children"),
    Input("save-agenda-btn", "n_clicks"),
    State("agenda-lines-store", "data"),
    State("agenda-uplink-start-store", "data"),
    prevent_initial_call=True,
)
def save_agenda_with_comments(n_clicks, lines, uplink_start_str):
    if not lines:
        return "❌ No agenda generated yet."
    try:
        path = _agenda_filename(uplink_start_str)
        _write_agenda_file(lines, path)
        return f"💾 Saved: {path.resolve()}"
    except Exception as exc:
        return f"❌ Save failed: {exc}"


@callback(
    Output("save-agenda-status", "children", allow_duplicate=True),
    Input("save-clean-agenda-btn", "n_clicks"),
    State("agenda-lines-store", "data"),
    State("agenda-uplink-start-store", "data"),
    prevent_initial_call=True,
)
def save_clean_agenda(n_clicks, lines, uplink_start_str):
    if not lines:
        return "❌ No agenda generated yet."
    try:
        clean_lines = _make_clean_lines(lines)
        path = _agenda_filename(uplink_start_str, suffix="_clean")
        _write_agenda_file(clean_lines, path)
        return f"🧹 Saved clean: {path.resolve()}"
    except Exception as exc:
        return f"❌ Save failed: {exc}"


# ---------------------------------------------------------------------------
# Callbacks – RX/TX log
# ---------------------------------------------------------------------------


def generate_rx_tx_log(
    *,
    show_end_of_line_chars: bool = False,
    show_timestamp: bool = False,
    auto_format_json: bool = False,
    pause_min_idx: int | None = None,
    pause_max_idx: int | None = None,
) -> html.Div:
    if pause_min_idx is None:
        pause_min_idx = app_store.rxtx_log.keys()[0]
    if pause_max_idx is None:
        pause_max_idx = app_store.rxtx_log.keys()[-1]
    return html.Div(
        [
            html.Pre(
                entry.to_string(
                    show_end_of_line_chars=show_end_of_line_chars,
                    show_timestamp=show_timestamp,
                    auto_format_json=auto_format_json,
                ),
                style=(entry.css_style | {"margin": "0", "lineHeight": "1.1"}),
            )
            for idx, entry in app_store.rxtx_log.items()
            if (idx >= pause_min_idx) and (idx <= pause_max_idx)
        ],
        id="rx-tx-log",
        className="p-3",
        style={"display": "block", "width": "fit-content"},
    )


@callback(
    Output("rx-tx-log-container", "children"),
    Output("uart-update-interval-component", "interval"),
    Input("uart-port-dropdown", "value"),
    Input("send-button", "n_clicks"),
    Input("clear-log-button", "n_clicks"),
    Input("uart-update-interval-component", "n_intervals"),
    Input("display-options-checklist", "value"),
    Input("stored-rxtx-log-pause-limits", "data"),
)
def update_uart_log_interval(
    _uart_port_name,
    _n_clicks_send,
    _n_clicks_clear,
    _update_count,
    display_options_checklist,
    stored_rxtx_log_pause_limits,
):
    sec_since_send = time.time() - app_store.last_tx_timestamp_sec
    if sec_since_send < 10:
        app_store.uart_log_refresh_rate_ms = 250
    elif sec_since_send < 60:
        app_store.uart_log_refresh_rate_ms = 800
    else:
        app_store.uart_log_refresh_rate_ms = 2000

    opts = display_options_checklist or []
    return (
        generate_rx_tx_log(
            show_end_of_line_chars="show_end_of_line_chars" in opts,
            show_timestamp="show_timestamp" in opts,
            auto_format_json="auto_format_json" in opts,
            pause_min_idx=stored_rxtx_log_pause_limits.get("pause_min_idx"),
            pause_max_idx=stored_rxtx_log_pause_limits.get("pause_max_idx"),
        ),
        app_store.uart_log_refresh_rate_ms,
    )


# ---------------------------------------------------------------------------
# Left-pane Telecommand Input layout helper
# ---------------------------------------------------------------------------


def _generate_left_pane_send_commands(
    *, selected_command_name: str, enable_advanced: bool
) -> list:
    return [
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Serial Port:", html_for="uart-port-dropdown"),
                        dcc.Dropdown(
                            id="uart-port-dropdown",
                            options=(
                                [
                                    {
                                        "label": UART_PORT_OPTION_LABEL_DISCONNECTED,
                                        "value": UART_PORT_NAME_DISCONNECTED,
                                    }
                                ]
                                + [{"label": p, "value": p} for p in list_serial_ports()]
                            ),
                            value=UART_PORT_NAME_DISCONNECTED,
                            className="mb-2",
                        ),
                        dcc.Interval(
                            id="uart-port-dropdown-interval-component",
                            interval=2500,
                            n_intervals=0,
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dbc.Label("Display Options:"),
                        dbc.Checklist(
                            options={
                                "show_end_of_line_chars": "Show EOL chars",
                                "show_timestamp": "Timestamps",
                                "auto_format_json": "Auto-format JSON",
                            },
                            id="display-options-checklist",
                            value=["auto_format_json"],
                            inline=True,
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-2",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Label("Select a Telecommand:", html_for="telecommand-dropdown"),
                dcc.Dropdown(
                    id="telecommand-dropdown",
                    options=[{"label": cmd, "value": cmd} for cmd in get_telecommand_name_list()],
                    value=selected_command_name,
                    className="mb-3",
                    style={"fontFamily": "monospace"},
                ),
            ]
        ),
        html.Div(
            update_argument_inputs(selected_command_name),
            id="argument-inputs-container",
            className="mb-3",
        ),
        html.Hr(),
        dbc.Label("Suffix Tag Options:"),
        dbc.Checklist(
            options={"enable_tssent_tag": "Send '@tssent=current_timestamp' Tag?"},
            id="suffix-tags-checklist",
        ),
        dbc.FormFloating(
            [
                dbc.Input(
                    type="text",
                    id="input-tsexec-suffix-tag",
                    placeholder="Timestamp to Execute Command (@tsexec=xxx)",
                    style={"fontFamily": "monospace"},
                ),
                dbc.Label(
                    "Timestamp to Execute Command (@tsexec=xxx)",
                    html_for="input-tsexec-suffix-tag",
                ),
            ],
            className="mb-3",
        ),
        dbc.FormFloating(
            [
                dbc.Input(
                    type="text",
                    id="input-tssent-datetime",
                    placeholder="YYYY-MM-DD HH:MM MST or UTC",
                    style={"fontFamily": "monospace"},
                ),
                dbc.Label(
                    "Schedule execution (e.g. 2026-04-25 16:13 UTC) — sets both @tssent and @tsexec",
                    html_for="input-tssent-datetime",
                ),
            ],
            className="mb-3",
        ),
        dbc.FormFloating(
            [
                dbc.Input(
                    type="text",
                    id="input-resp_fname-suffix-tag",
                    placeholder="File Name to log the response",
                    style={"fontFamily": "monospace"},
                ),
                dbc.Label(
                    "File Name to store TCMD response", html_for="input-resp_fname-suffix-tag"
                ),
            ],
            className="mb-3",
        ),
        dbc.FormFloating(
            [
                dbc.Input(
                    type="text",
                    id="extra-suffix-tags-input",
                    placeholder="Extra Suffix Tags Input (JSON)",
                    style={"fontFamily": "monospace"},
                ),
                dbc.Label("Extra Suffix Tags Input (JSON)", html_for="extra-suffix-tags-input"),
            ],
            className="mb-3",
            style=({} if enable_advanced else {"display": "none"}),
        ),
        html.Hr(),
        dbc.Label("Save Telecommands to a File:"),
        dbc.FormFloating(
            [
                dbc.Input(
                    id="filename-input",
                    type="text",
                    placeholder="20260422T1322",
                    style={"fontFamily": "monospace"},
                ),
                dbc.Label(
                    "Filename to save command (e.g. 20260422T1322)", html_for="filename-input"
                ),
            ],
            className="mb-3",
        ),
        html.Hr(),
        html.Div(id="command-preview-container", className="mb-3"),
        dbc.Row(
            [
                dbc.Button(
                    "Clear Log 🫗",
                    id="clear-log-button",
                    n_clicks=0,
                    className="m-1 px-3",
                    style={"width": "auto"},
                    color="warning",
                ),
                dbc.Button(
                    "Pause ⏯️",
                    id="pause-button",
                    n_clicks=0,
                    className="m-1 px-3",
                    style={"width": "auto"},
                ),
                dbc.Button(
                    "Send 📡",
                    id="send-button",
                    n_clicks=0,
                    className="m-1 px-5",
                    style={"width": "auto"},
                ),
                dbc.Button(
                    "Save File 💾",
                    id="save-button",
                    n_clicks=0,
                    className="m-1 px-3",
                    style={"width": "auto"},
                    color="secondary",
                ),
            ],
            justify="center",
            className="mb-3",
        ),
        html.Hr(),
        html.Div(id="selected-tcmd-info-container", className="mb-3"),
    ]


# ---------------------------------------------------------------------------
# App runner
# ---------------------------------------------------------------------------


def run_dash_app(*, enable_debug: bool = False, enable_advanced: bool = False) -> None:
    app_name = "CTS-SAT-1 Telecommand Input"
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=app_name,
        update_title=("Updating..." if enable_debug else ""),
    )

    app.layout = dbc.Container(
        [
            dash_split_pane.DashSplitPane(
                [
                    html.Div(
                        generate_left_pane(
                            selected_command_name=get_telecommand_name_list()[0],
                            enable_advanced=enable_advanced,
                        ),
                        className="p-3",
                        style={"height": "100%", "overflowY": "auto"},
                    ),
                    html.Div(
                        generate_rx_tx_log(),
                        id="rx-tx-log-container",
                        style={
                            "fontFamily": "monospace",
                            "backgroundColor": "black",
                            "height": "100%",
                            "overflowY": "auto",
                            "overflowX": "auto",
                            "flexDirection": "column-reverse",
                            "display": "flex",
                            "position": "absolute",
                        },
                    ),
                ],
                id="vertical-split-pane-1",
                split="vertical",
                size=580,
                minSize=380,
                pane2Style={"backgroundColor": "black", "overflowX": "auto"},
            ),
            dbc.Button(
                "Jump to Bottom ⬇️",
                id="scroll-to-bottom-button",
                style={
                    "display": "none",
                    "position": "fixed",
                    "bottom": "20px",
                    "right": "60px",
                    "zIndex": "99",
                },
                color="danger",
            ),
            dcc.Interval(id="uart-update-interval-component", interval=800, n_intervals=0),
            dcc.Store(id="stored-command-preview", data=""),
            dcc.Store(id="stored-rxtx-log-pause-limits", data={"paused": False}.copy()),
        ],
        fluid=True,
    )

    start_uart_listener()
    app.run_server(debug=enable_debug)
    logger.info("Dash app started and finished.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo", "--firmware-repo", dest="firmware_repo", type=str)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-a", "--advanced", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        if args.firmware_repo is None:
            firmware_repo_path, repo = clone_firmware_repo(Path(tmp_dir))
            logger.info(f"Cloned firmware repo (commit={repo.head.commit.hexsha[:7]}): {tmp_dir}")
        else:
            firmware_repo_path = Path(args.firmware_repo)
            if not firmware_repo_path.is_dir():
                raise FileNotFoundError(f"Repo not found: {args.firmware_repo}")
            logger.info(f"Using provided firmware repo: {args.firmware_repo}")

        app_store.firmware_repo_path = firmware_repo_path
        logger.info(f"Loaded {len(get_telecommand_name_list())} telecommands.")
        run_dash_app(enable_debug=args.debug, enable_advanced=args.advanced)


if __name__ == "__main__":
    main()
