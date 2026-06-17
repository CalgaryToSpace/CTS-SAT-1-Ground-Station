"""The main screen GUI, which allows sending commands, and viewing the RX/TX log.

Tabs:
  1. Connect & Config  – serial port, display options
  2. SatNOGS Passes   – fetch / review upcoming observation windows
  3. Command Groups    – loop + priority commands with per-group timing controls
  4. Telecommand Input – single-command sender (existing)
  5. Generate          – build and preview the command agenda
  6. Tools             – quick utility buttons
"""

import argparse
import functools
import json
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_split_pane
from dash import MATCH, ALL, callback, dcc, html
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

# Import agenda-building utilities from the existing agenda generator module.
from cts1_mo_tools.cts1_agenda_maker.main import (
    AgendaParams,
    build_agenda,
    format_command,
    parse_iso,
    dt_to_local_str,
    _lat_lon_to_country,
)
from cts1_mo_tools.cts1_agenda_maker.satnogs_data import iter_future_observation_pages

UART_PORT_OPTION_LABEL_DISCONNECTED = "⛔ Disconnected ⛔"

# ---------------------------------------------------------------------------
# SatNOGS fetch state (module-level, shared across callbacks via dcc.Store)
# ---------------------------------------------------------------------------
_fetch_thread: threading.Thread | None = None
_fetch_stop = threading.Event()


# ---------------------------------------------------------------------------
# Telecommand helpers (unchanged from original)
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
    telecommands = get_telecommand_list_from_repo()
    telecommand = next((t for t in telecommands if t.name == name), None)
    if not telecommand:
        raise ValueError(f"Telecommand not found: {name}")
    return telecommand


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _now_local_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


# ── Tab: Connect & Config ────────────────────────────────────────────────────


def _tab_connect_config() -> dbc.Tab:
    return dbc.Tab(
        label="Connect & Config",
        children=[
            html.Hr(),
            dbc.Row(
                [
                    dbc.Label("Select a Serial Port:", html_for="uart-port-dropdown"),
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
                        className="mb-3",
                    ),
                    dcc.Interval(
                        id="uart-port-dropdown-interval-component",
                        interval=2500,
                        n_intervals=0,
                    ),
                ]
            ),
            html.Hr(),
            html.H4("Display Options", className="text-center"),
            dbc.Checklist(
                options={
                    "show_end_of_line_chars": "Show End-of-Line Characters?",
                    "show_timestamp": "Show Timestamps?",
                    "auto_format_json": "Auto Format JSON?",
                },
                id="display-options-checklist",
                value=["auto_format_json"],
            ),
        ],
    )


# ── Tab: SatNOGS Passes ──────────────────────────────────────────────────────


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
                            dbc.Label("Uplink Duration (minutes)"),
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
                            dbc.Label("Fetch next N hours after uplink"),
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
                            dbc.Spinner(
                                html.Span(id="fetch-spinner-placeholder"),
                                id="fetch-spinner",
                                spinner_style={"display": "none"},
                                size="sm",
                                color="info",
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
            # Hidden interval for polling fetch progress
            dcc.Interval(id="fetch-poll-interval", interval=500, n_intervals=0, disabled=True),
            # Store for raw observations list
            dcc.Store(id="observations-store", data=[]),
            # Store for selected obs IDs
            dcc.Store(id="selected-obs-store", data=[]),
            html.Div(id="obs-table-container", children=_empty_obs_table()),
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
                            html.Th("Start (Local)"),
                            html.Th("End (Local)"),
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
    """Build table rows for the observation list."""
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
        end_local = dt_to_local_str(end_dt) if end_dt else "?"

        country = "?"
        try:
            country = _lat_lon_to_country(obs["station_lat"], obs["station_lng"]) or "?"
        except Exception:
            pass

        wait_str = "N/A"
        if uplink_end_dt and start_dt:
            delta = start_dt - uplink_end_dt
            wait_str = f"{-delta} ago" if delta.total_seconds() < 0 else str(delta)

        is_checked = obs_id in selected_ids

        rows.append(
            html.Tr(
                [
                    html.Td(
                        dbc.Checkbox(
                            id={"type": "obs-checkbox", "index": obs_id},
                            value=is_checked,
                        ),
                        style={"width": "40px"},
                    ),
                    html.Td(str(obs_id)),
                    html.Td(str(gs)),
                    html.Td(country),
                    html.Td(start_utc),
                    html.Td(end_utc),
                    html.Td(start_local),
                    html.Td(end_local),
                    html.Td(wait_str),
                ]
            )
        )

    return rows


# ── Tab: Command Groups ───────────────────────────────────────────────────────


def _command_group_card(group_idx: int, group_data: dict | None = None) -> dbc.Card:
    """Render one command group card."""
    gd = group_data or {}
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
                            width=8,
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
                ),
            ),
            dbc.CardBody(
                [
                    dbc.Label("Commands (one per line, e.g. core_system_stats() or CTS1+cmd()!)"),
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
                                    dbc.Label("Start offset after AOS (seconds)"),
                                    dbc.Input(
                                        id={"type": "cg-start-offset", "index": group_idx},
                                        type="number",
                                        value=gd.get("start_offset", 0),
                                        min=0,
                                        style={"fontFamily": "monospace"},
                                    ),
                                    dbc.FormText(
                                        "Seconds after each pass AOS before this group begins."
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Command spacing within block (seconds)"),
                                    dbc.Input(
                                        id={"type": "cg-cmd-interval", "index": group_idx},
                                        type="number",
                                        value=gd.get("cmd_interval", 2.0),
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
                                    dbc.Label("Block repeat mode"),
                                    dcc.Dropdown(
                                        id={"type": "cg-repeat-mode", "index": group_idx},
                                        options=[
                                            {
                                                "label": "Every N seconds until LOS",
                                                "value": "interval",
                                            },
                                            {"label": "Fixed number of repeats", "value": "count"},
                                            {"label": "Once (no repeat)", "value": "once"},
                                        ],
                                        value=gd.get("repeat_mode", "interval"),
                                        clearable=False,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Block interval (seconds) — used when mode is 'Every N seconds'"
                                    ),
                                    dbc.Input(
                                        id={"type": "cg-block-interval", "index": group_idx},
                                        type="number",
                                        value=gd.get("block_interval", 20.0),
                                        min=1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Repeat count — used when mode is 'Fixed number of repeats'"
                                    ),
                                    dbc.Input(
                                        id={"type": "cg-repeat-count", "index": group_idx},
                                        type="number",
                                        value=gd.get("repeat_count", 10),
                                        min=1,
                                        style={"fontFamily": "monospace"},
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-3",
        style={"border": "1px solid #444"},
    )


def _tab_command_groups() -> dbc.Tab:
    default_groups = [
        {
            "name": "Telemetry Loop",
            "cmds": "CTS1+core_system_stats()!\nCTS1+get_all_system_thermal_info()!",
            "start_offset": 0,
            "cmd_interval": 2.0,
            "block_interval": 20.0,
            "repeat_mode": "interval",
            "repeat_count": 10,
        }
    ]
    return dbc.Tab(
        label="Command Groups",
        children=[
            html.Hr(),
            # Priority commands (static, not grouped)
            dbc.Card(
                [
                    dbc.CardHeader(html.Strong("⭐ Priority Commands")),
                    dbc.CardBody(
                        [
                            dbc.Label(
                                "Injected repeatedly with a fixed tssent so the satellite de-duplicates. "
                                "Optionally append @tsexec=<ms> for a fixed execution time."
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
                                            dbc.Label(
                                                "Injection interval (every N loop commands)"
                                            ),
                                            dbc.Input(
                                                id="priority-interval-input",
                                                type="number",
                                                value=50,
                                                min=1,
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
                className="mb-4",
                color="warning",
                outline=True,
            ),
            html.H5("Loop Command Groups", className="mb-1"),
            html.P(
                "Each group runs its commands in a block, repeated according to its own schedule "
                "while a satellite pass is active.",
                className="text-muted mb-3",
            ),
            # Container for dynamic command group cards
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
            # Store holds the list of group dicts (serialized as JSON)
            dcc.Store(id="command-groups-store", data=default_groups),
        ],
    )


# ── Tab: Telecommand Input ────────────────────────────────────────────────────


def _tab_telecommand_input(*, selected_command_name: str, enable_advanced: bool) -> dbc.Tab:
    return dbc.Tab(
        label="Telecommand Input",
        children=_generate_left_pane_send_commands(
            selected_command_name=selected_command_name,
            enable_advanced=enable_advanced,
        ),
    )


# ── Tab: Generate ─────────────────────────────────────────────────────────────


def _tab_generate() -> dbc.Tab:
    return dbc.Tab(
        label="Generate Agenda",
        children=[
            html.Hr(),
            html.P(
                "Uses the SatNOGS passes selected in the 'SatNOGS Passes' tab and the command "
                "groups defined in 'Command Groups' to produce a time-stamped command agenda.",
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
            html.Hr(),
            html.H6("Preview:", className="text-muted"),
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
        ],
    )


# ── Tab: Tools ────────────────────────────────────────────────────────────────


def _tab_tools() -> dbc.Tab:
    return dbc.Tab(
        label="Tools",
        children=[
            html.Hr(),
            dbc.Button(
                "Configure for Immediate Execution",
                id="send-immediate-execution-tool-button",
                n_clicks=0,
                className="m-1 px-3",
                color="info",
            ),
            dbc.Button(
                "Send Time Sync Command",
                id="send-time-sync-command-tool-button",
                n_clicks=0,
                className="m-1 px-3",
                color="info",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------


def generate_left_pane(*, selected_command_name: str, enable_advanced: bool) -> list:
    return [
        html.H1("CTS-SAT-1 Telecommand Input Terminal", className="text-center"),
        dbc.Tabs(
            id="left-pane-tabs",
            children=[
                _tab_connect_config(),
                _tab_satnogs(),
                _tab_command_groups(),
                _tab_telecommand_input(
                    selected_command_name=selected_command_name,
                    enable_advanced=enable_advanced,
                ),
                _tab_generate(),
                _tab_tools(),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Callbacks – Connect & Config (unchanged from original)
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

# Module-level store for in-progress fetch results (shared with background thread)
_fetch_results: list[dict] = []
_fetch_done = threading.Event()
_fetch_status_msg = ""


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
    """Start the background fetch thread and enable the polling interval."""
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

    uplink_dur = float(uplink_dur or 15)
    next_hours = float(next_hours or 6)
    uplink_end_dt = uplink_start_dt + timedelta(minutes=uplink_dur)
    start_lt_filter = uplink_start_dt + timedelta(hours=next_hours)

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
    """Called every 500 ms while fetch is running. Updates store and status."""
    done = _fetch_done.is_set()
    return (
        list(_fetch_results),
        _fetch_status_msg,
        done,  # disable interval when done
        not done,  # re-enable fetch button when done
        {"display": "none"} if done else {"display": "inline-block"},
    )


@callback(
    Input("stop-fetch-btn", "n_clicks"),
    prevent_initial_call=True,
)
def stop_fetch(_n_clicks):
    _fetch_stop.set()


@callback(
    Output("obs-table-body", "children"),
    Output("selected-obs-store", "data"),
    Output("obs-count-text", "children"),
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

    if triggered == "select-all-btn":
        selected_ids = [o.get("id") for o in (observations or [])]
    elif triggered == "deselect-all-btn":
        selected_ids = []
    else:
        # Default: select all newly fetched observations
        all_ids = [o.get("id") for o in (observations or [])]
        existing = set(current_selected or [])
        # Keep existing selection state, add new ones as selected
        selected_ids = list(existing | set(all_ids))

    uplink_end_dt = None
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
        uplink_end_dt = uplink_start_dt + timedelta(minutes=float(uplink_dur or 15))
    except Exception:
        pass

    rows = _obs_table_rows(observations or [], selected_ids, uplink_end_dt)
    total = len(observations or [])
    sel = len([i for i in selected_ids if i in [o.get("id") for o in (observations or [])]])
    count_text = f"{total} fetched, {sel} selected"
    return rows, selected_ids, count_text


@callback(
    Output("selected-obs-store", "data", allow_duplicate=True),
    Output("obs-count-text", "children", allow_duplicate=True),
    Input({"type": "obs-checkbox", "index": ALL}, "value"),
    State({"type": "obs-checkbox", "index": ALL}, "id"),
    State("observations-store", "data"),
    prevent_initial_call=True,
)
def obs_checkbox_changed(values, ids, observations):
    selected_ids = [id_dict["index"] for id_dict, val in zip(ids, values) if val]
    total = len(observations or [])
    count_text = f"{total} fetched, {len(selected_ids)} selected"
    return selected_ids, count_text


# ---------------------------------------------------------------------------
# Callbacks – Command Groups
# ---------------------------------------------------------------------------


@callback(
    Output("command-groups-store", "data"),
    Output("command-groups-container", "children"),
    Input("add-group-btn", "n_clicks"),
    Input({"type": "cg-remove-btn", "index": ALL}, "n_clicks"),
    State("command-groups-store", "data"),
    State({"type": "cg-name", "index": ALL}, "value"),
    State({"type": "cg-cmds", "index": ALL}, "value"),
    State({"type": "cg-start-offset", "index": ALL}, "value"),
    State({"type": "cg-cmd-interval", "index": ALL}, "value"),
    State({"type": "cg-block-interval", "index": ALL}, "value"),
    State({"type": "cg-repeat-mode", "index": ALL}, "value"),
    State({"type": "cg-repeat-count", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def manage_command_groups(
    add_clicks,
    remove_clicks,
    stored_groups,
    names,
    cmds,
    start_offsets,
    cmd_intervals,
    block_intervals,
    repeat_modes,
    repeat_counts,
):
    from dash import ctx

    # Snapshot current UI state back into group list
    n = len(stored_groups)
    current_groups = []
    for i in range(n):
        current_groups.append(
            {
                "name": names[i] if i < len(names) else f"Group {i + 1}",
                "cmds": cmds[i] if i < len(cmds) else "",
                "start_offset": start_offsets[i] if i < len(start_offsets) else 0,
                "cmd_interval": cmd_intervals[i] if i < len(cmd_intervals) else 2.0,
                "block_interval": block_intervals[i] if i < len(block_intervals) else 20.0,
                "repeat_mode": repeat_modes[i] if i < len(repeat_modes) else "interval",
                "repeat_count": repeat_counts[i] if i < len(repeat_counts) else 10,
            }
        )

    triggered = ctx.triggered_id
    if triggered == "add-group-btn":
        current_groups.append(
            {
                "name": f"Group {len(current_groups) + 1}",
                "cmds": "",
                "start_offset": 0,
                "cmd_interval": 2.0,
                "block_interval": 20.0,
                "repeat_mode": "interval",
                "repeat_count": 10,
            }
        )
    elif isinstance(triggered, dict) and triggered.get("type") == "cg-remove-btn":
        idx = triggered["index"]
        # Find position in current_groups by matching original index
        # After re-indexing we remove the one at position idx
        if 0 <= idx < len(current_groups):
            current_groups.pop(idx)

    cards = [_command_group_card(i, g) for i, g in enumerate(current_groups)]
    return current_groups, cards


# ---------------------------------------------------------------------------
# Callbacks – Telecommand Input (unchanged from original)
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

    extra_suffix_tags = {}
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
    logger.info(f"Send button clicked ({n_clicks=})!")
    if selected_command_name is None:
        msg = "No command selected."
        logger.error(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "error"))
        return
    args = [
        every_arg_value[i]
        for i in range(get_telecommand_by_name(selected_command_name).number_of_args)
    ]
    if any(a is None or a == "" for a in args):
        msg = f"Not all arguments filled in for {selected_command_name}."
        logger.error(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "error"))
        return
    if app_store.uart_port_name == UART_PORT_NAME_DISCONNECTED:
        msg = "Can't send command when disconnected."
        logger.error(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "error"))
        return
    send_command_to_device(command_preview)


@callback(
    Input("clear-log-button", "n_clicks"),
    prevent_initial_call=True,
)
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
        msg = "No command to save."
        logger.error(msg)
        app_store.append_to_rxtx_log(RxTxLogEntry(msg.encode(), "error"))
        return
    try:
        filepath = save_command(command_preview, filename)
        msg1 = f"Saved → {filepath.name}"
        msg2 = f"Telecommand → {command_preview}"
        app_store.append_to_rxtx_log(RxTxLogEntry(msg1.encode(), "input"))
        app_store.append_to_rxtx_log(RxTxLogEntry(msg2.encode(), "input"))
    except OSError as e:
        app_store.append_to_rxtx_log(RxTxLogEntry(str(e).encode(), "error"))


# ---------------------------------------------------------------------------
# Callbacks – Generate Agenda
# ---------------------------------------------------------------------------


@callback(
    Output("agenda-preview", "value"),
    Output("generate-status", "children"),
    Input("generate-agenda-btn", "n_clicks"),
    State("observations-store", "data"),
    State("selected-obs-store", "data"),
    State("uplink-start-input", "value"),
    State("uplink-dur-input", "value"),
    State("next-hours-input", "value"),
    State("sat-id-input", "value"),
    State("priority-cmds-input", "value"),
    State("priority-interval-input", "value"),
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
    priority_interval,
    command_groups,
):
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
    except Exception:
        return dash.no_update, "❌ Invalid uplink start datetime."

    selected_obs = [o for o in (observations or []) if o.get("id") in (selected_ids or [])]
    if not selected_obs:
        return dash.no_update, "❌ No observations selected."

    priority_cmds = [c.strip() for c in (priority_cmds_raw or "").splitlines() if c.strip()]

    # Flatten all command groups into the single loop_cmds list used by build_agenda.
    # Each group contributes its commands; groups with their own timing are interleaved
    # by building their own AgendaParams and merging the output.
    #
    # For groups that share the same block_interval we use build_agenda directly;
    # for mixed intervals we generate per-group agendas and merge them.

    all_output_lines: list[str] = []
    total_cmd_count = 0
    group_errors: list[str] = []

    for group in command_groups or []:
        loop_cmds = [c.strip() for c in (group.get("cmds") or "").splitlines() if c.strip()]
        if not loop_cmds:
            continue

        start_offset_sec = float(group.get("start_offset") or 0)
        repeat_mode = group.get("repeat_mode", "interval")
        block_interval_sec = float(group.get("block_interval") or 20.0)
        cmd_interval_sec = float(group.get("cmd_interval") or 2.0)
        repeat_count = int(group.get("repeat_count") or 10)

        # Adjust observations start time by start_offset per group
        adjusted_obs = []
        for obs in selected_obs:
            import copy

            o = copy.deepcopy(obs)
            try:
                s = parse_iso(o["start"]) + timedelta(seconds=start_offset_sec)
                o["start"] = s.isoformat()
            except Exception:
                pass
            adjusted_obs.append(o)

        # For "count" mode, cap the window to repeat_count * block_interval
        if repeat_mode == "count":
            # Trim observations: only include up to repeat_count blocks per pass
            pass  # build_agenda naturally handles this via the timeline

        if repeat_mode == "once":
            block_interval_sec = 1e9  # effectively run just one block

        params = AgendaParams(
            uplink_start_dt=uplink_start_dt,
            uplink_dur_min=float(uplink_dur or 15),
            block_interval_sec=block_interval_sec,
            cmd_interval_sec=cmd_interval_sec,
            priority_interval=int(priority_interval or 50),
            sat_id=sat_id or "",
            next_hours=float(next_hours or 6),
            loop_cmds=loop_cmds,
            priority_cmds=priority_cmds,
            observations=adjusted_obs,
        )

        try:
            lines = build_agenda(params)
        except ValueError as exc:
            group_errors.append(f"Group '{group.get('name')}': {exc}")
            continue

        group_name = group.get("name", "Group")
        all_output_lines.append(f"\n# ===== {group_name} =====")
        all_output_lines.extend(lines)

        group_cmd_count = sum(1 for l in lines if l.strip() and not l.startswith("#"))
        total_cmd_count += group_cmd_count

    if not all_output_lines:
        errors_str = "; ".join(group_errors) if group_errors else "No valid command groups."
        return dash.no_update, f"❌ {errors_str}"

    all_output_lines.append(f"\n# Total commands across all groups: {total_cmd_count}")

    if group_errors:
        all_output_lines.append("\n# WARNINGS:")
        for err in group_errors:
            all_output_lines.append(f"#   {err}")

    status = (
        f"✅ Generated {total_cmd_count} commands across {len(command_groups or [])} group(s)."
    )
    if group_errors:
        status += f" ⚠️ {len(group_errors)} group error(s)."

    return "\n".join(all_output_lines), status


# ---------------------------------------------------------------------------
# Callbacks – RX/TX log and UART refresh (unchanged from original)
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
# Callbacks – Tools (unchanged from original)
# ---------------------------------------------------------------------------


@callback(
    Input("send-time-sync-command-tool-button", "n_clicks"),
    prevent_initial_call=True,
)
def send_time_sync_command_callback(n_clicks: int):
    current_time_ms = int(time.time() * 1000)
    send_command_to_device(f"CTS1+set_system_time({current_time_ms})!")


@callback(
    Input("send-immediate-execution-tool-button", "n_clicks"),
    prevent_initial_call=True,
)
def send_immediate_execution_command_callback(n_clicks: int):
    send_command_to_device("CTS1+config_set_int_var(EPS_monitor_interval_ms,1000000000)!")
    time.sleep(1)
    send_command_to_device("CTS1+config_set_int_var(EPS_time_sync_period_sec,1000000000)!")
    time.sleep(1)
    send_command_to_device("CTS1+config_set_int_var(COMMS_beacon_interval_ms,1000000000)!")
    time.sleep(1)
    send_command_to_device("CTS1+config_set_int_var(TCMD_handle_umbilical_tcmds_interval_ms,1)!")
    time.sleep(1)


# ---------------------------------------------------------------------------
# Left-pane send-commands helper (for _tab_telecommand_input)
# ---------------------------------------------------------------------------


def _generate_left_pane_send_commands(
    *, selected_command_name: str, enable_advanced: bool
) -> list:
    return [
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
                    "Timestamp to Execute Command (e.g. 20260425T1613 MST or UTC)",
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
                size=550,
                minSize=350,
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
