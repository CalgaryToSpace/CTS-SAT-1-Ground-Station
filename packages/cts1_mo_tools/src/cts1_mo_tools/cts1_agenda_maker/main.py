"""
CTS-SAT-1 Command Agenda Generator
==================================
Fetches SatNOGS observations, lets you define repeating and priority
telecommands, and produces a time-stamped command agenda file.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

import asyncio
import collections
import contextlib
import functools
import re
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, assert_never

import polars as pl
import polars_reverse_geocode
import pycountry
from dotenv import load_dotenv
from nicegui import ui

from .satnogs_data import iter_future_observation_pages

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
COMMAND_PREFIX = "CTS1+"
COMMAND_SUFFIX = "!"
VERY_EVIL_COMMAND_SUBSTRING = ")!@"  # Causes suffix tags to be ignored.
TSSENT_INCREMENT_MS = 1000

# -------------------------------------------------------------
# HELPERS (framework-independent -- unchanged from the dpg version)
# -------------------------------------------------------------


def parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 string to a timezone-aware datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        msg = f"datetime has no timezone! That's bad! Input string: {s}"
        raise ValueError(msg)

    return dt


def dt_to_local_str(dt: datetime) -> str:
    """Format a timezone-aware datetime as local time in ISO format with offset."""
    return dt.astimezone().replace(microsecond=0).isoformat()


def format_command(name_args: str, tssent_ms: int, tsexec_ms: int) -> str:
    """
    Build a CTS1 telecommand string.

    Args:
    name_args: e.g. 'hello_world()' or 'echo_back_args(foo,bar)'
    tssent_ms: unix ms when command is sent
    tsexec_ms: unix ms for scheduled execution (0 = immediate)
    """
    name_args = (
        name_args.strip().removeprefix(COMMAND_PREFIX).removesuffix(COMMAND_SUFFIX)
    )

    out = "".join(
        [
            COMMAND_PREFIX,
            name_args,
            f"@tssent={tssent_ms}",
            f"@tsexec={tsexec_ms}",
            COMMAND_SUFFIX,
        ]
    )

    tssent_utc = datetime.fromtimestamp(tssent_ms / 1000, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    tsexec_utc = (
        "immediate"
        if tsexec_ms == 0
        else datetime.fromtimestamp(tsexec_ms / 1000, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    return f"{out}  # tssent={tssent_utc} tsexec={tsexec_utc}"


def format_timedelta(delta: timedelta) -> str:
    if delta < timedelta(seconds=0):
        return f"{-delta} ago"

    return str(delta)


@functools.lru_cache(maxsize=2500)
def _lat_lon_to_country(latitude: float, longitude: float) -> str | None:
    df_row = pl.DataFrame({"lat": [latitude], "lon": [longitude]}).select(
        polars_reverse_geocode.find_closest_country(
            lat=pl.col("lat"),
            long=pl.col("lon"),
            cache_mode="cache_forever",
        )
    )
    country_code = df_row.item()
    gs_country_obj = pycountry.countries.get(alpha_2=country_code)
    return gs_country_obj.name if gs_country_obj else None


@functools.lru_cache(maxsize=2500)
def _country_code_to_country_name(country_code: str) -> str | None:

    gs_country_obj = pycountry.countries.get(alpha_2=country_code)
    return gs_country_obj.name if gs_country_obj else None


def _format_satnogs_observation_info(satnogs_observation: dict[str, Any]) -> str:
    obs_id = satnogs_observation.get("id", "?")
    gs = satnogs_observation.get("ground_station", "?")
    gs_lat_lon = (
        str(satnogs_observation.get("station_lat", "?"))
        + ", "
        + str(satnogs_observation.get("station_lng", "?"))
    )
    gs_country_name = _lat_lon_to_country(
        satnogs_observation["station_lat"], satnogs_observation["station_lng"]
    )

    return " | ".join(
        [
            f"Observation {obs_id}",
            f"GS {gs} @ {gs_lat_lon} (in {gs_country_name})",
            f"Observer: {satnogs_observation.get('observer', '?')}",
        ]
    )


def _parse_priority_cmd(raw: str) -> tuple[str, int]:
    s = raw.strip()
    s = s.removeprefix(COMMAND_PREFIX)
    s = s.removesuffix(COMMAND_SUFFIX)
    p_tsexec = 0
    m = re.search(r"@tsexec=(\d+)", s)
    if m:
        p_tsexec = int(m.group(1))
        # Remove @tsexec section, leaving the rest (including any other suffix tags).
        s = (s[: m.start()] + s[m.end() :]).strip()
    return s, p_tsexec


@dataclass
class AgendaParams:
    uplink_start_dt: datetime
    uplink_dur_min: float
    block_interval_sec: float
    cmd_interval_sec: float
    priority_interval: int
    sat_id: str
    next_hours: float
    loop_cmds: list[str]
    priority_cmds: list[str] = field(default_factory=list[str])
    observations: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def build_agenda(params: AgendaParams) -> list[str]:  # noqa: C901, PLR0912, PLR0915
    """Pure agenda generation. Raises ValueError for invalid inputs."""
    if not params.loop_cmds:
        msg = "No loop commands entered."
        raise ValueError(msg)

    uplink_end_dt = params.uplink_start_dt + timedelta(minutes=params.uplink_dur_min)

    valid_obs = [
        obs for obs in params.observations if parse_iso(obs["start"]) > uplink_end_dt
    ]
    if not valid_obs:
        msg_0 = "No valid observations (all are before end of uplink window)."
        raise ValueError(msg_0)

    def _fmt(dt: datetime) -> str:
        return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # -- Build AOS/LOS event list -------------------------------
    # Each entry: (datetime, event_type, obs)
    # event_type is "AOS" or "LOS"
    events: list[tuple[datetime, Literal["AOS", "LOS"], dict[str, Any]]] = []
    for obs in valid_obs:
        events.append((parse_iso(obs["start"]), "AOS", obs))
        events.append((parse_iso(obs["end"]), "LOS", obs))
    events.sort(key=lambda e: e[0])

    # -- Header -------------------------------------------------
    output_lines: list[str] = []
    output_lines.append("# CTS-SAT-1 Command Agenda")
    output_lines.append(f"# Satellite NORAD ID: {params.sat_id}")
    output_lines.append(
        f"# Uplink window: {_fmt(params.uplink_start_dt)} -> {_fmt(uplink_end_dt)}"
        f" ({params.uplink_dur_min:.1f} min)"
    )
    output_lines.append(
        f"# Observation fetch window end: {params.next_hours} hrs after uplink"
    )
    output_lines.append(f"# Generated at: {datetime.now(tz=UTC).isoformat()}")
    output_lines.append(f"# Valid observations: {len(valid_obs)}")
    output_lines.append(
        f"# Time between loop command blocks: {params.block_interval_sec:.1f} s"
    )
    output_lines.append(
        f"# Time between commands in a loop block: {params.cmd_interval_sec:.1f} s"
    )
    output_lines.append(
        f"# Priority command injection interval: every {params.priority_interval} loop commands"  # noqa: E501
    )
    output_lines.append(f"# Loop commands ({len(params.loop_cmds)}):")
    output_lines.extend([f"#   {cmd}" for cmd in params.loop_cmds])
    output_lines.append(f"# Priority commands ({len(params.priority_cmds)}):")
    output_lines.extend([f"#   {cmd}" for cmd in params.priority_cmds])
    output_lines.append("")

    tssent_dt = params.uplink_start_dt
    cmd_count = 0
    block_interval = timedelta(seconds=params.block_interval_sec)
    cmd_interval = timedelta(seconds=params.cmd_interval_sec)

    # Assign each priority command a fixed tssent and emit them upfront
    priority_tssent: dict[str, datetime] = {}
    if params.priority_cmds:
        output_lines.append("# PRIORITY COMMANDS")
        for priority_cmd in params.priority_cmds:
            p_name, p_tsexec = _parse_priority_cmd(priority_cmd)
            priority_tssent[priority_cmd] = tssent_dt
            output_lines.append(
                format_command(p_name, int(tssent_dt.timestamp() * 1000), p_tsexec)
            )
            tssent_dt += timedelta(milliseconds=TSSENT_INCREMENT_MS)
        output_lines.append("")

    # -- Single unified timeline --------------------------------
    # Determine overall start and end of the command window.
    first_aos = parse_iso(valid_obs[0]["start"])
    last_los = max(parse_iso(obs["end"]) for obs in valid_obs)

    event_idx = 0
    active_passes: set[int] = set()

    tsexec_dt = first_aos
    output_lines.append(f"# Timeline start: {_fmt(first_aos)}")
    output_lines.append("")

    while tsexec_dt < last_los:
        # Inject any AOS/LOS events that fall before (or at) the current tsexec.
        while event_idx < len(events) and events[event_idx][0] <= tsexec_dt:
            ev_dt, ev_type, satnogs_observation = events[event_idx]
            obs_id = satnogs_observation.get("id", "?")

            if ev_type == "AOS":
                active_passes.add(obs_id)
            elif ev_type == "LOS":
                active_passes.discard(obs_id)
            else:
                assert_never(ev_type)

            output_lines.append(
                " | ".join(
                    [
                        f"# {ev_type}",
                        _fmt(ev_dt),
                        _format_satnogs_observation_info(satnogs_observation),
                        f"{len(active_passes)} station(s) in sight",
                    ]
                )
            )
            event_idx += 1

        # Only emit commands while at least one pass is active.
        if active_passes:
            cmd_tsexec_dt = tsexec_dt
            for cmd_raw in params.loop_cmds:
                if (
                    params.priority_cmds
                    and cmd_count > 0
                    and (cmd_count % params.priority_interval) == 0
                ):
                    output_lines.append(f"# PRIORITY [{cmd_count}]")
                    for priority_cmd in params.priority_cmds:
                        p_name, p_tsexec = _parse_priority_cmd(priority_cmd)
                        output_lines.append(
                            format_command(
                                p_name,
                                int(priority_tssent[priority_cmd].timestamp() * 1000),
                                p_tsexec,
                            )
                        )

                output_lines.append(
                    format_command(
                        cmd_raw,
                        int(tssent_dt.timestamp() * 1000),
                        int(cmd_tsexec_dt.timestamp() * 1000),
                    )
                )
                tssent_dt += timedelta(milliseconds=TSSENT_INCREMENT_MS)
                cmd_tsexec_dt += cmd_interval
                cmd_count += 1

        tsexec_dt += block_interval

    # Flush any remaining LOS events after the last command tick.
    while event_idx < len(events):
        ev_dt, ev_type, satnogs_observation = events[event_idx]
        obs_id = satnogs_observation.get("id", "?")

        if ev_type == "AOS":
            active_passes.add(obs_id)
        elif ev_type == "LOS":
            active_passes.discard(obs_id)
        else:
            assert_never(ev_type)

        output_lines.append(
            " | ".join(
                [
                    f"# {ev_type}",
                    _fmt(ev_dt),
                    _format_satnogs_observation_info(satnogs_observation),
                    f"{len(active_passes)} station(s) in sight",
                ]
            )
        )
        event_idx += 1

    # Validate bad error case:
    for line in output_lines:
        if VERY_EVIL_COMMAND_SUBSTRING in line:
            raise ValueError(
                "Internal error: Command line contains bad error case: " + line
            )

    return output_lines


# -------------------------------------------------------------
# TABLE COLUMNS
# -------------------------------------------------------------
OBS_TABLE_COLUMNS = [
    {"name": "id", "label": "Obs ID", "field": "id", "sortable": True, "align": "left"},
    {
        "name": "ground_station",
        "label": "GS ID",
        "field": "ground_station",
        "sortable": True,
        "align": "left",
    },
    {"name": "country", "label": "Country", "field": "country", "align": "left"},
    {
        "name": "start_utc",
        "label": "Start (UTC)",
        "field": "start_utc",
        "sortable": True,
        "align": "left",
    },
    {"name": "end_utc", "label": "End (UTC)", "field": "end_utc", "align": "left"},
    {
        "name": "start_local",
        "label": "Start (Local)",
        "field": "start_local",
        "align": "left",
    },
    {
        "name": "end_local",
        "label": "End (Local)",
        "field": "end_local",
        "align": "left",
    },
    {
        "name": "wait",
        "label": "Wait (uplink LOS -> pass AOS)",
        "field": "wait",
        "align": "left",
    },
]


# -------------------------------------------------------------
# PER-SESSION UI STATE
# -------------------------------------------------------------
@dataclass
class SessionState:
    """Holds per-browser-tab state (mirrors the old module-level `state` dict)."""

    observations: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    generated_commands: list[str] = field(default_factory=list[str])
    fetch_stop: threading.Event = field(default_factory=threading.Event)
    fetching: bool = False


def _make_obs_row(
    obs: dict[str, Any], uplink_end_dt: datetime | None
) -> dict[str, Any]:
    """Build a table row dict (raw obs fields + computed display fields)."""
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    with contextlib.suppress(Exception):
        start_dt = parse_iso(obs["start"])
    with contextlib.suppress(Exception):
        end_dt = parse_iso(obs["end"])

    start_utc = (
        start_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if start_dt
        else obs.get("start", "?")
    )
    end_utc = (
        end_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_dt
        else obs.get("end", "?")
    )
    start_local = dt_to_local_str(start_dt) if start_dt else "?"
    end_local = dt_to_local_str(end_dt) if end_dt else "?"

    country_str = "?"
    with contextlib.suppress(Exception):
        lat = obs["station_lat"]
        lng = obs["station_lng"]
        country_str = _lat_lon_to_country(lat, lng) or "?"

    wait_str = "N/A"
    if uplink_end_dt is not None and start_dt is not None:
        delta = start_dt - uplink_end_dt
        wait_str = format_timedelta(delta)

    # Spread the raw obs dict so build_agenda() can consume selected rows
    # directly (it needs "start", "end", "station_lat", etc.) and layer the
    # computed display fields on top.
    row = dict(obs)
    row.update(
        {
            "id": obs.get("id", "?"),
            "ground_station": obs.get("ground_station", "?"),
            "country": country_str,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "start_local": start_local,
            "end_local": end_local,
            "wait": wait_str,
        }
    )
    return row


def _build_coverage_series(observations: list[dict[str, Any]]) -> list[list[Any]]:
    """Build a strictly-stepped [timestamp_ms, concurrent_count, top_countries]
    series.

    Each observation contributes a +1 event at AOS and a -1 event at LOS.
    Every event is emitted as two points sharing the same x (the state
    just before, then just after), so a plain straight-line series draws
    vertical jumps at event times and flat horizontal runs in between --
    no reliance on ECharts' own step interpolation.

    Each point's third element is a human-readable summary of the up-to-5
    most frequent ground station countries among the observations active
    at that instant, for display in the chart tooltip.
    """
    events: list[tuple[datetime, int, str | None]] = []
    for obs in observations:
        start_dt = parse_iso(obs["start"])
        end_dt = parse_iso(obs["end"])

        country: str | None = None
        with contextlib.suppress(Exception):
            country = _lat_lon_to_country(obs["station_lat"], obs["station_lng"])

        events.append((start_dt, 1, country))
        events.append((end_dt, -1, country))

    if not events:
        return []

    events.sort(key=lambda e: e[0])

    active_countries: collections.Counter[str] = collections.Counter()

    def _top_countries_summary() -> str:
        if not active_countries:
            return "(none)"
        return ", ".join(f"{name} ({n})" for name, n in active_countries.most_common(5))

    points: list[list[Any]] = []
    count = 0
    for t, delta, country in events:
        ts_ms = t.timestamp() * 1000
        points.append([ts_ms, count, _top_countries_summary()])
        if country is not None:
            active_countries[country] += delta
            if active_countries[country] <= 0:
                del active_countries[country]
        count += delta
        points.append([ts_ms, count, _top_countries_summary()])
    return points


async def _fetch_pages(
    sat_id: str,
    start_gt_filter: datetime,
    start_lt_filter: datetime,
    stop_event: threading.Event,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    """Async generator that drives the blocking SatNOGS iterator in a
    background thread and yields pages back on the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def worker() -> None:
        try:
            for page in iter_future_observation_pages(
                sat_id,
                start_gt_filter=start_gt_filter,
                start_lt_filter=start_lt_filter,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, ("page", page))
                if stop_event.is_set():
                    break
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = await queue.get()
        if kind == "page":
            yield payload
        elif kind == "done":
            return
        elif kind == "error":
            raise payload


# -------------------------------------------------------------
# GUI
# -------------------------------------------------------------
@ui.page("/")
async def index() -> None:  # noqa: C901, PLR0915
    ui.add_head_html(
        """
        <style>
            body { background-color: #161b1e; }
        </style>
        """
    )
    ui.colors(primary="#2870a0")

    session = SessionState()
    now_local = datetime.now().astimezone().replace(microsecond=0).isoformat()

    ui.label("CTS-SAT-1 Command Agenda Generator").classes("text-2xl font-bold mt-2")

    with ui.tabs().classes("w-full") as tabs:
        settings_tab = ui.tab("Settings")
        commands_tab = ui.tab("Commands")
        generate_tab = ui.tab("Generate")

    with ui.tab_panels(tabs, value=settings_tab).classes("w-full"):
        # ==================================================
        # TAB 1 - SETTINGS
        # ==================================================
        with ui.tab_panel(settings_tab):
            with ui.expansion("Uplink Pass Window", value=True).classes("w-full"):
                uplink_start = ui.input(
                    "Start of Uplink Pass (ISO with timezone)",
                    value=now_local,
                    placeholder="2024-05-01T12:00:00-07:00",
                ).classes("w-96")
                uplink_start.tooltip(
                    "ISO 8601 with timezone offset. Timezone is required.\n"
                    "Examples: 2024-05-01T12:00:00-07:00 or 2024-05-01T19:00:00Z\n"
                    "This sets the tssent for the first command.\n"
                    "Only observations that START after (uplink_start + duration)\n"
                    "will be included."
                )

                uplink_dur = ui.number(
                    "Uplink Pass Duration (minutes)",
                    value=15.0,
                    min=0.1,
                    max=60.0,
                    format="%.1f",
                ).classes("w-64")
                uplink_dur.tooltip(
                    "Fine to overestimate the duration by a few minutes."
                )

            with ui.expansion("SatNOGS Observations", value=True).classes("w-full"):
                with ui.row().classes("items-center"):
                    sat_id_input = ui.input(
                        "Satellite NORAD ID", value="69015", placeholder="e.g. 69015"
                    ).classes("w-40")
                    next_hours_input = ui.number(
                        "Fetch next (hrs after uplink)",
                        value=3.0,
                        min=0.1,
                        max=720.0,
                        format="%.1f",
                    ).classes("w-56")
                    fetch_btn = ui.button("Fetch Observations")
                    stop_fetch_btn = ui.button("Stop", color="negative")
                    stop_fetch_btn.set_visibility(False)
                    fetch_spinner = ui.spinner(size="lg")
                    fetch_spinner.set_visibility(False)

                ui.label("Select the observations to target with downlinks.").classes(
                    "text-gray-400"
                )

                obs_table = ui.table(
                    columns=OBS_TABLE_COLUMNS,
                    rows=[],
                    row_key="id",
                    selection="multiple",
                ).classes("w-full")
                obs_table.props("dense bordered separator=cell")

                ui.label(
                    "Observation coverage: number of observations active at "
                    "each point in time."
                ).classes("text-gray-400 mt-2")
                coverage_chart = ui.echart(
                    {
                        # Axis ticks/labels are computed in UTC (see xAxis
                        # below); the tooltip formatter additionally shows
                        # local time for convenience.
                        "useUTC": True,
                        "xAxis": {
                            "type": "time",
                            "name": "Time (UTC)",
                            "nameLocation": "middle",
                            "nameGap": 40,
                            "axisLabel": {
                                "formatter": "{yyyy}-{MM}-{dd}\n{HH}:{mm} UTC",
                            },
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "Observations",
                            "minInterval": 1,
                            "min": 0,
                            # Cap y-scale. 8 observations means "definitely covered",
                            # no need to indicate even more above 8.
                            "max": 8,
                        },
                        "tooltip": {
                            "trigger": "axis",
                            "axisPointer": {"type": "cross"},
                            ":formatter": (
                                "params => {"
                                "  if (!params || !params.length) return '';"
                                "  const p = params[0];"
                                "  const ts = p.value[0];"
                                "  const d = new Date(ts);"
                                "  const pad = n => String(n).padStart(2, '0');"
                                "  const utc = `${d.getUTCFullYear()}-"
                                "${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} "
                                "${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:"
                                "${pad(d.getUTCSeconds())} UTC`;"
                                "  const local = `${d.getFullYear()}-"
                                "${pad(d.getMonth() + 1)}-${pad(d.getDate())} "
                                "${pad(d.getHours())}:${pad(d.getMinutes())}:"
                                "${pad(d.getSeconds())} Local`;"
                                "  const countries = p.value[2] ?? '';"
                                "  return `Observations active: ${p.value[1]}"
                                "<br/>${utc}<br/>${local}"
                                "<br/>Station countries: ${countries}`;"
                                "}"
                            ),
                        },
                        "grid": {
                            "left": 60,
                            "right": 20,
                            "top": 20,
                            "bottom": 60,
                        },
                        "series": [
                            {
                                "type": "line",
                                "data": [],
                                "showSymbol": False,
                                "step": False,
                                "lineStyle": {"width": 2},
                                "areaStyle": {"opacity": 0.15},
                            }
                        ],
                    }
                ).classes("w-full h-64")

                with ui.row().classes("items-center"):
                    obs_count_text = ui.label("").classes("text-gray-400")
                    select_all_btn = ui.button("Select All")
                    deselect_all_btn = ui.button("Deselect All")

            with ui.expansion("Timing & Output", value=True).classes("w-full"):
                block_interval = ui.number(
                    "Time between loop command blocks (s)",
                    value=20.0,
                    min=0.1,
                    max=3600.0,
                    format="%.1f",
                ).classes("w-72")
                ui.label(
                    "How often a new block of loop commands is scheduled."
                ).classes("text-gray-400 text-sm")

                cmd_interval = ui.number(
                    "Time between commands in a loop block (s)",
                    value=2.0,
                    min=0.1,
                    max=600.0,
                    format="%.1f",
                ).classes("w-72")
                ui.label(
                    "tsexec spacing between consecutive commands within one block."
                ).classes("text-gray-400 text-sm")

                priority_interval = ui.number(
                    "Priority Command Injection Interval",
                    value=50,
                    min=1,
                    max=10000,
                    format="%.0f",
                ).classes("w-72")
                ui.label("Insert priority commands every N loop commands.").classes(
                    "text-gray-400 text-sm"
                )

        # ==================================================
        # TAB 2 - COMMANDS
        # ==================================================
        with ui.tab_panel(commands_tab):
            with ui.expansion(
                "Loop Commands  (executed repeatedly during each pass)", value=True
            ).classes("w-full"):
                ui.label(
                    "Enter one command per line.  Format:  command_name(arg1,arg2)\n"
                    "The tssent / tsexec tags are generated automatically."
                ).classes("text-gray-400 whitespace-pre-line")
                loop_cmds_input = (
                    ui.textarea(
                        value="\n".join(  # noqa: FLY002
                            [
                                "CTS1+core_system_stats()!",
                                "CTS1+get_all_system_thermal_info()!",
                            ]
                        )
                    )
                    .classes("w-full font-mono")
                    .props("rows=8")
                )

            with ui.expansion(
                "Priority Commands  (injected every N commands with the SAME "
                "tssent each time)",
                value=True,
            ).classes("w-full"):
                ui.label(
                    "Enter one command per line.\n"
                    "Optionally append  @tsexec=<ms>  for a fixed execution time, "
                    "or omit for immediate (0).\n"
                    "Each priority command keeps its first tssent so the "
                    "satellite de-duplicates."
                ).classes("text-gray-400 whitespace-pre-line")
                priority_cmds_input = (
                    ui.textarea(
                        value="\n".join(  # noqa: FLY002
                            [
                                "CTS1+config_set_int_var(TCMD_require_unique_tssent,1)!",
                                "CTS1+obc_set_stm32_sysclk_to_hse()!",
                            ]
                        )
                    )
                    .classes("w-full font-mono")
                    .props("rows=6")
                )

        # ==================================================
        # TAB 3 - GENERATE / PREVIEW
        # ==================================================
        with ui.tab_panel(generate_tab):
            with ui.row().classes("items-center"):
                generate_btn = ui.button("Generate Command Agenda")
                status_text = ui.label("")
                download_btn = ui.button("Download Agenda File", color="positive")
                download_btn.set_visibility(False)

            ui.separator()
            ui.label("Preview:").classes("text-gray-400")
            preview_text = ui.textarea(value="(generate agenda to see preview)")
            preview_text.props("readonly rows=25").classes("w-full font-mono")

    # -----------------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------------

    def set_status(msg: str, css_class: str = "text-white") -> None:
        status_text.text = msg
        status_text.classes(replace=f"{css_class}")

    def _update_obs_count() -> None:
        total = len(obs_table.rows)
        selected = len(obs_table.selected)
        obs_count_text.text = f"{total} fetched, {selected} selected"

    def _on_selection_change() -> None:
        _update_obs_count()

    obs_table.on("selection", _on_selection_change)

    def _update_coverage_chart() -> None:
        series = coverage_chart.options["series"][0]
        series["data"] = _build_coverage_series(session.observations)
        coverage_chart.update()

    def get_float(element: ui.number, default: float = 0.0) -> float:
        try:
            return float(element.value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            return default

    def get_int(element: ui.number, default: int = 0) -> int:
        try:
            return int(element.value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            return default

    def get_str(element: ui.input | ui.textarea) -> str:
        return str(element.value or "").strip()

    def _select_all() -> None:
        obs_table.selected = list(obs_table.rows)
        _update_obs_count()

    def _deselect_all() -> None:
        obs_table.selected = []
        _update_obs_count()

    select_all_btn.on_click(_select_all)
    deselect_all_btn.on_click(_deselect_all)

    async def fetch_observations() -> None:
        if session.fetching:
            return

        sat_id = get_str(sat_id_input)
        if not sat_id:
            set_status("[!] Enter a SatNOGS satellite ID first.", "text-yellow-400")
            return

        try:
            uplink_start_dt = parse_iso(get_str(uplink_start))
        except Exception:  # noqa: BLE001
            set_status(
                "[x] Invalid 'Start of Uplink Pass'. "
                "Use ISO format with timezone: 2024-05-01T12:00:00-07:00",
                "text-red-400",
            )
            return

        uplink_end_dt = uplink_start_dt + timedelta(minutes=get_float(uplink_dur, 15.0))
        next_hours = get_float(next_hours_input, 6.0)
        start_lt_filter = uplink_start_dt + timedelta(hours=next_hours)

        session.fetch_stop = threading.Event()
        session.fetching = True
        session.observations = []
        obs_table.rows = []
        obs_table.selected = []
        _update_obs_count()
        _update_coverage_chart()

        set_status("Fetching observations from SatNOGS...", "text-blue-300")
        fetch_btn.set_visibility(False)
        stop_fetch_btn.set_visibility(True)
        fetch_spinner.set_visibility(True)

        stopped = False
        try:
            async for page in _fetch_pages(
                sat_id, uplink_end_dt, start_lt_filter, session.fetch_stop
            ):
                session.observations.extend(page)
                session.observations.sort(key=lambda o: parse_iso(o["start"]))

                for obs in page:
                    row = _make_obs_row(obs, uplink_end_dt)
                    obs_table.rows.append(row)
                obs_table.rows.sort(key=lambda r: parse_iso(r["start"]))
                # Select newly-fetched observations by default, like the
                # original dpg checkbox-defaults-to-True behaviour.
                obs_table.selected = list(obs_table.rows)
                obs_table.update()
                _update_obs_count()
                _update_coverage_chart()

                set_status(
                    f"Fetching... {len(session.observations)} so far",
                    "text-blue-300",
                )
                if session.fetch_stop.is_set():
                    stopped = True
                    break

            if stopped:
                set_status(
                    f"[stop] Stopped. {len(session.observations)} observations loaded.",
                    "text-yellow-400",
                )
            else:
                set_status(
                    f"[ok] Loaded {len(session.observations)} future observations.",
                    "text-green-400",
                )
        except Exception as exc:  # noqa: BLE001
            set_status(f"[x] Fetch error: {exc}", "text-red-400")
        finally:
            session.fetching = False
            fetch_btn.set_visibility(True)
            stop_fetch_btn.set_visibility(False)
            fetch_spinner.set_visibility(False)

    def _stop_fetch() -> None:
        session.fetch_stop.set()

    fetch_btn.on_click(fetch_observations)
    stop_fetch_btn.on_click(_stop_fetch)

    def generate_agenda() -> None:
        try:
            uplink_start_dt = parse_iso(get_str(uplink_start))
        except Exception:  # noqa: BLE001
            set_status(
                "[x] Invalid 'Start of Uplink Pass'. "
                "Use ISO format with timezone: 2024-05-01T12:00:00-07:00",
                "text-red-400",
            )
            return

        loop_cmds = [
            c.strip() for c in get_str(loop_cmds_input).splitlines() if c.strip()
        ]
        priority_cmds = [
            c.strip() for c in get_str(priority_cmds_input).splitlines() if c.strip()
        ]
        # obs_table.selected rows already contain the raw SatNOGS fields
        # (start/end/station_lat/...) layered with the display fields.
        selected_obs = list(obs_table.selected)

        params = AgendaParams(
            uplink_start_dt=uplink_start_dt,
            uplink_dur_min=get_float(uplink_dur, 10.0),
            block_interval_sec=get_float(block_interval, 20.0),
            cmd_interval_sec=get_float(cmd_interval, 2.0),
            priority_interval=get_int(priority_interval, 50),
            sat_id=get_str(sat_id_input),
            next_hours=get_float(next_hours_input, 6.0),
            loop_cmds=loop_cmds,
            priority_cmds=priority_cmds,
            observations=selected_obs,
        )

        try:
            output_lines = build_agenda(params)
        except ValueError as exc:
            set_status(f"[x] {exc}", "text-red-400")
            download_btn.set_visibility(False)
            return

        session.generated_commands = output_lines
        preview_text.value = "\n".join(output_lines)
        total_cmd_count = sum(
            1 for line in output_lines if line.strip() and not line.startswith("#")
        )
        set_status(f"[ok] Generated {total_cmd_count} commands.", "text-green-400")
        download_btn.set_visibility(True)

    generate_btn.on_click(generate_agenda)

    def download_agenda() -> None:
        if not session.generated_commands:
            return
        content = "\n".join(session.generated_commands)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        ui.download(content.encode(), f"cts-sat1-agenda-{timestamp}.txt")

    download_btn.on_click(download_agenda)

    _update_obs_count()


def main() -> None:
    load_dotenv()
    ui.run(title="CTS-SAT-1 Command Agenda Generator", dark=True, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
