"""
CTS-SAT-1 Command Agenda Generator
====================================
Fetches SatNOGS observations, lets you define repeating and priority
telecommands, and produces a time-stamped command agenda file.
"""

# pyright: standard
# dearpygui has typing issues.

import contextlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import dearpygui.dearpygui as dpg
from dotenv import load_dotenv

from .satnogs_data import iter_future_observation_pages

_fetch_stop = threading.Event()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COMMAND_PREFIX = "CTS1+"
COMMAND_SUFFIX = "!"

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
state = {
    "observations": [],  # raw SatNOGS observation dicts
    "selected_obs_ids": set(),  # user-selected observation IDs
    "generated_commands": [],  # list of formatted command strings
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


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
    name_args = name_args.strip()

    out = f"{name_args}@tssent={tssent_ms}@tsexec={tsexec_ms}"

    if not out.startswith(COMMAND_PREFIX):
        out = f"{COMMAND_PREFIX}{out}"
    if not out.endswith(COMMAND_SUFFIX):
        out = f"{out}{COMMAND_SUFFIX}"

    return out


def set_status(
    msg: str, colour: tuple[int, int, int, int] = (255, 255, 255, 255)
) -> None:
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", msg)
        dpg.configure_item("status_text", color=colour)


def get_float(tag: str, default: float = 0.0) -> float:
    try:
        return float(dpg.get_value(tag))
    except Exception:
        return default


def get_int(tag: str, default: int = 0) -> int:
    try:
        return int(dpg.get_value(tag))
    except Exception:
        return default


def get_str(tag: str) -> str:
    try:
        return str(dpg.get_value(tag)).strip()
    except Exception:
        return ""


def _update_obs_count() -> None:
    if dpg.does_item_exist("obs_count_text"):
        total = len(state["observations"])
        selected = len(state["selected_obs_ids"])
        dpg.set_value("obs_count_text", f"{total} fetched, {selected} selected")


def format_timedelta(delta: timedelta) -> str:
    if delta < timedelta(seconds=0):
        return f"{-delta} ago"

    return str(delta)


# ─────────────────────────────────────────────────────────────
# SATNOGS
# ─────────────────────────────────────────────────────────────


def _append_obs_rows(
    obs_list: list[dict[str, Any]], uplink_end_dt: datetime | None
) -> None:
    for obs in obs_list:
        obs_id = obs.get("id", "?")
        gs = obs.get("ground_station", "?")

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

        wait_str = "N/A"
        if uplink_end_dt is not None and start_dt is not None:
            delta = start_dt - uplink_end_dt
            wait_str = format_timedelta(delta)

        with dpg.table_row(parent="obs_table"):

            def make_cb(oid: int) -> Callable[[Any, bool], None]:
                def cb(_: Any, v: bool) -> None:
                    if v:
                        state["selected_obs_ids"].add(oid)
                    else:
                        state["selected_obs_ids"].discard(oid)
                    _update_obs_count()

                return cb

            dpg.add_checkbox(default_value=True, callback=make_cb(obs_id))
            state["selected_obs_ids"].add(obs_id)  # select by default

            dpg.add_text(str(obs_id))
            dpg.add_text(str(gs))
            dpg.add_text(start_utc)
            dpg.add_text(end_utc)
            dpg.add_text(start_local)
            dpg.add_text(end_local)
            dpg.add_text(wait_str)

    _update_obs_count()


def fetch_observations() -> None:
    sat_id = get_str("sat_id_input")
    if not sat_id:
        set_status("⚠ Enter a SatNOGS satellite ID first.", (255, 200, 0, 255))
        return

    _fetch_stop.clear()
    set_status("Fetching observations from SatNOGS…", (180, 200, 255, 255))
    dpg.configure_item("fetch_btn", enabled=False)
    dpg.configure_item("stop_fetch_btn", show=True)
    dpg.configure_item("fetch_spinner", show=True)

    # Clear table and state
    if dpg.does_item_exist("obs_table"):
        for row in dpg.get_item_children("obs_table", slot=1) or []:
            dpg.delete_item(row)
    state["observations"] = []
    state["selected_obs_ids"] = set()
    _update_obs_count()

    # Parse uplink start. Used for API filters and the wait-time column.
    uplink_start_dt: datetime = parse_iso(get_str("uplink_start"))
    uplink_end_dt = uplink_start_dt + timedelta(minutes=get_float("uplink_dur"))

    next_hours = get_int("next_hours_input", 6)
    start_lt_filter = uplink_start_dt + timedelta(hours=next_hours)

    def _thread() -> None:
        try:
            all_obs: list[dict[str, Any]] = []
            state["observations"] = all_obs  # share reference so counter updates live
            stopped = False
            for page in iter_future_observation_pages(
                sat_id,
                start_gt_filter=uplink_end_dt,
                start_lt_filter=start_lt_filter,
            ):
                all_obs.extend(page)
                all_obs.sort(key=lambda o: parse_iso(o["start"]))
                if dpg.does_item_exist("obs_table"):
                    for row in dpg.get_item_children("obs_table", slot=1) or []:
                        dpg.delete_item(row)
                state["selected_obs_ids"] = set()
                _append_obs_rows(all_obs, uplink_end_dt)
                set_status(
                    f"Fetching… {len(all_obs)} so far",
                    (180, 200, 255, 255),
                )
                if _fetch_stop.is_set():
                    stopped = True
                    break

            if stopped:
                set_status(
                    f"⊘ Stopped. {len(all_obs)} observations loaded.",
                    (255, 200, 0, 255),
                )
            else:
                set_status(
                    f"✓ Loaded {len(all_obs)} future observations.",
                    (100, 255, 150, 255),
                )
        except Exception as exc:
            set_status(f"✗ Fetch error: {exc}", (255, 100, 100, 255))
        finally:
            dpg.configure_item("fetch_btn", enabled=True)
            dpg.configure_item("stop_fetch_btn", show=False)
            dpg.configure_item("fetch_spinner", show=False)

    threading.Thread(target=_thread, daemon=True).start()


def _stop_fetch() -> None:
    _fetch_stop.set()


# ─────────────────────────────────────────────────────────────
# COMMAND GENERATION
# ─────────────────────────────────────────────────────────────


def generate_agenda() -> None:
    """Core generation logic."""
    # ── Settings ──────────────────────────────────────────────
    uplink_start_str = get_str("uplink_start")
    uplink_dur_min = get_float("uplink_dur", 10.0)
    cmd_interval_sec = get_float("cmd_interval", 5.0)
    priority_interval = get_int("priority_interval", 50)
    output_path = get_str("output_path") or "command_agenda.txt"

    # Parse uplink window
    try:
        uplink_start_dt = parse_iso(uplink_start_str)
        if uplink_start_dt.tzinfo is None:
            set_status(
                "✗ 'Start of Uplink Pass' must include a timezone "
                "(e.g. 2024-05-01T12:00:00-07:00 or ...Z).",
                (255, 100, 100, 255),
            )
            return
    except Exception:
        set_status(
            "✗ Invalid 'Start of Uplink Pass'. "
            "Use ISO format with timezone: 2024-05-01T12:00:00-07:00",
            (255, 100, 100, 255),
        )
        return

    uplink_end_dt = uplink_start_dt + timedelta(minutes=uplink_dur_min)

    # ── Commands ──────────────────────────────────────────────
    loop_cmds_raw = get_str("loop_cmds_input").splitlines()
    loop_cmds = [c.strip() for c in loop_cmds_raw if c.strip()]

    priority_cmds_raw = get_str("priority_cmds_input").splitlines()
    priority_cmds = [c.strip() for c in priority_cmds_raw if c.strip()]

    if not loop_cmds:
        set_status("✗ No loop commands entered.", (255, 100, 100, 255))
        return

    # ── Filter observations ────────────────────────────────────
    selected = [
        obs
        for obs in state["observations"]
        if obs.get("id") in state["selected_obs_ids"]
    ]
    # Keep only observations that start AFTER end of uplink pass
    valid_obs = [obs for obs in selected if parse_iso(obs["start"]) > uplink_end_dt]

    if not valid_obs:
        set_status(
            "✗ No valid observations (all are before end of uplink window).",
            (255, 100, 100, 255),
        )
        return

    def _fmt(dt: datetime) -> str:
        return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Build command list ─────────────────────────────────────
    output_lines = []
    output_lines.append("# CTS-SAT-1 Command Agenda")
    output_lines.append(f"# Generated: {datetime.now(tz=UTC).isoformat()}")
    output_lines.append(
        f"# Uplink window: {_fmt(uplink_start_dt)} → {_fmt(uplink_end_dt)}"
    )
    output_lines.append(f"# Valid observations: {len(valid_obs)}")
    output_lines.append("")

    tssent_dt = uplink_start_dt
    cmd_count = 0
    cmd_interval = timedelta(seconds=cmd_interval_sec)

    # Track priority cmd tssent datetimes (de-dup key = tssent of first send)
    priority_tssent: dict[str, datetime] = {}

    for obs in valid_obs:
        obs_start_dt = parse_iso(obs["start"])
        obs_end_dt = parse_iso(obs["end"])

        output_lines.append(
            f"# ── Observation {obs['id']} | GS {obs.get('ground_station', '?')} "
            f"| {_fmt(obs_start_dt)} → {_fmt(obs_end_dt)}"
        )

        tsexec_dt = obs_start_dt
        while tsexec_dt < obs_end_dt:
            for cmd_raw in loop_cmds:
                # Priority injection?
                if (
                    priority_cmds
                    and cmd_count > 0
                    and (cmd_count % priority_interval) == 0
                ):
                    for pcmd in priority_cmds:
                        # Parse optional explicit tsexec from the line: "cmd_name()@tsexec=..."  # noqa: E501
                        p_tsexec = 0
                        p_name = pcmd
                        if "@tsexec=" in pcmd:
                            parts = pcmd.split("@tsexec=", 1)
                            p_name = parts[0].strip()
                            try:
                                p_tsexec = int(parts[1].strip())
                            except ValueError:
                                p_tsexec = 0

                        # Use same tssent every time (de-dup on satellite)
                        if pcmd not in priority_tssent:
                            priority_tssent[pcmd] = tssent_dt
                            tssent_dt += timedelta(milliseconds=100)

                        line = format_command(
                            p_name,
                            int(priority_tssent[pcmd].timestamp() * 1000),
                            p_tsexec,
                        )
                        output_lines.append(f"# PRIORITY [{cmd_count}]")
                        output_lines.append(line)

                # Regular loop command
                line = format_command(
                    cmd_raw,
                    int(tssent_dt.timestamp() * 1000),
                    int(tsexec_dt.timestamp() * 1000),
                )
                output_lines.append(line)
                tssent_dt += timedelta(milliseconds=100)
                tsexec_dt += cmd_interval
                cmd_count += 1

                if tsexec_dt >= obs_end_dt:
                    break

        output_lines.append("")

    # Append any priority commands that haven't been sent yet (first time)
    if priority_cmds and not priority_tssent:
        output_lines.append("# PRIORITY COMMANDS (standalone, none injected above)")
        for pcmd in priority_cmds:
            p_tsexec = 0
            p_name = pcmd
            if "@tsexec=" in pcmd:
                parts = pcmd.split("@tsexec=", 1)
                p_name = parts[0].strip()
                try:
                    p_tsexec = int(parts[1].strip())
                except ValueError:
                    p_tsexec = 0

            priority_tssent[pcmd] = tssent_dt
            line = format_command(p_name, int(tssent_dt.timestamp() * 1000), p_tsexec)
            output_lines.append(line)
            tssent_dt += timedelta(milliseconds=100)

    # ── Write file ─────────────────────────────────────────────
    try:
        with open(output_path, "w") as f:
            f.write("\n".join(output_lines))
        state["generated_commands"] = output_lines

        # Show preview
        dpg.set_value("preview_text", "\n".join(output_lines[:80]))
        set_status(
            f"✓ Wrote {cmd_count} commands to '{output_path}'.",
            (100, 255, 150, 255),
        )
    except Exception as exc:
        set_status(f"✗ Write error: {exc}", (255, 100, 100, 255))


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────


def build_gui() -> None:
    dpg.create_context()

    with dpg.font_registry():
        pass  # use default font

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 27, 34, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (30, 40, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 80, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (35, 45, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50, 65, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 90, 160, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 120, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 70, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 80, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 200, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 230, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (30, 50, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (40, 90, 160, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (55, 120, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (60, 80, 110, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)

    dpg.bind_theme(global_theme)

    now_local = datetime.now().astimezone().replace(microsecond=0).isoformat()

    with dpg.window(
        label="CTS-SAT-1 Command Agenda Generator",
        tag="main_window",
        width=1200,
        height=800,
        no_close=True,
    ):
        with dpg.tab_bar():
            # ══════════════════════════════════════════════════
            # TAB 1 - SETTINGS
            # ══════════════════════════════════════════════════
            with dpg.tab(label="⚙  Settings"):
                dpg.add_spacer(height=8)

                # ── Uplink window ──────────────────────────────
                with dpg.collapsing_header(
                    label="Uplink Pass Window", default_open=True
                ):
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Start of Uplink Pass (ISO with timezone):")
                        dpg.add_input_text(
                            tag="uplink_start",
                            default_value=now_local,
                            width=260,
                            hint="2024-05-01T12:00:00-07:00",
                        )
                    dpg.add_tooltip("uplink_start")
                    with dpg.tooltip("uplink_start"):
                        dpg.add_text(
                            "ISO 8601 with timezone offset. Timezone is required.\n"
                            "Examples: 2024-05-01T12:00:00-07:00  or  2024-05-01T19:00:00Z\n"
                            "This sets the tssent for the first command.\n"
                            "Only observations that START after (uplink_start + duration)\n"
                            "will be included."
                        )

                    with dpg.group(horizontal=True):
                        dpg.add_text("Uplink Pass Duration (minutes):     ")
                        dpg.add_input_float(
                            tag="uplink_dur",
                            default_value=15.0,
                            min_value=0.1,
                            max_value=60.0,
                            width=120,
                            format="%.1f",
                        )

                        dpg.add_tooltip("uplink_dur")
                        with dpg.tooltip("uplink_dur"):
                            dpg.add_text(
                                "Fine to overestimate the duration by a few minutes."
                            )

                dpg.add_spacer(height=10)

                # ── SatNOGS fetch ──────────────────────────────
                with dpg.collapsing_header(
                    label="SatNOGS Observations", default_open=True
                ):
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Satellite NORAD ID:")
                        dpg.add_input_text(
                            tag="sat_id_input",
                            default_value="69015",
                            width=120,
                            hint="e.g. 69015",
                        )
                        dpg.add_text("Fetch next")
                        dpg.add_input_int(
                            tag="next_hours_input",
                            default_value=3,
                            min_value=1,
                            max_value=720,
                            width=70,
                        )
                        dpg.add_text("hrs after uplink")
                        dpg.add_button(
                            label="Fetch Observations",
                            tag="fetch_btn",
                            callback=fetch_observations,
                        )
                        dpg.add_button(
                            label="Stop",
                            tag="stop_fetch_btn",
                            callback=_stop_fetch,
                            show=False,
                        )
                        dpg.add_loading_indicator(
                            tag="fetch_spinner",
                            show=False,
                            radius=1.5,
                            speed=1.5,
                            color=(100, 180, 255, 255),
                        )
                    dpg.add_spacer(height=6)

                    # Observations table
                    with dpg.table(
                        tag="obs_table",
                        header_row=True,
                        borders_outerH=True,
                        borders_innerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        scrollY=True,
                        height=180,
                        resizable=True,
                    ):
                        dpg.add_table_column(
                            label="✓", width_fixed=True, init_width_or_weight=30
                        )
                        dpg.add_table_column(
                            label="Obs ID", width_fixed=True, init_width_or_weight=80
                        )
                        dpg.add_table_column(
                            label="GS ID", width_fixed=True, init_width_or_weight=80
                        )
                        dpg.add_table_column(label="Start (UTC)")
                        dpg.add_table_column(label="End (UTC)")
                        dpg.add_table_column(label="Start (Local)")
                        dpg.add_table_column(label="End (Local)")
                        dpg.add_table_column(label="Wait (uplink LOS -> pass AOS)")

                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text(
                            "", tag="obs_count_text", color=(160, 170, 190, 255)
                        )
                    dpg.add_text(
                        "(All observations are selected by default. Uncheck to exclude.)",
                        color=(160, 170, 190, 255),
                    )

                dpg.add_spacer(height=10)

                # ── Timing settings ────────────────────────────
                with dpg.collapsing_header(label="Timing & Output", default_open=True):
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Command Execution Interval (seconds):")
                        dpg.add_input_float(
                            tag="cmd_interval",
                            default_value=5.0,
                            min_value=0.1,
                            max_value=600.0,
                            width=120,
                            format="%.1f",
                        )
                    dpg.add_text(
                        "  Time between consecutive tsexec values in the loop.",
                        color=(160, 170, 190, 255),
                    )

                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Priority Command Injection Interval: ")
                        dpg.add_input_int(
                            tag="priority_interval",
                            default_value=50,
                            min_value=1,
                            max_value=10000,
                            width=120,
                        )
                    dpg.add_text(
                        "  Insert priority commands every N loop commands.",
                        color=(160, 170, 190, 255),
                    )

                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Output File Path:                    ")
                        dpg.add_input_text(
                            tag="output_path",
                            default_value="command_agenda.txt",
                            width=360,
                        )

            # ══════════════════════════════════════════════════
            # TAB 2 - COMMANDS
            # ══════════════════════════════════════════════════
            with dpg.tab(label="📋  Commands"):
                dpg.add_spacer(height=8)

                with dpg.collapsing_header(
                    label="Loop Commands  (executed repeatedly during each pass)",
                    default_open=True,
                ):
                    dpg.add_spacer(height=4)
                    dpg.add_text(
                        "Enter one command per line.  Format:  command_name(arg1,arg2)\n"
                        "The tssent / tsexec tags are generated automatically.",
                        color=(160, 170, 190, 255),
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_input_text(
                        tag="loop_cmds_input",
                        multiline=True,
                        height=200,
                        width=-1,
                        default_value=(
                            "hello_world()\nrun_all_unit_tests()\nget_power_telemetry()"
                        ),
                        hint="hello_world()\nget_power_telemetry()\n...",
                    )

                dpg.add_spacer(height=10)

                with dpg.collapsing_header(
                    label="Priority Commands  (injected every N commands with the SAME tssent)",
                    default_open=True,
                ):
                    dpg.add_spacer(height=4)
                    dpg.add_text(
                        """
Enter one command per line.
Optionally append  @tsexec=<ms>  for a fixed execution time, or omit for immediate (0).
Each priority command keeps its first tssent so the satellite de-duplicates.""".strip(),
                        color=(160, 170, 190, 255),
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_input_text(
                        tag="priority_cmds_input",
                        multiline=True,
                        height=140,
                        width=-1,
                        default_value=(
                            "CTS1+config_set_int_var(TCMD_require_unique_tssent,1)!\n"
                        ),
                    )

            # ══════════════════════════════════════════════════
            # TAB 3 - GENERATE / PREVIEW
            # ══════════════════════════════════════════════════
            with dpg.tab(label="🚀  Generate"):
                dpg.add_spacer(height=8)

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="  Generate Command Agenda  ",
                        callback=generate_agenda,
                        height=36,
                    )
                    dpg.add_spacer(width=20)
                    dpg.add_text("", tag="status_text")

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_text("Preview (first 80 lines):", color=(160, 170, 190, 255))
                dpg.add_spacer(height=4)
                dpg.add_input_text(
                    tag="preview_text",
                    multiline=True,
                    readonly=True,
                    height=-1,
                    width=-1,
                    default_value="(generate agenda to see preview)",
                )

    dpg.create_viewport(
        title="CTS-SAT-1 Command Agenda Generator",
        width=1220,
        height=840,
        min_width=900,
        min_height=600,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", value=True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def main() -> None:
    load_dotenv()

    build_gui()


if __name__ == "__main__":
    main()
