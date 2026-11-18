"""The Plan page: the shared, password-gated plan of telecommands,
each assigned to an RAO uplink pass (or to "later"), editable by everyone
signed in at once -- every open page reloads when anyone changes it -- and
downloadable per pass as an agenda file. See `plan_ops` for the data layer.
"""

# pyright: standard
# NiceGUI doesn't support pyright strict very well.

from __future__ import annotations

__all__ = ["PLAN_OPS_PATH", "build_plan_ops_page"]

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from nicegui import app, run, ui

from cts1_mo_tools.cts1_spreadsheet_to_agenda import default_agenda_filename

from . import overpasses, plan_ops
from .layout import page_shell
from .plan_sheet import PlanSheet

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nicegui.events import GenericEventArguments

PLAN_OPS_PATH = "/plan"

_USER_KEY = "plan_ops_user"
_ALL = "all"
_LATER = "later"
_POLL_SEC = 2.0
_PASSES_BACK = timedelta(hours=12)
_PASSES_AHEAD = timedelta(days=3)
_PLAN_HORIZON = timedelta(days=1)  # agendas only reach 24 h past their AOS


def _signed_in_user(users_file: Path) -> str | None:
    """This browser's user, if still in the users file."""
    user = app.storage.user.get(_USER_KEY)
    return user if user in plan_ops.load_users(users_file) else None


def build_plan_ops_page(data_dir: Path, users_file: Path) -> None:
    with page_shell():
        user = _signed_in_user(users_file)
        if user is None:
            _build_login(users_file)
        else:
            _build_planner(data_dir, users_file, user)


def _build_login(users_file: Path) -> None:
    with ui.card().classes("mx-auto w-80"):
        ui.label("Plan sign-in").classes("text-lg font-bold")
        name = ui.input("Name").props("autocomplete=username").classes("w-full")
        password = (
            ui.input("Password", password=True)
            .props("autocomplete=current-password")
            .classes("w-full")
        )

        async def _submit() -> None:
            name_, password_ = name.value or "", password.value or ""
            if plan_ops.check_login(users_file, name_, password_):
                app.storage.user[_USER_KEY] = name_
                ui.navigate.reload()
                return
            await asyncio.sleep(1)  # Slow down password guessing.
            ui.notify("Wrong name or password.", type="negative")

        password.on("keydown.enter", _submit)
        ui.button("Sign in", on_click=_submit).classes("w-full")
        if not plan_ops.load_users(users_file):
            ui.label("No users are configured on the server yet.").classes(
                "text-sm text-negative"
            )


_SHEET_COLUMNS: list[dict[str, Any]] = [
    {"name": "id", "type": "hidden"},
    {"name": "uplink", "title": "Uplink pass", "width": 290, "readOnly": True},
    {
        "name": "mode",
        "title": "Mode",
        "width": 90,
        "type": "dropdown",
        "source": ["Single", "Interval"],
    },
    {"name": "tsexec_start", "title": "tsexec Start (UTC)", "width": 170},
    {"name": "tsexec_end", "title": "tsexec End (UTC)", "width": 170},
    {"name": "interval", "title": "Interval", "width": 80},
    {"name": "telecommand", "title": "Telecommand", "width": 560},
    {"name": "resp_fname", "title": "resp_fname", "width": 140},
    {"name": "repeat", "title": "Repeat", "width": 70},
    {"name": "random", "title": "Random", "width": 70},
    {"name": "notes", "title": "Notes", "width": 300},
    {"name": "updated", "title": "Last edit", "width": 220, "readOnly": True},
]
"""The sheet's columns (Jspreadsheet column options), by command field."""


def _pass_label(p: overpasses.Overpass) -> str:
    local = p.aos.astimezone(ZoneInfo(overpasses.RAO.timezone))
    return (
        f"{p.aos:%a %Y-%m-%d %H:%M}Z ({local:%H:%M} local), "
        f"max {p.max_elevation_deg:.0f}°"
    )


def _pass_key(p: overpasses.Overpass) -> str:
    # Whole minutes, so a fresher TLE nudging the AOS by seconds keeps the key.
    return plan_ops.format_time(p.aos.replace(second=0, microsecond=0))


def _build_planner(  # noqa: C901, PLR0915
    data_dir: Path, users_file: Path, user: str
) -> None:
    store = plan_ops.PlanOpsStore(data_dir / plan_ops.DB_FILENAME)
    passes: dict[str, overpasses.Overpass] = {}
    seen_revision = -1

    def _who() -> str | None:
        """The user, if still allowed to edit."""
        if _signed_in_user(users_file) is None:
            ui.notify("You've been signed out -- reload the page.", type="negative")
            return None
        return user

    with ui.row().classes("w-full items-center"):
        ui.label("Plan").classes("text-2xl font-bold")
        ui.space()
        ui.label(f"Signed in as {user}")

        def _sign_out() -> None:
            app.storage.user.pop(_USER_KEY, None)
            ui.navigate.reload()

        ui.button("Sign out", on_click=_sign_out).props("flat")

    with ui.row().classes("w-full items-center gap-4"):
        view = ui.select({_ALL: "All commands"}, label="Show", value=_ALL).classes(
            "w-96"
        )
        download = ui.button("Download agenda", icon="download")
        copy = ui.button("Copy table", icon="content_copy").props("outline")
        excel = ui.button("Download Excel", icon="grid_on").props("outline")
        ui.space()
        autopop = ui.button("Auto-populate…", icon="auto_fix_high")

    sheet = PlanSheet(_SHEET_COLUMNS)
    shown: list[dict[str, Any]] = []  # The commands in the sheet, in order.

    with ui.row().classes("w-full items-center gap-4"):
        ui.label("Rows with a selected cell:")
        move_to = ui.select(
            {_LATER: "Uplink later"}, label="Move to", value=_LATER
        ).classes("w-96")
        move = ui.button("Move", icon="move_down")
        delete = ui.button("Delete", icon="delete", color="negative")

    def _options() -> dict[str, str]:
        options = {key: _pass_label(p) for key, p in passes.items()}
        for c in store.list_commands():  # Passes too old (or re-predicted).
            if c["uplink_aos"] and c["uplink_aos"] not in options:
                options[c["uplink_aos"]] = f"{c['uplink_aos']}Z"
        return dict(sorted(options.items()))

    def _reload(*, force: bool = False) -> None:
        nonlocal seen_revision
        revision = store.revision()
        if revision == seen_revision and not force:
            return
        seen_revision = revision
        options = _options()
        view.set_options({_ALL: "All commands", _LATER: "Uplink later"} | options)
        move_to.set_options({_LATER: "Uplink later"} | options)
        shown[:] = [
            c
            | {
                "uplink": options.get(c["uplink_aos"] or "", "Later"),
                "updated": f"{c['updated_by']} @ {c['updated_at']}",
            }
            for c in store.list_commands()
            if view.value == _ALL or (c["uplink_aos"] or _LATER) == view.value
        ]
        sheet.set_rows([[c[col["name"]] for col in _SHEET_COLUMNS] for c in shown])
        download.set_enabled(view.value not in {_ALL, _LATER})

    view.on_value_change(lambda: _reload(force=True))

    def _selected_aos() -> str | None:
        return None if view.value in {_ALL, _LATER} else view.value

    def _on_cells_changed(e: GenericEventArguments) -> None:
        """A typed, pasted or filled edit -- new rows (no id) become commands."""
        if (who := _who()) is None:
            return
        new_rows: dict[int, dict[str, Any]] = {}
        updates: list[tuple[int, str, str]] = []
        for c in e.args:
            if c["id"] is None:
                new_rows.setdefault(c["row"], {})[c["field"]] = c["value"]
            else:
                updates.append((int(c["id"]), c["field"], c["value"]))
        try:
            if updates:
                store.update_fields(updates, who)
            if new_rows:
                store.add(
                    [
                        # Unpasted columns get defaults; read-only ones are ignored.
                        {k: v for k, v in r.items() if k in plan_ops.EDITABLE_FIELDS}
                        | {"uplink_aos": _selected_aos()}
                        for _, r in sorted(new_rows.items())
                    ],
                    who,
                )
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
        _reload(force=True)  # Also undoes edits to read-only columns.

    sheet.on("cells_changed", _on_cells_changed)

    async def _selected_ids() -> list[int]:
        ids = await sheet.selected_ids()
        if not ids:
            ui.notify("Select a cell in each row first.")
        return ids

    async def _move() -> None:
        if (ids := await _selected_ids()) and (who := _who()) is not None:
            store.move(ids, None if move_to.value == _LATER else move_to.value, who)

    move.on_click(_move)

    async def _delete() -> None:
        if not (ids := await _selected_ids()) or _who() is None:
            return
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete {len(ids)} command(s) for everyone?")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Delete", color="negative", on_click=lambda: dialog.submit("yes")
                )
        confirmed = await dialog == "yes"
        dialog.delete()
        if confirmed:
            store.delete(ids)

    delete.on_click(_delete)

    def _download() -> None:
        aos = view.value
        commands = [c for c in store.list_commands() if c["uplink_aos"] == aos]
        start = plan_ops.parse_stored_time(aos)
        try:
            text = plan_ops.build_pass_agenda(commands, start)
        except ValueError as exc:
            ui.notify(str(exc), type="negative", multi_line=True)
            return
        ui.download.content(text, default_agenda_filename(start))

    download.on_click(_download)

    def _table() -> list[list[str]]:
        return plan_ops.table_rows(shown)

    def _copy() -> None:
        ui.clipboard.write(plan_ops.to_tsv(_table()))
        ui.notify("Copied -- paste into Excel or Google Sheets.")

    copy.on_click(_copy)

    def _excel() -> None:
        name = "all" if view.value == _ALL else view.value.replace(":", "")
        ui.download.content(
            plan_ops.to_xlsx(_table()),
            f"plan_ops_{name.replace(' ', '_')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    excel.on_click(_excel)

    autopop.on_click(
        lambda: _open_autopopulate(data_dir, store, _who, _selected_aos(), passes)
    )

    async def _load_passes() -> None:
        now = datetime.now(UTC)
        try:
            tle = overpasses.load_latest_tle(data_dir, overpasses.DEFAULT_NORAD_ID)
            found = await run.io_bound(
                overpasses.compute_overpasses,
                tle,
                overpasses.RAO,
                now - _PASSES_BACK,
                now + _PASSES_AHEAD,
            )
        except LookupError as exc:
            ui.notify(f"No pass predictions: {exc}", type="warning")
            found = []
        passes.update({_pass_key(p): p for p in found or []})
        _reload(force=True)
        # Default to planning the next pass (setting it reloads the sheet).
        upcoming = [key for key, p in sorted(passes.items()) if p.aos > now]
        if upcoming and view.value == _ALL:
            view.value = upcoming[0]

    ui.timer(0, _load_passes, once=True)
    ui.timer(_POLL_SEC, _reload)


def _plan_window(
    uplink_aos: str | None, passes: dict[str, overpasses.Overpass]
) -> tuple[datetime, datetime]:
    """When auto-populated commands may run: after the uplink pass, and
    within its agenda's 24 h (or the next 24 h, for "later").
    """
    if uplink_aos is None:
        start = datetime.now(UTC)
        return start, start + _PLAN_HORIZON
    aos = plan_ops.parse_stored_time(uplink_aos)
    p = passes.get(uplink_aos)
    return (p.los if p else aos), aos + _PLAN_HORIZON


_MAX_RANGE = timedelta(days=7)
"""Longest auto-populate time range -- the photo search samples every second."""


def _time_range(start_text: str, end_text: str) -> tuple[datetime, datetime]:
    """The auto-populate dialog's From/To, as UTC datetimes.

    Raises:
        ValueError: If either isn't a date and time, or they're out of order
            or over `_MAX_RANGE` apart.
    """
    start, end = (datetime.fromisoformat(t.strip()) for t in (start_text, end_text))
    # Naive means UTC; a typed offset is converted.
    start, end = (
        (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC) for t in (start, end)
    )
    if not start < end <= start + _MAX_RANGE:
        msg = f"'To' must be after 'From', by at most {_MAX_RANGE.days} days."
        raise ValueError(msg)
    return start, end


def _open_autopopulate(  # noqa: C901, PLR0915
    data_dir: Path,
    store: plan_ops.PlanOpsStore,
    who: Callable[[], str | None],
    uplink_aos: str | None,
    passes: dict[str, overpasses.Overpass],
) -> None:
    start, end = _plan_window(uplink_aos, passes)
    files = plan_ops.load_incomplete_files(data_dir)
    generation = 0  # Of the row count shown, so a stale count can't land last.

    with ui.dialog() as dialog, ui.card().classes("w-[48rem] max-w-full"):
        ui.label(
            f"Add commands to {uplink_aos + 'Z' if uplink_aos else 'Uplink later'}"
        ).classes("font-bold")
        # Defaults to after the pass, within its agenda's reach.
        with ui.row().classes("w-full gap-4 no-wrap"):
            range_from = ui.input(
                "From (UTC)", value=f"{start:%Y-%m-%d %H:%M}"
            ).classes("flex-1")
            range_to = ui.input("To (UTC)", value=f"{end:%Y-%m-%d %H:%M}").classes(
                "flex-1"
            )
        with ui.tabs() as tabs:
            windows_tab = ui.tab("SatNOGS windows")
            downlink_tab = ui.tab("Downlink a missing chunk")
            photo_tab = ui.tab("Photos over dark region")
        with ui.tab_panels(tabs, value=windows_tab).classes("w-full") as panels:
            with ui.tab_panel(windows_tab):
                ui.label(
                    "An Interval row per SatNOGS window (from step 12's "
                    "schedule) in the time range, with only its times filled in."
                ).classes("text-sm")
            with ui.tab_panel(downlink_tab):
                ui.label(
                    "Downlinks one chunk over and over during every SatNOGS "
                    "window in the time range, every chunk length / "
                    f"{plan_ops.DOWNLINK_BYTES_PER_SEC} B/s."
                ).classes("text-sm")
                chunk_bytes = ui.number(
                    "Chunk size (B)",
                    value=plan_ops.DEFAULT_CHUNK_BYTES,
                    min=1,
                    step=1000,
                    precision=0,
                )
                chunk = ui.select({}, label="Missing chunk").classes("w-full")
            with ui.tab_panel(photo_tab):
                ui.label("Each capture in the time range with its own setup.").classes(
                    "text-sm"
                )
                lat = ui.input("Latitude range", value="55 to 65").classes("w-full")
                lon = ui.input("Longitude range (blank = any)", value="").classes(
                    "w-full"
                )
                photo_every = ui.number("Every (s)", value=30, min=1).classes("w-full")
                eclipse = ui.select(
                    ["penumbra", "umbra"], value="penumbra", label="Dark from"
                ).classes("w-full")
        count = ui.label().classes("text-sm")

        def _on(tab: ui.tab) -> bool:
            return panels.value in {tab, tab.props["name"]}

        def _refresh_chunks() -> None:
            """The chunk options for the chunk size, most recent file first."""
            try:
                targets = plan_ops.missing_chunk_targets(
                    files, int(chunk_bytes.value or 0)
                )
            except ValueError:
                targets = []
            options = {}
            for t in targets:
                path, offset, length = t.rsplit(";", 2)
                options[t] = f"{path}, bytes {int(offset):,} + {int(length):,}"
            chunk.set_options(options)
            if chunk.value not in options:
                chunk.value = next(iter(options), None)

        async def _build() -> list[dict[str, Any]]:
            """The rows Populate adds, for the current tab and inputs."""
            start, end = _time_range(range_from.value or "", range_to.value or "")
            if _on(windows_tab):
                return plan_ops.satnogs_window_rows(
                    plan_ops.load_satnogs_windows(data_dir, start, end)
                )
            if _on(downlink_tab):
                if not chunk.value:
                    return []
                return plan_ops.satnogs_downlink_rows(
                    plan_ops.load_satnogs_windows(data_dir, start, end), chunk.value
                )
            tle = overpasses.load_latest_tle(data_dir, overpasses.DEFAULT_NORAD_ID)
            rows = await run.io_bound(
                plan_ops.dark_region_photo_rows,
                tle.to_satkit(),
                start,
                end,
                latitude_range=lat.value or None,
                longitude_range=lon.value or None,
                every_sec=float(photo_every.value or 30),
                eclipse_start=eclipse.value,
            )
            return rows or []

        async def _count() -> None:
            nonlocal generation
            generation += 1
            mine = generation
            await asyncio.sleep(0.3)  # Wait out a burst of keystrokes.
            if mine != generation:
                return
            count.text = "Counting rows..."
            try:
                text = f"Populate will add {len(await _build())} row(s)."
            except Exception as exc:  # noqa: BLE001 -- show any failure to the operator
                text = f"Can't populate: {exc}"
            if mine == generation:
                count.text = text

        async def _populate() -> None:
            if (user := who()) is None:
                return
            try:
                rows = await _build()  # Afresh: the count shown may be stale.
            except Exception as exc:  # noqa: BLE001 -- show any failure to the operator
                ui.notify(f"Couldn't populate: {exc}", type="negative", multi_line=True)
                return
            if not rows:
                ui.notify("Nothing to add.", type="warning")
                return
            store.add([r | {"uplink_aos": uplink_aos} for r in rows], user)
            ui.notify(f"Added {len(rows)} row(s).", type="positive")
            dialog.delete()

        _refresh_chunks()
        chunk_bytes.on_value_change(_refresh_chunks)
        for element in (
            panels,
            range_from,
            range_to,
            chunk_bytes,
            chunk,
            lat,
            lon,
            photo_every,
            eclipse,
        ):
            element.on_value_change(_count)
        ui.timer(0, _count, once=True)

        with ui.row():
            ui.button("Cancel", on_click=dialog.delete).props("flat")
            ui.button("Populate", on_click=_populate)
    dialog.on("hide", dialog.delete)
    dialog.open()
