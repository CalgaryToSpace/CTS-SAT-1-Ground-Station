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
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl
from nicegui import app, run, ui

from cts1_mo_tools.cts1_spreadsheet_to_agenda import default_agenda_filename

from . import overpasses, plan_ops
from .layout import page_shell

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
_RECENT_FILES = timedelta(days=7)


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
        add = ui.button("Add command", icon="add")
        autopop = ui.button("Auto-populate…", icon="auto_fix_high")

    grid = ui.aggrid(
        {
            "columnDefs": [
                {"field": "uplink", "headerName": "Uplink pass", "editable": False},
                {
                    "field": "mode",
                    "headerName": "Mode",
                    "cellEditor": "agSelectCellEditor",
                    "cellEditorParams": {"values": ["Single", "Interval"]},
                    "width": 100,
                },
                {"field": "tsexec_start", "headerName": "tsexec Start (UTC)"},
                {"field": "tsexec_end", "headerName": "tsexec End (UTC)"},
                {"field": "interval", "headerName": "Interval", "width": 100},
                {
                    "field": "telecommand",
                    "headerName": "Telecommand",
                    "flex": 3,
                    "cellClass": "font-mono",
                },
                {"field": "resp_fname", "headerName": "resp_fname"},
                {"field": "repeat", "headerName": "Repeat", "width": 90},
                {"field": "random", "headerName": "Random", "width": 90},
                {"field": "notes", "headerName": "Notes", "flex": 2},
                {"field": "updated", "headerName": "Last edit", "editable": False},
            ],
            "defaultColDef": {"editable": True, "resizable": True, "minWidth": 80},
            "rowSelection": {"mode": "multiRow"},
            ":getRowId": "(params) => String(params.data.id)",
            "stopEditingWhenCellsLoseFocus": True,
            "rowData": [],
        }
    ).classes("w-full h-[65vh] plan-ops-grid")
    # Spreadsheet-style vertical lines between columns. Recoloring the
    # cell's own (transparent) right border keeps AG Grid's focus outline.
    ui.add_css("""
        .plan-ops-grid .ag-cell { border-right-color: var(--ag-border-color); }
        .plan-ops-grid .ag-header-cell {
            box-shadow: inset -1px 0 0 var(--ag-border-color);
        }
    """)

    with ui.row().classes("w-full items-center gap-4"):
        ui.label("Selected:")
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
        grid.options["rowData"] = [
            c
            | {
                "uplink": options.get(c["uplink_aos"] or "", "Later"),
                "updated": f"{c['updated_by']} @ {c['updated_at']}",
            }
            for c in store.list_commands()
            if view.value == _ALL or (c["uplink_aos"] or _LATER) == view.value
        ]
        grid.update()
        download.set_enabled(view.value not in {_ALL, _LATER})

    view.on_value_change(lambda: _reload(force=True))

    def _on_edit(e: GenericEventArguments) -> None:
        if (who := _who()) is not None:
            value = e.args["newValue"]
            store.update_field(
                e.args["data"]["id"],
                e.args["colId"],
                "" if value is None else str(value),
                who,
            )

    grid.on("cellValueChanged", _on_edit)

    def _selected_aos() -> str | None:
        return None if view.value in {_ALL, _LATER} else view.value

    def _add() -> None:
        if (who := _who()) is not None:
            store.add([{"uplink_aos": _selected_aos()}], who)

    add.on_click(_add)

    async def _selected_ids() -> list[int]:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Tick some rows first.")
        return [r["id"] for r in rows]

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
        return plan_ops.table_rows(grid.options["rowData"])

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


def _open_autopopulate(
    data_dir: Path,
    store: plan_ops.PlanOpsStore,
    who: Callable[[], str | None],
    uplink_aos: str | None,
    passes: dict[str, overpasses.Overpass],
) -> None:
    start, end = _plan_window(uplink_aos, passes)
    files = plan_ops.load_incomplete_files(data_dir)
    recent = datetime.now(UTC) - _RECENT_FILES

    with ui.dialog() as dialog, ui.card().classes("w-[48rem] max-w-full"):
        ui.label(
            f"Add commands to {uplink_aos + 'Z' if uplink_aos else 'Uplink later'}, "
            f"to run {start:%Y-%m-%d %H:%M}Z to {end:%Y-%m-%d %H:%M}Z"
        ).classes("font-bold")
        with ui.tabs() as tabs:
            satnogs_tab = ui.tab("Downlink during SatNOGS passes")
            photo_tab = ui.tab("Photos over dark region")
        with ui.tab_panels(tabs, value=satnogs_tab).classes("w-full") as panels:
            with ui.tab_panel(satnogs_tab):
                ui.label(
                    "Each SatNOGS coverage window (from step 12's schedule) "
                    "downlinks one target, repeatedly; targets take turns across "
                    "windows. Missing chunks go down at least "
                    f"{plan_ops.MIN_DOWNLINK_BYTES:,} B at a time."
                ).classes("text-sm")
                file_select = (
                    ui.select(
                        {
                            r[
                                "file_path"
                            ]: f"{r['file_path']} ({r['missing_bytes']:,} B missing)"
                            for r in files.iter_rows(named=True)
                        },
                        label="Missing chunks of files",
                        multiple=True,
                        value=files.filter(pl.col("last_downlinked_at") >= recent)[
                            "file_path"
                        ].to_list(),
                    )
                    .props("use-chips")
                    .classes("w-full")
                )
                extra_targets = ui.textarea(
                    "More targets, one per line: path;start_offset;length"
                ).classes("w-full")
                every = ui.input("Every", value="2 min")
            with ui.tab_panel(photo_tab):
                lat = ui.input("Latitude range", value="55 to 65")
                lon = ui.input("Longitude range (blank = any)", value="")
                photo_every = ui.number("Every (s)", value=30, min=1)
                eclipse = ui.select(
                    ["penumbra", "umbra"], value="penumbra", label="Dark from"
                )

        async def _generate() -> None:
            if (user := who()) is None:
                return
            try:
                if panels.value not in {photo_tab, photo_tab.props["name"]}:
                    targets = plan_ops.missing_chunk_targets(
                        files.filter(pl.col("file_path").is_in(file_select.value or []))
                    ) + [
                        t.strip()
                        for t in (extra_targets.value or "").splitlines()
                        if t.strip()
                    ]
                    if not targets:
                        ui.notify("Pick some files or targets.")
                        return
                    windows = plan_ops.load_satnogs_windows(data_dir, start, end)
                    rows = plan_ops.satnogs_downlink_rows(
                        windows, targets, every.value or "2 min"
                    )
                else:
                    tle = overpasses.load_latest_tle(
                        data_dir, overpasses.DEFAULT_NORAD_ID
                    )
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
            except Exception as exc:  # noqa: BLE001 -- show any failure to the operator
                ui.notify(f"Couldn't generate: {exc}", type="negative", multi_line=True)
                return
            if not rows:
                ui.notify("Nothing to add in that window.", type="warning")
                return
            store.add([r | {"uplink_aos": uplink_aos} for r in rows], user)
            ui.notify(f"Added {len(rows)} command(s).", type="positive")
            dialog.delete()

        with ui.row():
            ui.button("Cancel", on_click=dialog.delete).props("flat")
            ui.button("Add commands", on_click=_generate)
    dialog.on("hide", dialog.delete)
    dialog.open()
