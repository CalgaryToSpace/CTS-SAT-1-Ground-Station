"""File-based signalling between the daemon and the web UI.

The two run as separate containers (see ../DEPLOY.md) that share nothing
but `data_dir` -- the same directory every pipeline step already reads and
writes its parquet files in -- so the two directions of "what's going on"
both ride on small files in there rather than a socket, a queue, or a
third process:

  - Web UI -> daemon: `daemon_pipeline_trigger.json`. The web UI's "Trigger
    Pipeline" button drops this file; the daemon notices it while sleeping
    between requeries, deletes it, and starts its next run immediately
    instead of waiting out the rest of the interval.

  - Daemon -> web UI: `daemon_status.json`. The daemon rewrites this on a
    short heartbeat (see `StatusReporter`) with what it's doing right now,
    so the web UI can show a live "daemon is running..." indicator -- and,
    because the heartbeat stops when the daemon does, tell a working daemon
    apart from one that died mid-run (see `DaemonStatus.is_live`).

Both files are written atomically (temp file + `os.replace`) so a reader
racing a writer sees either the old contents or the new ones, never a
half-written file. Both are also entirely disposable: deleting either one
costs nothing more than a missing indicator until the next heartbeat.
"""

from __future__ import annotations

__all__ = [
    "DAEMON_STATUS_FILENAME",
    "HEARTBEAT_INTERVAL_SEC",
    "STATUS_STALE_AFTER",
    "TRIGGER_REQUEST_FILENAME",
    "DaemonState",
    "DaemonStatus",
    "StatusReporter",
    "TriggerRequest",
    "clear_trigger_request",
    "read_status",
    "read_trigger_request",
    "request_pipeline_run",
    "write_status",
]

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

if TYPE_CHECKING:
    from types import TracebackType

TRIGGER_REQUEST_FILENAME = "daemon_pipeline_trigger.json"
DAEMON_STATUS_FILENAME = "daemon_status.json"

# How often the daemon rewrites `daemon_status.json` while it's alive, and
# how often it checks for a trigger request while it's sleeping. Short
# enough that the web UI's indicator and the button both feel immediate;
# the file is a few hundred bytes, so the write cost is irrelevant next to
# what the pipeline itself is doing.
HEARTBEAT_INTERVAL_SEC = 5.0

# A status file older than this is treated as "not running" rather than
# believed -- covers the daemon being killed hard enough that it never got
# to write its final `DaemonState.STOPPED`. Several heartbeats wide so a
# loaded box that's merely slow doesn't flap the indicator.
STATUS_STALE_AFTER = timedelta(seconds=HEARTBEAT_INTERVAL_SEC * 6)


class DaemonState(StrEnum):
    """What the daemon is doing, as written to `daemon_status.json`.

    Serialized by value, and `read_status` maps an unrecognized value back
    to `UNKNOWN` -- a web UI running an older build than the daemon (or the
    reverse, mid-deploy) degrades to "something's running, can't say what"
    rather than blowing up on the status file.
    """

    STARTING = "starting"
    PROCESSING = "processing"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` via a temp file + rename, so a reader never
    catches a half-written file.

    The temp file sits in the same directory as `path` (`os.replace` is only
    atomic within one filesystem) and gets a unique name: the status file
    has more than one writer (the daemon's heartbeat thread and whichever
    thread called `StatusReporter.set`), and a shared fixed temp name lets
    two of them scribble over each other's half-written file and strand it
    there. It's also removed if the write itself fails, rather than being
    left behind to accumulate.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_path, path)  # noqa: PTH105
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    """Parse `path`, or None if it's missing, unreadable, or not valid JSON.

    A malformed file is a normal transient state here (a writer crashed
    mid-write on a filesystem where the rename isn't atomic, someone
    hand-edited it), and neither side has anything better to do about it
    than ignore it until it's rewritten.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring unreadable {path}: {exc}")
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Written with a UTC offset by everything here, but a hand-edited or
    # older file might be naive -- assume UTC rather than returning
    # something that raises on the first comparison against `now(UTC)`.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Web UI -> daemon: pipeline-trigger requests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerRequest:
    """A pending "run the pipeline now" request from the web UI."""

    requested_at: datetime | None
    """When the request was written. None if the file didn't carry a
    parseable timestamp -- the request still counts."""

    note: str | None = None
    """Free-text note about where the request came from, for the log line
    the daemon writes when it picks the request up."""


def request_pipeline_run(data_dir: Path, *, note: str | None = None) -> None:
    """Ask a running daemon to start its next run immediately.

    Idempotent: a second call before the daemon has picked the first one up
    just overwrites the file, so mashing the button doesn't queue up a pile
    of redundant runs.

    Does nothing about a daemon that isn't running -- the file simply sits
    there until one starts and consumes it on its first sleep. Check
    `read_status` first if you want to tell the user that up front.
    """
    _write_json_atomic(
        data_dir / TRIGGER_REQUEST_FILENAME,
        {"requested_at": datetime.now(UTC).isoformat(), "note": note},
    )


def read_trigger_request(data_dir: Path) -> TriggerRequest | None:
    """The pending trigger request, or None if there isn't one."""
    payload = _read_json(data_dir / TRIGGER_REQUEST_FILENAME)
    if payload is None:
        return None
    note = payload.get("note")
    return TriggerRequest(
        requested_at=_parse_timestamp(payload.get("requested_at")),
        note=note if isinstance(note, str) else None,
    )


def clear_trigger_request(data_dir: Path) -> None:
    """Delete the pending trigger request, if any. Safe to call when there
    isn't one.
    """
    (data_dir / TRIGGER_REQUEST_FILENAME).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Daemon -> web UI: status heartbeat.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    """The daemon's last-published state, as read from `daemon_status.json`."""

    state: DaemonState
    updated_at: datetime | None
    """When this status was written -- the heartbeat that `is_live` checks."""

    detail: str | None = None
    """One line of human-readable context for the current state, e.g. which
    step is running or when the next requery is due."""

    next_run_at: datetime | None = None
    """When the next requery is due, while sleeping. None otherwise."""

    last_run_finished_at: datetime | None = None
    """When the most recent full steps-1-through-4 run finished."""

    pid: int | None = None
    """The daemon process's PID, for matching a status file against a
    process when debugging a deployment."""

    @property
    def is_live(self) -> bool:
        """Whether a daemon appears to actually be running right now.

        False for a `STOPPED` daemon (it said so on the way out) and for a
        heartbeat that's gone stale past `STATUS_STALE_AFTER` (it didn't
        get the chance to).
        """
        if self.state is DaemonState.STOPPED:
            return False
        if self.updated_at is None:
            return False
        return datetime.now(UTC) - self.updated_at <= STATUS_STALE_AFTER

    @property
    def is_processing(self) -> bool:
        """Whether the daemon is actively running pipeline steps right now."""
        return self.is_live and self.state is DaemonState.PROCESSING


def write_status(  # noqa: PLR0913
    data_dir: Path,
    *,
    state: DaemonState,
    detail: str | None = None,
    next_run_at: datetime | None = None,
    last_run_finished_at: datetime | None = None,
    pid: int | None = None,
) -> None:
    """Publish `state` to `daemon_status.json` for the web UI to read."""
    _write_json_atomic(
        data_dir / DAEMON_STATUS_FILENAME,
        {
            "state": str(state),
            "updated_at": datetime.now(UTC).isoformat(),
            "detail": detail,
            "next_run_at": next_run_at.isoformat() if next_run_at else None,
            "last_run_finished_at": (
                last_run_finished_at.isoformat() if last_run_finished_at else None
            ),
            "pid": pid if pid is not None else os.getpid(),
        },
    )


def read_status(data_dir: Path) -> DaemonStatus | None:
    """The daemon's published status, or None if it has never written one.

    Note that a non-None result doesn't mean a daemon is running -- a status
    file outlives the process that wrote it. Check `DaemonStatus.is_live`.
    """
    payload = _read_json(data_dir / DAEMON_STATUS_FILENAME)
    if payload is None:
        return None

    try:
        state = DaemonState(payload.get("state"))
    except ValueError:
        state = DaemonState.UNKNOWN

    detail = payload.get("detail")
    pid = payload.get("pid")
    return DaemonStatus(
        state=state,
        updated_at=_parse_timestamp(payload.get("updated_at")),
        detail=detail if isinstance(detail, str) else None,
        next_run_at=_parse_timestamp(payload.get("next_run_at")),
        last_run_finished_at=_parse_timestamp(payload.get("last_run_finished_at")),
        pid=pid if isinstance(pid, int) else None,
    )


class StatusReporter:
    """Keeps `daemon_status.json` warm from a background thread.

    The daemon spends most of its time blocked inside a pipeline step, with
    no chance to write a heartbeat from the main thread, so a plain "write
    the status whenever something changes" scheme would look identical to a
    daemon that had died mid-step. A thread rewriting the current state
    every `HEARTBEAT_INTERVAL_SEC` keeps the two distinguishable.

    Use as a context manager: the exit writes a final `STOPPED` status so
    the web UI's indicator goes dark immediately on a clean shutdown,
    rather than after `STATUS_STALE_AFTER`.

        with StatusReporter(data_dir) as reporter:
            reporter.set(DaemonState.PROCESSING, detail="step 1")
            ...
    """

    def __init__(
        self, data_dir: Path, *, interval_sec: float = HEARTBEAT_INTERVAL_SEC
    ) -> None:
        self._data_dir = data_dir
        self._interval_sec = interval_sec
        # Guards the published fields below against the heartbeat thread
        # reading them mid-`set`; `_stop` doubles as the sleep between
        # beats, so `close()` doesn't wait out a full interval.
        self._lock = threading.Lock()
        # Held across the file write itself (rather than just the field
        # read), so a slow heartbeat write can't land *after* a newer
        # `set`'s write and leave the stale state on disk.
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = DaemonState.STARTING
        self._detail: str | None = None
        self._next_run_at: datetime | None = None
        self._last_run_finished_at: datetime | None = None

    def set(
        self,
        state: DaemonState,
        *,
        detail: str | None = None,
        next_run_at: datetime | None = None,
    ) -> None:
        """Publish a new state immediately, and keep beating it afterward.

        Leaving `PROCESSING` stamps `last_run_finished_at`, so the web UI
        can show when the last run ended without the daemon having to track
        that separately.
        """
        with self._lock:
            if self._state is DaemonState.PROCESSING and state is not (
                DaemonState.PROCESSING
            ):
                self._last_run_finished_at = datetime.now(UTC)
            self._state = state
            self._detail = detail
            self._next_run_at = next_run_at
        self._publish()

    def start(self) -> None:
        """Begin heartbeating. Idempotent."""
        if self._thread is not None:
            return
        self._publish()
        self._thread = threading.Thread(
            target=self._run, name="daemon-status-heartbeat", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop heartbeating and publish a final `STOPPED` status."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_sec)
            self._thread = None
        with self._lock:
            self._state = DaemonState.STOPPED
            self._detail = None
            self._next_run_at = None
        self._publish()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def _publish(self) -> None:
        """Write the current state out, swallowing any I/O error.

        A status file that can't be written (read-only mount, disk full)
        must never take the daemon down with it -- the pipeline's actual
        work matters, this indicator doesn't.
        """
        with self._write_lock:
            with self._lock:
                state = self._state
                detail = self._detail
                next_run_at = self._next_run_at
                last_run_finished_at = self._last_run_finished_at
            try:
                write_status(
                    self._data_dir,
                    state=state,
                    detail=detail,
                    next_run_at=next_run_at,
                    last_run_finished_at=last_run_finished_at,
                )
            except OSError as exc:
                logger.warning(f"Could not write daemon status file: {exc}")

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            self._publish()
