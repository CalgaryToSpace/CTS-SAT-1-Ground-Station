"""Subprocess tracking so a Ctrl+C can kill in-flight children promptly.

The pipeline shells out to sox/sso_rx_replay/gr_satellites from worker
threads. `subprocess.run()` blocks that thread until the child exits, and a
Python thread can't be interrupted from the outside (only the main thread
gets `KeyboardInterrupt`) — so on SIGINT, the only way to unblock those
worker threads quickly is to kill the children they're waiting on. Every
call here registers its child in a shared, lock-protected set so
`terminate_all()` can do exactly that.
"""

from __future__ import annotations

__all__ = ["run_tracked", "terminate_all"]

import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_lock = threading.Lock()
_active: set[subprocess.Popen[Any]] = set()


def run_tracked(
    cmd: Sequence[str], *, check: bool = False, text: bool = False
) -> subprocess.CompletedProcess[Any]:
    """Like `subprocess.run(capture_output=True)`, but killable via terminate_all()."""
    proc = subprocess.Popen(  # noqa: S603
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text
    )
    with _lock:
        _active.add(proc)
    try:
        stdout, stderr = proc.communicate()
    finally:
        with _lock:
            _active.discard(proc)

    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)
    return result


def terminate_all(*, kill_after: float = 3.0) -> None:
    """Terminate every tracked in-flight child; escalate to SIGKILL after a grace period."""
    with _lock:
        procs = list(_active)
    if not procs:
        return

    for proc in procs:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + kill_after
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
