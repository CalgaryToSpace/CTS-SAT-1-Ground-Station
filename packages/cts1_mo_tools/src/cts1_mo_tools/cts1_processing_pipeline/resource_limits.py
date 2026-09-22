"""Caps on how much of a small box this pipeline is allowed to take.

### Slop Explanation

The daemon and the web UI are deployed side by side on one small VPS (2
vCPUs, 3 GB RAM), so every refresh the daemon runs is competing for CPU,
RAM and disk with the web server serving the very data it's refreshing.
Left at their defaults, the thread-pool-sizing libraries in this stack
each independently size themselves to "all the cores" and the pipeline
happily saturates the box:

* polars (and every other Rust/rayon-backed library here) spins up one
  worker thread per core, per process.
* DuckDB does the same (see `common.DEFAULT_DUCKDB_THREADS`).
* Step 1's decoder pool runs `workers` native decoders at once, each a
  CPU-bound subprocess (see `DEFAULT_DECODER_WORKERS`).

None of these are individually unreasonable; stacked on 2 vCPUs they add up
to several times the machine. So the defaults here take *half* the cores
this process is allowed to run on, rather than all of them: on the
deployment box that works out to the same conservative numbers it was
hand-tuned to (1 polars thread, 2 decoder workers), while a 16-core
development machine running the same daemon gets 8 of each and finishes a
backfill in the time it used to.

Half rather than all, everywhere, because nothing here is ever the only
thing running: on the deployment box it shares with the web server, and on
a development box with whatever the person is actually doing. Finishing a
requery a minute later is far cheaper than a box that stalls whenever the
data refreshes.

The operator-facing version of all this -- what to turn when, and the
host-side knobs that aren't in Python at all -- is in
`cts1_mo_tools/docs/resource-tuning.md`.

Every cap is overridable by the environment -- the constants below read
`CTS1_*` env vars, and `apply_thread_limits` only ever uses `setdefault`,
so an operator who *wants* the box saturated (a one-off backfill, say, with
no one watching the web UI) can set `POLARS_MAX_THREADS` and friends in
`docker-compose.yml` and have them stick.

Nothing in this module may import polars, duckdb, or anything else that
builds a thread pool at import time: `apply_thread_limits` has to run
*before* the first such import to have any effect, and it is called from
this package's `__init__` precisely so that it does.
"""

from __future__ import annotations

__all__ = [
    "DAEMON_NICENESS",
    "DEFAULT_DECODER_WORKERS",
    "DEFAULT_DEMOD_DOWNLOAD_WORKERS",
    "DEFAULT_POLARS_THREADS",
    "THREAD_LIMIT_ENV_VARS",
    "apply_thread_limits",
    "env_int",
    "half_the_cores",
    "lower_process_priority",
    "usable_cpu_count",
    "usable_memory_bytes",
]

import os
import pathlib
import sys

from loguru import logger


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive int from the environment, falling back to `default`.

    A missing, unparseable or out-of-range value logs and falls back rather
    than raising: a typo'd tuning knob in `docker-compose.yml` shouldn't
    stop the daemon from starting.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using {default}")
        return default
    if value < minimum:
        logger.warning(f"{name}={value} is below {minimum}; using {default}")
        return default
    return value


def usable_cpu_count() -> int:
    """How many cores this process is actually allowed to run on.

    `sched_getaffinity` rather than `cpu_count` so a taskset/cpuset pinning
    is respected. Note that neither sees Docker's `cpus:` quota, which is a
    CFS bandwidth limit rather than a mask -- a container capped at 1.5 CPUs
    on a 2-core host still reports 2 here, which is what we want: the caps
    below want the size of the box, and the quota then does its own
    throttling on top.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def half_the_cores(*, minimum: int = 1) -> int:
    """Half this process's cores, never below `minimum`.

    The sizing rule for every CPU-bound default in this module -- see the
    module docstring for why half.
    """
    return max(minimum, usable_cpu_count() // 2)


def usable_memory_bytes() -> int | None:
    """Total memory this process may use, or None if it can't be determined.

    Its cgroup's limit where there is one, so a container with a `mem_limit`
    sizes itself to that rather than to the host it happens to be on, and
    the host's total otherwise. Only used to size a soft cap (DuckDB's
    working memory, see `common`), so None is a fine answer -- the caller
    falls back to a fixed conservative value.
    """
    for path, unlimited in (
        ("/sys/fs/cgroup/memory.max", "max"),  # cgroup v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None),  # cgroup v1
    ):
        try:
            raw = pathlib.Path(path).read_text().strip()
        except OSError:
            continue
        if raw == unlimited:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 reports "no limit" as a number so large it's meaningless
        # (PAGE_SIZE * 2**63-ish); treat anything past a petabyte as unset.
        if 0 < value < 1 << 50:
            return value

    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


# Threads this process gives polars (and the other rayon-backed libraries).
DEFAULT_POLARS_THREADS = env_int("CTS1_POLARS_THREADS", half_the_cores())

# Step 1's decoder concurrency: how many observations are decoded at once,
# each running native CPU-bound decoders (askew_demod_from_file,
# sso_rx_replay, gr_satellites x2) as subprocesses, one decoder at a time.
DEFAULT_DECODER_WORKERS = env_int("CTS1_DECODER_WORKERS", half_the_cores(minimum=2))

# Downloads in flight inside a single observation's satnogs_client_live_data
# call. Cheap, tiny downloads from a CDN. Note: These are per-worker.
DEFAULT_DEMOD_DOWNLOAD_WORKERS = env_int("CTS1_DEMOD_DOWNLOAD_WORKERS", 50)

# How much nicer than normal the daemon runs -- see
# `lower_process_priority`.
DAEMON_NICENESS = env_int("CTS1_DAEMON_NICENESS", 10, minimum=0)

# The thread-count env vars every library in this stack reads. polars reads
# POLARS_MAX_THREADS; the rest are the conventional names for the thread
# pools underneath it and the native libraries the decoders link against,
# set together so nothing sizes itself to the host's core count behind our
# back.
THREAD_LIMIT_ENV_VARS = (
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_thread_limits(threads: int) -> None:
    """Cap this process's library thread pools at `threads`, via the environment.

    Must be called before polars is first imported: polars reads
    `POLARS_MAX_THREADS` once, when its thread pool is built at import
    time, and ignores the variable thereafter. The one caller is this
    package's `__init__`, which Python runs before any of its submodules --
    see there.

    Uses `setdefault`, so a value already set in the environment (by
    `docker-compose.yml`, say) wins.
    """
    for name in THREAD_LIMIT_ENV_VARS:
        os.environ.setdefault(name, str(threads))

    if "polars" in sys.modules:
        logger.warning(
            "apply_thread_limits() ran after polars was already imported; "
            "POLARS_MAX_THREADS has no effect now and polars is still sized "
            "to the host's core count. Something imported polars before "
            "this package's __init__ ran."
        )


def lower_process_priority(niceness: int = DAEMON_NICENESS) -> None:
    """Renice this process (and so every decoder subprocess it spawns) down.

    This is the cheap half of keeping the web UI responsive: the caps above
    stop the daemon from *sizing* itself to the whole box, and this stops
    the CPU it does use from being taken at the web server's expense. A
    niced daemon still gets every idle cycle -- it just loses the CPU the
    moment a request comes in, which is exactly the trade wanted here,
    since nobody is waiting on a requery.

    Niceness is inherited across `fork`/`exec`, so renicing the daemon
    process once here covers `sox`, `gr_satellites`, `sso_rx_replay` and
    every other child without having to wrap each spawn site.

    A no-op where it isn't permitted or supported (raising the niceness of
    an already-niced process needs privileges); the daemon runs fine
    either way, so a failure is logged and swallowed.
    """
    if niceness == 0:
        return
    try:
        new_niceness = os.nice(niceness)
    except OSError as exc:
        logger.debug(f"Could not lower process priority by {niceness}: {exc}")
    else:
        logger.debug(f"Process priority lowered to niceness {new_niceness}")
