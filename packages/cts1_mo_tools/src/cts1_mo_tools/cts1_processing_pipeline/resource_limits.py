"""Caps on how much of a small box this pipeline is allowed to take.

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
to several times the machine. The caps here are deliberately conservative:
this pipeline handles one satellite's packets, and finishing a requery a
minute later is far cheaper than a web UI that stalls whenever the data
refreshes.

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
    "lower_process_priority",
]

import os
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


# Threads this process gives polars (and the other rayon-backed libraries).
# One, because the process this default is aimed at is the daemon: steps
# 2-4 are seconds of work on this data volume, and a second polars thread
# buys nothing next to the web server it would be taking a core from.
#
# The web UI wants a wider cap -- someone is waiting on that work -- but
# the cap has to be applied from this package's `__init__` (see there for
# why), which runs long before anything knows which entry point it's under.
# So the split lives in the environment instead of in the code: the `web`
# service sets CTS1_POLARS_THREADS=2 in docker-compose.yml. That's the
# right place for it anyway, since it's a property of the deployment rather
# than of the web UI.
DEFAULT_POLARS_THREADS = env_int("CTS1_POLARS_THREADS", 1)

# Step 1's decoder concurrency: how many observations are decoded at once,
# each running native CPU-bound decoders (askew_demod_from_file,
# sso_rx_replay, gr_satellites x2) as subprocesses. At the old default of 4
# on a 2-vCPU box this alone was a 2x oversubscription before polars,
# DuckDB or the web server got a look in.
DEFAULT_DECODER_WORKERS = env_int("CTS1_DECODER_WORKERS", 2)

# Downloads in flight inside a single observation's satnogs_data_demod
# call -- and this pool is nested inside the decoder pool above, so the
# real ceiling is this times `DEFAULT_DECODER_WORKERS`. These are
# I/O-bound, so the cap is about sockets and buffered response bodies
# rather than CPU; the old default of 50 (200 with the nesting) was enough
# concurrent connections to be rude to SatNOGS as well as expensive here.
DEFAULT_DEMOD_DOWNLOAD_WORKERS = env_int("CTS1_DEMOD_DOWNLOAD_WORKERS", 8)

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
