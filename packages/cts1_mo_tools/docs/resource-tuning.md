# Resource tuning

How the processing pipeline is kept from taking the whole deployment box
out from under the web UI. You shouldn't need any of this for a normal
deploy -- the defaults are already set in code and in
`docker-compose.yml`. Read it when a refresh is visibly slowing the web
UI down, when you're running a backfill, or when the box changes size.

See [DEPLOY.md](../DEPLOY.md) for actually deploying.

## Why

The daemon and the web UI share one small box (2 vCPUs, 3 GB RAM + 4 GB
swap, btrfs with compression, zswap). Left at their defaults, the
thread-pool-sizing libraries in this stack each independently size
themselves to "all the cores", so the pipeline happily oversubscribes the
machine several times over while the web server is trying to serve the very
data being refreshed.

So the daemon is capped on purpose. Nothing is waiting on a requery, and
finishing one a minute later is far cheaper than a web UI that stalls every
15 minutes.

## The knobs

Every cap below is defaulted in code (see
[`resource_limits.py`](../src/cts1_mo_tools/cts1_processing_pipeline/resource_limits.py))
and overridable from `daemon.environment` / `web.environment` in
`docker-compose.yml`, where they're repeated so there's one place to change
them:

| Variable | Default | What it caps |
| --- | --- | --- |
| `CTS1_DECODER_WORKERS` | 2 | Observations decoded at once in step 1, each running native CPU-bound decoders |
| `CTS1_DEMOD_DOWNLOAD_WORKERS` | 8 | Packet downloads in flight *per observation* -- nested inside the pool above, so the real ceiling is the product |
| `CTS1_POLARS_THREADS` | 1 | polars (and the other rayon/OpenMP pools). Set per service: the `web` service raises it to 2, since someone is waiting on that work |
| `CTS1_DAEMON_NICENESS` | 10 | How far the daemon (and every decoder subprocess, which inherits it) is niced below the web server. 0 disables |
| `CTS1_DUCKDB_MEMORY_LIMIT` | `500MB` | DuckDB's working memory, which otherwise defaults to ~80% of *host* RAM |

`CTS1_POLARS_THREADS` has to be applied before polars is first imported --
polars sizes its thread pool once, at import, and ignores the variable
afterwards -- so it's applied from the `cts1_processing_pipeline` package's
`__init__.py`, which Python guarantees runs before any of its submodules.
Setting `POLARS_MAX_THREADS` directly still wins over all of this.

DuckDB's thread count is capped in code rather than by env var, at one
(`common.py`) -- it otherwise sizes that from the *host*, not the
container.

On top of those, `docker-compose.yml` caps the daemon's CPU (`cpus: 1.5`)
and gives it half the default `cpu_shares`, so `web` wins whenever both
want the CPU. `web` gets no caps at all.

## Memory

There is deliberately **no container memory limit** on either service.

A cgroup memory cap (`mem_limit` / `memswap_limit` in `docker-compose.yml`)
is a hard ceiling, not backpressure: crossing it gets the process SIGKILLed
by the kernel -- `docker compose ps` shows exit code 137 -- even when the
host has gigabytes of swap sitting free. This box has 4 GB of it, and a
heavy run that swaps and finishes slowly is much better than one that dies
and restarts. An earlier revision of this file set `mem_limit: 1600m` /
`memswap_limit: 2000m` on the daemon and caused exactly that; don't add
them back without a specific reason.

To check whether something *was* OOM-killed:

```bash
docker inspect -f '{{.Name}} OOMKilled={{.State.OOMKilled}} exit={{.State.ExitCode}}' \
  $(docker compose ps -aq)
dmesg -T | grep -i -E 'oom|killed process' | tail
```

The one memory limit left is DuckDB's `CTS1_DUCKDB_MEMORY_LIMIT` (500 MB by
default), and it can't kill anything: DuckDB spills to its temp directory
when it hits the limit, and raises `duckdb.OutOfMemoryException` only if it
can't. So if that one is too low you get an error in the log, not a dead
daemon. Raise it if you see those.

The thread and worker caps above also hold peak memory down indirectly --
fewer concurrent decoders and download buffers -- so lowering
`CTS1_DECODER_WORKERS` is usually a better first move than raising a
ceiling.

## Backfills

A backfill is the one case where these defaults are the wrong trade: it's a
long run, and with the rolling daemon stopped (as
[DEPLOY.md](../DEPLOY.md#backfill-history) requires anyway) there's nothing
for it to be polite to but the web UI. Raise them for that container only:

```bash
docker compose run --rm \
  -e CTS1_DECODER_WORKERS=4 \
  -e CTS1_POLARS_THREADS=2 \
  -e CTS1_DAEMON_NICENESS=0 \
  daemon 69015 --start "2026-05-01" --interval 15
```

## Disk

Step 1 rewrites every parquet file in full at each checkpoint, which on
btrfs-with-compression is the most disruptive thing the pipeline does to
the web UI reading those same files. It's rate-limited to one rewrite per
10 minutes (`MIN_SECONDS_BETWEEN_CHECKPOINTS`, in step 1's `pipeline.py`);
a crash loses at most that much decode work.

If I/O is still the bottleneck, the daemon's disk priority can be dropped
from the host as well -- niceness covers CPU only:

```bash
sudo ionice -c 3 -p $(docker inspect -f '{{.State.Pid}}' \
  $(docker compose ps -q daemon))
```
