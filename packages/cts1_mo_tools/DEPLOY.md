# Deployment Guide

Two containers, one shared data directory:

- **`daemon`** ([`Dockerfile.daemon`](Dockerfile.daemon)) runs `cts1_processing_pipeline daemon` (see [README.md](src/cts1_mo_tools/cts1_processing_pipeline/README.md)) -- the initial backfill, then steps 1-3 on a loop -- writing its DuckDB database and every step's parquet output into `CTS1_DATA_DIR`.
- **`web`** ([`Dockerfile.web`](Dockerfile.web)) serves the NiceGUI web UI (`cts1_data_web_ui`), reading the same directory read-only.

Both Dockerfiles live in this package directory, but are still built with the **repo root** as build context (see `docker-compose.yml` at the repo root) -- `uv sync` resolves against the whole workspace's single shared `uv.lock`, so every sibling `packages/*` member's `pyproject.toml` needs to be visible in the build context, not just this package's.

## The `CTS1_DATA_DIR` environment variable

Every default path in the pipeline (`step_1`'s DuckDB file, every later step's parquet files, and the web UI's `--parquet-path` default) derives from one place: `step_1_download_and_demodulate.pipeline.DEFAULT_DATA_DIR`, which reads the `CTS1_DATA_DIR` environment variable (falling back to `./output` if unset). Point both containers at the same directory and every step's parquet path lines up automatically, with no `--db-path`/`--parquet-path` flags to keep in sync by hand.

`--db-path`/`--parquet-path` CLI flags still exist and still win over the environment variable if you pass them explicitly.

## The shared data directory is a bind mount, not a named volume

Both containers mount a plain host directory, `/opt/cts1_processing_pipeline`, at that same path inside each container -- not a Docker-managed named volume. That keeps the data (a DuckDB file + a handful of parquet files) somewhere you can `ls`/back up/inspect directly on the host without going through `docker run -v ... alpine cp ...` gymnastics.

Create it once, before the first `docker compose up` (Docker will auto-create a missing bind-mount source as root-owned, which is fine here since both containers also run as root, but it's one less surprise to create it explicitly):

```bash
sudo mkdir -p /opt/cts1_processing_pipeline
```

## Quick start (docker compose)

Run from the **repo root**:

```bash
cp .env.sample .env   # optional: add SATNOGS_NETWORK_API_KEY for a higher API rate limit
sudo mkdir -p /opt/cts1_processing_pipeline
docker compose up --build -d
docker compose logs -f daemon   # watch the initial backfill
```

Then open the web UI at `http://localhost:8089` -- its **Pipeline Status** page is the fastest way to confirm the daemon is actually running and producing fresh data (row counts and freshness timestamps at every stage; see [README.md](src/cts1_mo_tools/cts1_processing_pipeline/README.md)).

`docker-compose.yml`'s `daemon` service command is `69015 --start "24 hours" --interval 15` -- edit those in place, or override via `docker compose run daemon <norad_id> --start "..." --interval ...`.

## Running the images by hand (no compose)

Run from the **repo root** -- both `-f` paths are relative to it, but `.` (the build context) still needs to be the repo root itself:

```bash
sudo mkdir -p /opt/cts1_processing_pipeline

docker build -f packages/cts1_mo_tools/Dockerfile.daemon -t cts1-daemon .
docker run -d --name cts1-daemon \
    -e CTS1_DATA_DIR=/opt/cts1_processing_pipeline \
    -e SATNOGS_NETWORK_API_KEY=... \
    -v /opt/cts1_processing_pipeline:/opt/cts1_processing_pipeline \
    cts1-daemon 69015 --start "24 hours" --interval 15

docker build -f packages/cts1_mo_tools/Dockerfile.web -t cts1-web .
docker run -d --name cts1-web \
    -e CTS1_DATA_DIR=/opt/cts1_processing_pipeline \
    -v /opt/cts1_processing_pipeline:/opt/cts1_processing_pipeline:ro \
    -p 8089:8089 \
    cts1-web
```

## The daemon image is not complete out of the box

Step 1 shells out to four external decoder tools (see the step's own docstring/README): `sox`, `gr_satellites`, `askew_demod_from_file`, and `sso_rx_replay`. This repo doesn't vendor, build, or document an install recipe for the latter two -- in local development they're just "on `PATH`," installed by hand outside anything `uv sync` manages. `Dockerfile.daemon` installs what's readily scriptable (`sox` via apt, a best-effort `gnuradio`/`gr-satellites` install) and leaves clearly-marked `TODO` `COPY`/`RUN` steps for the other two. Until those are filled in, the daemon container starts and runs fine, but `askew_demod_from_file`/`sso_rx_replay` will simply fail to decode anything (the other three decoders still work) -- check `decoder_runs`/the Pipeline Status page's decoder scoreboard to see which tools are actually contributing.

The `web` image has no such gap: it's pure Python, fully self-contained, and was built + run end-to-end while writing this guide.

## Backing up the data directory

It's a plain host directory -- back it up however you'd back up any other directory:

```bash
sudo cp -a /opt/cts1_processing_pipeline /path/to/backup/
```

## Updating

```bash
git pull
docker compose up --build -d
```

Both images copy the whole repo into the build context and run `uv sync --frozen`, so a rebuild always picks up the current lockfile/source without needing any manual dependency bookkeeping.
