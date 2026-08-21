# Deployment Guide

## Deploy

Run from the repo root, on the target machine:

```bash
sudo mkdir -p /opt/cts1_processing_pipeline
cp .env.sample .env
nano .env   # SATNOGS_NETWORK_API_KEY, optional
docker compose up --build -d
docker compose logs -f daemon
```

Web UI: `http://<host>:8089` -- check the **Pipeline Status** page for freshness.

## Update

```bash
git pull
docker compose up --build -d
```

## Backup

```bash
sudo cp -a /opt/cts1_processing_pipeline /path/to/backup/
```

## Configure

Edit NORAD ID / `--start` / `--interval`: `daemon.command` in `docker-compose.yml`.

## Backfill history

The running `daemon` service only backfills `--start` (24 hours by default)
on startup, then rolls forward. To pull in more history -- e.g. the last 3
months -- run a one-off container with an earlier `--start`, sharing the
same data volume. DuckDB only allows one writer at a time, so stop the
regular `daemon` service first:

```bash
docker compose stop daemon

# One-off backfill container -- runs in the foreground so you can watch it.
docker compose run --rm daemon 69015 --start "2026-05-01" --interval 15

# Once complete, resume the normal rolling daemon.
docker compose start daemon
```
