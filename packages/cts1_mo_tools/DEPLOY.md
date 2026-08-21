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

## Update:

```bash
git pull
docker compose up --build -d
```

## Backup:

```bash
sudo cp -a /opt/cts1_processing_pipeline /path/to/backup/
```

## Configure

Edit NORAD ID / `--start` / `--interval`: `daemon.command` in `docker-compose.yml`.

## Caveat

`Dockerfile.daemon` doesn't install `gr_satellites` (needs GNU Radio + the
gr-satellites OOT module; only a best-effort pip install is attempted) --
that one decoder will fail until it's sorted out. `sox`,
`askew_demod_from_file`, and `sso_rx_replay` are all built/installed and
verified working; the other four decoders are complete as-is.
