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
