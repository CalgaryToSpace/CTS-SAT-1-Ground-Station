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

## HTTPS + per-IP rate limiting (nginx)

`docker-compose.yml` fronts the `web` service with an `nginx` container that
does SSL termination and a basic per-IP request rate limit, plus a `certbot` container that keeps the
Let's Encrypt cert renewed. `web` now only publishes to `127.0.0.1:8089`
(for local debugging). `nginx` (80/443) is the only port reachable from
outside.

One-time setup, on the target machine, with a domain already pointed at it:

```bash
sudo mkdir -p /opt/cts1_ssl/letsencrypt /opt/cts1_ssl/webroot

# Issue the initial cert (port 80 must be free -- stop nginx/anything else
# using it first if this isn't a fresh box).
docker run --rm -p 80:80 \
  -v /opt/cts1_ssl/letsencrypt:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d frontiersat.mooo.com --agree-tos -m you@example.com

docker compose up --build -d
```

The `certbot` service renews (via the HTTP-01 webroot challenge, served
through `nginx`) automatically every 12 hours once a cert is due. nginx
doesn't reload on its own after a renewal, so pick up renewed certs with an
occasional reload -- e.g. a host cron entry:

```
0 3 * * * cd /path/to/repo && docker compose exec nginx nginx -s reload
```

## Update

```bash
git pull
docker compose build
docker compose up -d
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
