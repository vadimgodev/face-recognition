# Production Deployment

## Prerequisites

- A server with Docker 20.10+ and Docker Compose 2.0+ (4 GB RAM minimum, 8 GB recommended)
- A domain name pointing to the server
- Ports 80 and 443 open

## Configure

```bash
git clone https://github.com/vadimgodev/face-recognition.git /opt/faceguard
cd /opt/faceguard
cp .env.example .env
```

Production essentials in `.env`:

```bash
APP_ENV=production
DEBUG=false

DOMAIN=faces.example.com                 # your domain — Traefik routes and issues TLS for it
ACME_EMAIL=admin@example.com             # Let's Encrypt registration

BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=$(openssl rand -base64 24)
SECRET_KEY=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 24)
REDIS_PASSWORD=$(openssl rand -base64 24)

ALLOWED_ORIGINS=https://faces.example.com

RECOGNITION_MODE=local                   # or hybrid with AWS credentials for CompareFaces verification
LIVENESS_ENABLED=true                    # recommended for access control
SIMILARITY_THRESHOLD=0.7                 # stricter than the 0.6 dev default
```

Start and verify:

```bash
docker compose up -d
docker compose logs -f app        # wait for model load on first start
curl -u admin:$BASIC_AUTH_PASSWORD https://faces.example.com/health
```

Traefik terminates TLS with automatic Let's Encrypt certificates (TLS-ALPN challenge, state in `letsencrypt/acme.json`) and redirects HTTP to HTTPS.

## Security checklist

- [ ] All five credentials above generated randomly (never keep `.env.example` values)
- [ ] `DEBUG=false`, `APP_ENV=production`
- [ ] `ALLOWED_ORIGINS` restricted to your domain
- [ ] Firewall allows only 22/80/443 — Postgres and Redis are not published to the host and must stay that way
- [ ] Traefik dashboard not reachable from the internet
- [ ] `LIVENESS_ENABLED=true` if recognition controls physical access
- [ ] Log rotation configured for access logs; monitoring in place
- [ ] Backups tested (below)
- [ ] Data-retention and consent policy documented (see [SECURITY.md](../SECURITY.md))

## GPU

The Docker image installs `onnxruntime-gpu` on amd64. To use an NVIDIA GPU:

1. Install the NVIDIA container toolkit on the host.
2. Uncomment the `deploy.resources` block under `app` in `docker-compose.yaml`.
3. Set `INSIGHTFACE_CTX_ID=0` (and optionally `LIVENESS_DEVICE_ID=0`) in `.env`.

CPU-only works fine for a single camera at 2 FPS (~300 ms per frame on a modern x86 CPU).

## Webcam daemon on a door controller

For real door access, run the daemon near the camera (e.g. a small PC or Raspberry Pi at the entrance) pointing at your server:

```bash
WEBCAM_API_URL=https://faces.example.com \
BASIC_AUTH_USERNAME=admin BASIC_AUTH_PASSWORD=... SECRET_KEY=... \
python webcam_daemon.py --camera 0 --mode daemon
```

On Linux you can instead run it in Docker with camera passthrough:

```bash
docker compose -f docker-compose.yaml -f docker-compose.webcam.yaml up -d
```

Set `TRIGGER_PROVIDER=webhook` with your controller's `TRIGGER_WEBHOOK_URL`, or `gpio` on a Raspberry Pi (relay on pin 17).

## Backups

```bash
# Database (embeddings + metadata)
docker compose exec postgres pg_dump -U postgres facedb | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore
gunzip < backup_20260723.sql.gz | docker compose exec -T postgres psql -U postgres facedb

# Images
tar -czf images_$(date +%Y%m%d).tar.gz data/images/
```

A cron-friendly script that keeps 7 days of both plus your `.env`:

```bash
#!/bin/bash
set -euo pipefail
BACKUP_DIR=/backups/faceguard
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
docker compose exec postgres pg_dump -U postgres facedb | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"
tar -czf "$BACKUP_DIR/images_$DATE.tar.gz" data/images/
cp .env "$BACKUP_DIR/env_$DATE"
find "$BACKUP_DIR" -mtime +7 -delete
```

## Updating

```bash
cd /opt/faceguard
git pull
docker compose build app
docker compose up -d
docker compose exec app alembic upgrade head
```
