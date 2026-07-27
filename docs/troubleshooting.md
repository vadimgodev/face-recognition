# Troubleshooting

## Authentication

**Login prompt loops / 401 on every page**
- Verify `BASIC_AUTH_USERNAME`/`BASIC_AUTH_PASSWORD` are set in `.env` and containers were recreated after changes: `docker compose up -d --force-recreate web traefik`.
- Some browsers refuse to cache Basic Auth credentials for `localhost` or in private windows. If prompts keep reappearing, map a dev domain: add `127.0.0.1 face.test` to `/etc/hosts`, set `DOMAIN=face.test`, and open http://face.test.

**API calls return 401 but the UI loads**
- The SPA sends `x-face-token` from `VITE_API_TOKEN`; it must equal `SECRET_KEY`. Set both in the root `.env` and recreate: `docker compose up -d --force-recreate web app`.
- An empty `SECRET_KEY` with `DEBUG=false` rejects **all** token-protected calls by design.

**Images or the webcam stream fail with 401**
- `.../image` and `/webcam/stream` are intentionally exempt from the token (browsers can't add headers to `<img>`/`EventSource`) but still require Basic Auth — log in through the UI first.

## Recognition quality

**"No face detected"**
- Face should be frontal, well-lit, and ≥ 80 px. Try `INSIGHTFACE_DET_SIZE=1024` for small faces in large photos, or `320` for low-res webcams.

**Wrong or missed matches**
- Enroll 2–3 photos per person (different angles/lighting); auto-capture will add verified photos over time.
- Tune `SIMILARITY_THRESHOLD`: lower = more lenient, higher = fewer false positives.
- Prefer `INSIGHTFACE_MODEL=antelopev2` (the Docker default) over the lighter packs.

**Recognition is slow (> 1 s per frame)**
- Reduce `INSIGHTFACE_DET_SIZE` to `320`; use `RECOGNITION_MODE=local`; give Docker more CPU/RAM; on servers with NVIDIA GPUs see [deployment.md → GPU](deployment.md#gpu).

**Liveness rejects real faces / passes screens**
- Tune `LIVENESS_THRESHOLD` (higher = stricter). Passive single-image detection has limits — combine with other controls for high-security doors.

## Docker

**Containers won't start**
```bash
docker compose logs            # find the failing service
docker compose down && docker compose up -d
docker compose up -d --build   # after code changes
```

**Port 80/443 already in use** — stop the conflicting service or change the published ports on the `traefik` service.

**Database connection failures**
```bash
docker compose ps postgres && docker compose logs postgres
docker compose restart postgres
docker compose down -v && docker compose up -d   # ⚠ wipes all data
```

**InsightFace models missing/corrupted**
- Models are baked into the image at build time. Rebuild: `docker compose build --no-cache app`. Watch first-start logs: `docker compose logs -f app`.

## Torch on Python 3.13

Torch 2.11+ hits an AST parsing bug on Python 3.13.8 (`ast.parse` raises `SyntaxError` in `torch/_jit_internal.py:_check_overload_body`, which only catches `OSError`). Docker and CI apply the fix automatically. For a local venv:

```bash
sed -i '' 's/except OSError:/except (OSError, SyntaxError):/' \
  .venv/lib/python3.13/site-packages/torch/_jit_internal.py   # macOS; drop '' on Linux
```

Harmless warnings about "unable to retrieve source for @torch.jit._overload function" remain — that's expected.

## Webcam

**Daemon exits immediately** — set `WEBCAM_ENABLED=true` in `.env`.

**Camera not found in Docker** — camera passthrough works on Linux only (`docker-compose.webcam.yaml`). On macOS/Windows run `python webcam_daemon.py` on the host.

**Frames post but nothing is recognized** — the daemon needs valid `BASIC_AUTH_USERNAME`/`BASIC_AUTH_PASSWORD`/`SECRET_KEY` env vars and a reachable `WEBCAM_API_URL`.
