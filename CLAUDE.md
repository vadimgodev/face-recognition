# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceGuard — a self-hosted face recognition platform: FastAPI backend, Vue 3 SPA, PostgreSQL + pgvector, Redis, InsightFace (local, primary) with optional AWS Rekognition hybrid modes, Silent-Face passive liveness detection, webcam daemon with door-unlock integration (mock/HTTP/GPIO). Deployed with Docker Compose behind Traefik (TLS + Basic Auth at the edge, `x-face-token` middleware in the app).

## Environment

- Python 3.13 (venv at `.venv`), Node 22 for the frontend (`web/`)
- Torch on Python 3.13.8 may need a one-line patch in `torch/_jit_internal.py` (`except OSError:` → `except (OSError, SyntaxError):`); Dockerfile and CI apply it automatically via sed
- `docker compose up -d` starts app, postgres (pgvector/pg16), redis, web (Vite dev server), traefik

## Common Commands

- Tests: `pytest` — 545 tests, fully mocked, no Docker required, ~10 s
- Quality gates: `black . && ruff check . && mypy src/`
- Frontend: `cd web && npm run dev` (port 3000) / `npm run build`
- Migrations: `alembic upgrade head`; new: `alembic revision --autogenerate -m "..."`
- DB shell: `docker compose exec postgres psql -U postgres -d facedb`

## Architecture

```
src/
├── api/            routes/ package (recognition, faces, liveness, webcam), deps.py, schemas.py
├── services/       face_service (single entry, strategy engine), recognition_strategies
│                   (Local/Hybrid/CloudStrategy via create_strategy), template_service,
│                   auto_capture_service, multiface_service, webcam_service
├── triggers/       base (MatchEvent/Trigger), providers (log/webhook/gpio), service
├── providers/      registry + interfaces (EmbeddingProvider/CloudMatchProvider),
│                   insightface_provider, aws_rekognition (+ collection_manager sharding),
│                   silent_face_liveness; factory.py returns cached singletons
├── database/       models.py (single `faces` table, pgvector HNSW on embedding_local),
│                   repository.py (all queries)
├── storage/        local (OSFS) / s3 (S3FS) via PyFilesystem2; path-traversal guards
├── middleware/     auth.py — x-face-token check; exempt: /health /docs /redoc /openapi.json /,
│                   paths containing /image or /webcam/stream (Basic Auth only)
├── antispoof/      vendored Silent-Face MiniFASNet ensemble
├── config/         settings.py — SOURCE OF TRUTH for all env vars
└── exceptions.py   FaceRecognitionError hierarchy with status codes
web/src/            views: Enroll, Recognize, Faces, WebcamMonitor; api/faceApi.js (axios)
webcam_daemon.py    standalone daemon (browser mode lives in the Vue view)
```

## Key Facts

- Recognition modes (`RECOGNITION_MODE`, default `local`): `local` (fully offline), `hybrid` (3-tier confidence, AWS `CompareFaces` verify in 0.6–0.8 band, no collection indexing on enroll), `cloud` (AWS Rekognition collections)
- Embeddings: 512-d L2-normalized, similarity = `1 - cosine_distance/2`; user templates are averaged across enrolled + verified photos
- Auto-capture: matches ≥ 0.85 saved as `photo_type='verified'`, FIFO cap 4 per user
- Face IDs are integers (BigInteger autoincrement), not UUIDs
- Tests mock all heavy deps (InsightFace/AWS/torch inference); keep it that way
- Config defaults live in `src/config/settings.py`; `.env.example` documents every var
- Never commit face photos or `.env`; `data/`, `sample_data/`, `models/`, images are gitignored
