# Configuration Reference

All settings are environment variables, loaded from `.env` (see [`.env.example`](../.env.example) for a commented template). The source of truth is `src/config/settings.py`.

## Application

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `faceguard` | Service name reported in logs |
| `APP_ENV` | `development` | `development` / `production` |
| `DEBUG` | `false` | Verbose errors, SQL echo |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Bind address of the FastAPI app |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum accepted image upload size |
| `LOG_LEVEL` | `INFO` | Python log level |

## Database (PostgreSQL + pgvector)

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_PASSWORD` | `postgres` | **Change in production** |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `postgres` / `5432` | Use `localhost` for non-Docker dev |
| `POSTGRES_USER` / `POSTGRES_DB` | `postgres` / `facedb` | |
| `DATABASE_POOL_SIZE` / `DATABASE_MAX_OVERFLOW` | `10` / `20` | SQLAlchemy async pool |

## Redis

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_ENABLED` | `true` | Disable to run without cache (graceful fallback) |
| `REDIS_PASSWORD` | *(empty)* | **Change in production** |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `redis` / `6379` / `0` | |
| `REDIS_MAX_CONNECTIONS` | `50` | Connection pool size |
| `REDIS_CACHE_TTL` | `3600` | Seconds to cache embeddings / results |

## Recognition mode

| Variable | Default | Purpose |
|---|---|---|
| `RECOGNITION_MODE` | `local` | `local` (InsightFace + pgvector, fully offline) / `cloud` (AWS Rekognition collections) / `hybrid` (local search, AWS `CompareFaces` verification in the 0.6–0.8 band) |

Deprecated aliases `FACE_PROVIDER` / `USE_HYBRID_RECOGNITION` / `HYBRID_MODE` still work — mapped at startup (`insightface_only`→`local`, `smart_hybrid`/`insightface_aws`→`hybrid`, `aws_only`→`cloud`) with a deprecation warning logged. Set `RECOGNITION_MODE` directly instead.

## InsightFace model & matching

| Variable | Default | Purpose |
|---|---|---|
| `INSIGHTFACE_MODEL` | `buffalo_l` (code) / `antelopev2` (Docker) | Model pack: `buffalo_s` fast · `buffalo_l` balanced · `antelopev2` best |
| `INSIGHTFACE_DET_SIZE` | `640` | Detection resolution: `320` fast (480–720p) · `640` balanced · `1024` accurate (1080p+) |
| `INSIGHTFACE_CTX_ID` | `-1` | `-1` = CPU, `0`+ = GPU device id |
| `SIMILARITY_THRESHOLD` | `0.6` | Minimum cosine similarity for a match (0.6–0.7 recommended) |

## Providers (advanced)

| Variable | Default | Purpose |
|---|---|---|
| `LOCAL_PROVIDER` | `insightface` | Which registered `EmbeddingProvider` backs `local`/`hybrid` recognition |
| `CLOUD_PROVIDER` | `aws_rekognition` | Which registered `CloudMatchProvider` backs `cloud`/`hybrid` recognition |

Both are resolved from an in-tree registry (`src/providers/registry.py`), not hardcoded to InsightFace/AWS. Adding your own provider (or a new `TRIGGER_PROVIDER`): see [`docs/extending.md`](extending.md).

## Hybrid verification thresholds

| Variable | Default | Purpose |
|---|---|---|
| `INSIGHTFACE_HIGH_CONFIDENCE` | `0.8` | `hybrid`: trust local above this, no AWS call |
| `INSIGHTFACE_MEDIUM_CONFIDENCE` | `0.6` | `hybrid`: AWS-verify (`CompareFaces`) between medium and high |
| `VECTOR_SEARCH_CANDIDATES` | `3` | Reserved; not currently honored (hybrid uses a fixed candidate limit) |
| `AWS_VERIFICATION_COUNT` | `3` | Top candidates considered for AWS re-scoring (reserved; `hybrid` currently verifies a fixed top-3) |

## AWS (`cloud`/`hybrid` verification, or S3 storage)

| Variable | Default | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | *(empty)* | Credentials (or use the boto3 default chain) |
| `AWS_REGION` | `us-east-1` | |
| `AWS_REKOGNITION_COLLECTION_ID` | `faces-collection` | Base name; faces are sharded across `{base}-shard-NN` collections. Used by `cloud` mode only — `hybrid` verifies against the stored image via `CompareFaces` and does not index into collections on enroll |
| `NUM_REKOGNITION_COLLECTIONS` | `10` | Number of consistent-hash collection shards (`cloud` mode) |

## Liveness (anti-spoofing)

| Variable | Default | Purpose |
|---|---|---|
| `LIVENESS_ENABLED` | `false` | Master switch (models load at startup when on) |
| `LIVENESS_PROVIDER` | `silent_face` | Passive Silent-Face MiniFASNet ensemble |
| `LIVENESS_THRESHOLD` | `0.5` | Min "real" score (webcam UI uses 0.6) |
| `LIVENESS_ON_ENROLLMENT` | `true` | Reject spoofed enrollment photos |
| `LIVENESS_ON_RECOGNITION` | `false` | Also gate every recognition call |
| `LIVENESS_MODEL_DIR` | `./models/anti_spoof` | `.pth` ensemble location |
| `LIVENESS_DETECTOR_PATH` | `./models` | OpenCV-DNN fallback detector |
| `LIVENESS_DEVICE_ID` | `-1` | `-1` CPU / GPU device id |

## Multi-face detection & ROI

| Variable | Default | Purpose |
|---|---|---|
| `MULTIFACE_ENABLED` | `false` | Allow >1 face per frame (single-face requests auto-route here) |
| `FACE_DETECTION_METHOD` | `dnn` | `haar` fastest / `dnn` balanced / `insightface` accurate |
| `INSIGHTFACE_DETECTION_MODEL` | `buffalo_s` | Detector pack for detection-only stage |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.5` | Min detection confidence |
| `MAX_FACES_PER_FRAME` | `10` | Largest faces kept |
| `MIN_FACE_SIZE` | `80` | Min face size, px |
| `FACE_CROP_PADDING` | `0.2` | Crop padding ratio |
| `SAVE_ALL_DETECTED_FACES` | `true` | Persist crops of unmatched faces too |
| `ROI_ENABLED` | `false` | Only recognize faces inside a region of interest |
| `ROI_X` / `ROI_Y` | `0.3` / `0.2` | ROI top-left (normalized 0–1) |
| `ROI_WIDTH` / `ROI_HEIGHT` | `0.4` / `0.6` | ROI size (normalized) |
| `ROI_MIN_OVERLAP` | `0.3` | Min face/ROI overlap to count |

## Face quality gates (enrollment)

| Variable | Default | Purpose |
|---|---|---|
| `FACE_QUALITY_MIN_SIZE` | `80` | Reject tiny faces |
| `FACE_QUALITY_MAX_BLUR` | `100.0` | Laplacian blur threshold |
| `FACE_QUALITY_MIN_BRIGHTNESS` / `FACE_QUALITY_MAX_BRIGHTNESS` | `40.0` / `220.0` | Exposure bounds |

## Auto-capture

| Variable | Default | Purpose |
|---|---|---|
| `AUTO_CAPTURE_ENABLED` | `true` | Save high-confidence sightings as `verified` photos |
| `AUTO_CAPTURE_CONFIDENCE_THRESHOLD` | `0.85` | Min similarity to capture |
| `AUTO_CAPTURE_MAX_VERIFIED_PHOTOS` | `4` | FIFO cap per user |

## Webcam & triggers

| Variable | Default | Purpose |
|---|---|---|
| `WEBCAM_ENABLED` | `false` | Allow server-side capture (`/api/v1/webcam/*`, daemon) |
| `WEBCAM_DEVICE_ID` | `0` | Camera index |
| `WEBCAM_FPS` | `2` | Frames analyzed per second |
| `WEBCAM_SUCCESS_COOLDOWN_SECONDS` | `5` | Pause after successful recognition |
| `WEBCAM_API_URL` | `http://localhost:8000` | Where the daemon posts frames |
| `TRIGGER_PROVIDER` | `log` | On-match trigger: `log` / `webhook` / `gpio` (Raspberry Pi pin 17) |
| `TRIGGER_WEBHOOK_URL` | *(empty)* | POST target for the `webhook` provider |
| `TRIGGER_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence to fire the trigger |
| `TRIGGER_GPIO_PIN` | `17` | GPIO pin for the `gpio` provider |
| `ACCESS_LOG_OUTPUT` | `stdout` | `stdout` / `file` / `both` |
| `ACCESS_LOG_FILE_PATH` | `/var/log/faceguard/access.log` | When file output enabled |
| `ACCESS_LOG_FORMAT` | `json` | `json` / `text` |
| `ACCESS_LOG_INCLUDE_COOLDOWN_EVENTS` | `false` | Log skipped-due-to-cooldown frames |

Deprecated aliases `DOOR_UNLOCK_PROVIDER` / `DOOR_UNLOCK_URL` / `DOOR_UNLOCK_CONFIDENCE_THRESHOLD` still work (`mock`→`log`, `http`→`webhook`) — set `TRIGGER_*` directly instead.

The `webhook` provider POSTs the match as JSON (`MatchEvent.to_payload()`):

```json
{
  "user_name": "John Doe",
  "confidence": 0.9192,
  "processor": "antelopev2",
  "user_email": "john@example.com",
  "camera_id": 0,
  "liveness_passed": true,
  "timestamp": "2026-07-23T12:36:32.481203+00:00"
}
```

## Storage

| Variable | Default | Purpose |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` / `s3` |
| `STORAGE_LOCAL_PATH` | `./data/images` | Local image root |
| `STORAGE_S3_BUCKET` / `STORAGE_S3_REGION` | *(empty)* / `us-east-1` | Required for `s3` |

## Security & edge

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | *(empty)* | Value of the `x-face-token` header. **Required** — with an empty key (and `DEBUG=false`) every API call is rejected |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8000` | CORS allowlist (comma-separated) |
| `DOMAIN` | `localhost` | Host that Traefik routes (set your domain in production) |
| `BASIC_AUTH_USERNAME` / `BASIC_AUTH_PASSWORD` | — | Edge Basic Auth (htpasswd is generated at startup); also used by the webcam daemon |
| `ACME_EMAIL` | `admin@example.com` | Let's Encrypt registration email |

## Frontend (Vite)

| Variable | Purpose |
|---|---|
| `VITE_API_TOKEN` | Must equal `SECRET_KEY`; injected as `x-face-token` by the SPA |
| `VITE_API_URL` | API base URL (defaults to `/api/v1` behind Traefik) |

In the Docker setup these can live in the root `.env` (the `web` container inherits it). For standalone `npm run dev`, put them in `web/.env`.
