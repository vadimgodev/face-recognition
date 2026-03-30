# Face Recognition API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-236%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-34%25-yellow)](tests/)

A production-ready face recognition system with enrollment and identification capabilities. Built with FastAPI, PostgreSQL, InsightFace, and AWS Rekognition, featuring a modern Vue.js frontend and enterprise-grade two-layer authentication.

> **Privacy Notice:** This software processes biometric data. Please review the [Ethical Use](#ethical-use) section and ensure compliance with applicable privacy laws before deployment.

## 🚀 Features

### Core Functionality
- **Face Enrollment** - Register users with facial recognition
- **Face Recognition** - Identify users from uploaded photos
- **Smart Hybrid Recognition** - Combines local InsightFace with AWS Rekognition for optimal cost/accuracy
- **Auto-Capture** - Automatically captures high-confidence verification photos
- **Template Averaging** - Improves accuracy by averaging multiple face templates
- **Photo Management** - View, manage, and delete enrolled/verified photos
- **Webcam Monitoring** - Real-time face recognition from webcam for door access control
  - Sequential processing (2 FPS) with 5-second cooldown after success
  - Browser mode (development/testing) and daemon mode (production)
  - Structured JSON logging for ELK/Loki integration
  - Door unlock abstraction (mock/HTTP/GPIO providers)

### Technical Features
- **Multi-Provider Support** - InsightFace (local) and AWS Rekognition (cloud)
- **Cost Optimization** - Smart hybrid mode reduces AWS API calls by 70-90%
- **High Accuracy** - InsightFace achieves 95%+ accuracy on verification
- **Real-time Processing** - Fast face detection and recognition
- **RESTful API** - Well-documented FastAPI endpoints
- **Modern UI** - Vue.js 3 single-page application
- **Docker Deployment** - Complete containerized setup with Traefik

### Security
- **Two-Layer Authentication**
  - Layer 1: HTTP Basic Auth via Traefik
  - Layer 2: API Token via FastAPI middleware
- **HTTPS Support** - Let's Encrypt SSL certificates (production)
- **CORS Protection** - Configurable allowed origins
- **Secure Credentials** - Environment-based configuration

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/vadimgodev/face-recognition
cd face-recognition-api

# 2. Copy environment file
cp .env.example .env

# 3. Edit .env with your credentials
nano .env

# 4. Start all services
docker-compose up -d

# 5. Add to hosts file (development)
echo "127.0.0.1 face.test" | sudo tee -a /etc/hosts

# 6. Access the application
open http://face.test
```

**Default credentials (development only - change in production):**
- Username: set via `BASIC_AUTH_USERNAME` in `.env`
- Password: set via `BASIC_AUTH_PASSWORD` in `.env`
- See `.env.example` for configuration

**⚠️ IMPORTANT:** You must access via `http://face.test` (not localhost) for authentication to work properly.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Traefik                              │
│         (Reverse Proxy + SSL + Basic Auth)                   │
│                    Port 80/443                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼────┐         ┌────▼────┐
    │   Web   │         │   API   │
    │ Vue.js  │────────>│ FastAPI │
    │  :3000  │         │  :8000  │
    └─────────┘         └────┬────┘
                             │
                   ┌─────────┼─────────┐
                   │         │         │
              ┌────▼───┐ ┌───▼────┐ ┌─▼──────┐
              │Postgres│ │ Redis  │ │InsightF│
              │ :5432  │ │ :6379  │ │  ace   │
              └────────┘ └────────┘ └────────┘
                                         │
                                    ┌────▼────┐
                                    │   AWS   │
                                    │Rekognit.│
                                    └─────────┘
```

### Technology Stack

**Backend:**
- FastAPI 0.104+ - High-performance async API framework
- PostgreSQL 16 + pgvector - Database with vector extension
- Redis 7 - Caching and rate limiting
- InsightFace - Local face recognition (antelopev2 model)
- AWS Rekognition - Cloud face recognition (optional)
- SQLAlchemy 2.0 - Async ORM
- Alembic - Database migrations

**Frontend:**
- Vue.js 3 - Progressive JavaScript framework
- Vite - Fast build tool
- Axios - HTTP client with credential support

**Infrastructure:**
- Docker & Docker Compose - Containerization
- Traefik v3 - Reverse proxy and SSL termination
- Let's Encrypt - Free SSL certificates

## 📦 Prerequisites

- **Docker** 20.10+
- **Docker Compose** 2.0+
- **4GB RAM** minimum (8GB recommended)
- **AWS Account** (optional, for AWS Rekognition)

## 🔧 Installation

### 1. Clone Repository

```bash
git clone https://github.com/vadimgodev/face-recognition
cd face-recognition-api
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Essential settings:**

```bash
# Domain (use face.test for development)
DOMAIN=face.test

# Authentication - CHANGE THESE!
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=your-strong-password-here
SECRET_KEY=generate-random-32-character-key-here

# Face Recognition Mode
HYBRID_MODE=insightface_only  # No AWS required
INSIGHTFACE_MODEL=antelopev2  # Best accuracy
SIMILARITY_THRESHOLD=0.6

# AWS (optional, only if using hybrid/aws modes)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
```

### 3. Add to Hosts File (Development)

```bash
# macOS/Linux
echo "127.0.0.1 face.test" | sudo tee -a /etc/hosts

# Windows (run as Administrator)
# Add to C:\Windows\System32\drivers\etc\hosts:
127.0.0.1 face.test
```

### 4. Start Services

```bash
# Start all containers
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

First startup will download InsightFace models (~350MB), takes 1-2 minutes.

### 5. Access Application

Open browser: **http://face.test**

**⚠️ Must use `face.test` not `localhost`!**

Enter credentials when prompted:
- Username: From `BASIC_AUTH_USERNAME`
- Password: From `BASIC_AUTH_PASSWORD`

## ⚙️ Configuration

### Face Recognition Modes

**1. InsightFace Only** (Recommended for most users)
```bash
HYBRID_MODE=insightface_only
```
- ✅ Free, no AWS costs
- ✅ Fast local processing (~100-200ms)
- ✅ 95%+ accuracy
- ✅ Privacy (data stays local)
- ✅ No internet required

**2. Smart Hybrid** (Best for production with AWS)
```bash
HYBRID_MODE=smart_hybrid
```
- ✅ High confidence (≥0.8): InsightFace only → 0 cost
- ✅ Medium confidence (0.6-0.8): Local re-verification → 0 cost
- ✅ Low confidence (<0.6): AWS Rekognition fallback → minimal cost
- ✅ 70-90% cost reduction vs AWS-only
- ✅ Best accuracy for edge cases

**3. AWS Only**
```bash
HYBRID_MODE=aws_only
```
- ✅ Maximum accuracy
- ❌ Higher costs (~$1 per 1000 API calls)
- ❌ Requires internet
- ❌ Data sent to cloud

### Detection Settings

```bash
# Detection size - balance speed vs accuracy
INSIGHTFACE_DET_SIZE=640  # 320, 640, or 1024

# 320: Fast, good for webcams (480p-720p)
# 640: Balanced, recommended ← DEFAULT
# 1024: High accuracy, for 1080p+ cameras
```

### Similarity Threshold

```bash
SIMILARITY_THRESHOLD=0.6  # 0.0 to 1.0

# 0.5-0.6: More lenient, more matches
# 0.6-0.7: Balanced ← RECOMMENDED
# 0.7-0.8: Stricter, fewer false positives
```

### Auto-Capture Settings

```bash
AUTO_CAPTURE_ENABLED=true
AUTO_CAPTURE_CONFIDENCE_THRESHOLD=0.85
AUTO_CAPTURE_MAX_VERIFIED_PHOTOS=5
```

When recognition confidence ≥ 0.85, automatically saves the photo to improve future matches.

### Webcam Settings

```bash
# Enable webcam capture
WEBCAM_ENABLED=true
WEBCAM_DEVICE_ID=0  # Camera ID (0 = default)
WEBCAM_FPS=2  # Capture rate (1-5 recommended)
WEBCAM_SUCCESS_COOLDOWN_SECONDS=5  # Pause after successful recognition
WEBCAM_API_URL=http://localhost:8000  # API endpoint (use http://face.test for Traefik)

# Door unlock configuration
DOOR_UNLOCK_PROVIDER=mock  # Options: mock, http, gpio
DOOR_UNLOCK_URL=http://door-controller/unlock  # For HTTP provider
DOOR_UNLOCK_CONFIDENCE_THRESHOLD=0.85  # Min confidence to unlock

# Access logging
ACCESS_LOG_OUTPUT=stdout  # Options: stdout, file, both
ACCESS_LOG_FORMAT=json  # Options: json, text
```

**Usage:**
- **Browser mode** (development): Navigate to `/monitoring` in the Vue app
- **Daemon mode** (production): Run `python webcam_daemon.py --camera 0 --mode daemon`
- **macOS users**: Run daemon on host (Docker Desktop doesn't support camera devices)
- **Linux users**: Can use `docker-compose.webcam.yaml` to run in Docker

### Security Settings

```bash
# HTTP Basic Auth (Layer 1)
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=change-this-password

# API Token (Layer 2)
SECRET_KEY=change-this-to-random-32-char-key

# CORS (which origins can access API)
ALLOWED_ORIGINS=http://face.test,https://face.test

# SSL/HTTPS
ACME_EMAIL=admin@example.com
```

## 🔐 Authentication

### Two-Layer Security System

Every API request requires **both** authentication layers:

```
Request Flow:
    │
    ▼
┌─────────────────────────────┐
│ Layer 1: HTTP Basic Auth    │ ← Traefik checks
│ Username + Password          │
└──────────┬──────────────────┘
           │ ✓ Valid
           ▼
┌─────────────────────────────┐
│ Layer 2: API Token          │ ← FastAPI checks
│ x-face-token header          │
└──────────┬──────────────────┘
           │ ✓ Valid
           ▼
     API Endpoint
```

### Frontend Configuration

The frontend is pre-configured to send both authentication layers automatically.

**Configure in `web/.env`:**
```bash
VITE_API_TOKEN=your-secret-key  # Must match SECRET_KEY in root .env
```

### Making Direct API Requests

**Command Line:**
```bash
curl -u admin:password \
  -H "x-face-token: your-secret-key" \
  http://face.test/api/v1/faces
```

**Python:**
```python
import requests

auth = ('admin', 'password')
headers = {'x-face-token': 'your-secret-key'}

response = requests.get(
    'http://face.test/api/v1/faces',
    auth=auth,
    headers=headers
)
print(response.json())
```

**JavaScript:**
```javascript
const auth = btoa('admin:password');
const headers = {
    'Authorization': `Basic ${auth}`,
    'x-face-token': 'your-secret-key'
};

fetch('http://face.test/api/v1/faces', {
    credentials: 'include',
    headers
})
.then(r => r.json())
.then(data => console.log(data));
```

### Excluded Endpoints

These require **only Basic Auth** (no x-face-token):
- `/health` - Health check
- `/docs` - API documentation
- `/redoc` - Alternative API docs
- `/openapi.json` - OpenAPI spec
- `/api/v1/faces/{id}/image` - Face images (for `<img>` tags)

## 🎯 Usage

### Web Interface

**1. Enroll a New User**
1. Click "Enroll Face"
2. Upload photo or use webcam
3. Enter name and email
4. Click "Enroll Face"

**2. Recognize a User**
1. Click "Recognize Face"
2. Upload photo or use webcam
3. View matched user with confidence score
4. See which processor was used (InsightFace/AWS)

**3. Manage Enrolled Users**
1. Click "Faces"
2. View all enrolled users with photo counts
3. Click "View All Photos" to see enrolled + verified photos
4. Delete individual photos or entire users

### API Examples

#### Enroll a Face

```bash
curl -X POST http://face.test/api/v1/faces/enroll \
  -u admin:password \
  -H "x-face-token: your-secret-key" \
  -F "image=@photo.jpg" \
  -F "user_name=John Doe" \
  -F "user_email=john@example.com"
```

**Response:**
```json
{
  "success": true,
  "message": "Face enrolled successfully",
  "face": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "user_name": "John Doe",
    "user_email": "john@example.com",
    "photo_type": "enrolled",
    "confidence": null,
    "created_at": "2025-11-03T12:00:00Z"
  }
}
```

#### Recognize a Face

```bash
curl -X POST http://face.test/api/v1/faces/recognize \
  -u admin:password \
  -H "x-face-token: your-secret-key" \
  -F "image=@unknown.jpg"
```

**Response (match found):**
```json
{
  "success": true,
  "matched": true,
  "confidence": 0.92,
  "processor": "insightface",
  "match": {
    "face_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_name": "John Doe",
    "user_email": "john@example.com",
    "similarity": 0.92
  }
}
```

**Response (no match):**
```json
{
  "success": true,
  "matched": false,
  "confidence": 0.0,
  "processor": "insightface",
  "match": null
}
```

#### List All Faces

```bash
curl http://face.test/api/v1/faces \
  -u admin:password \
  -H "x-face-token: your-secret-key"
```

#### Get User Photos

```bash
curl http://face.test/api/v1/faces/user/John%20Doe/photos \
  -u admin:password \
  -H "x-face-token: your-secret-key"
```

#### Delete a Face

```bash
curl -X DELETE http://face.test/api/v1/faces/{face_id} \
  -u admin:password \
  -H "x-face-token: your-secret-key"
```

## 📚 API Documentation

Interactive API documentation available at:

- **Swagger UI**: http://face.test/docs
- **ReDoc**: http://face.test/redoc

### Available Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| POST | `/api/v1/faces/enroll` | Both | Enroll new face |
| POST | `/api/v1/faces/recognize` | Both | Recognize face |
| GET | `/api/v1/faces` | Both | List all faces |
| GET | `/api/v1/faces/{id}` | Both | Get face by ID |
| GET | `/api/v1/faces/{id}/image` | Basic only | Get face image |
| DELETE | `/api/v1/faces/{id}` | Both | Delete face |
| GET | `/api/v1/faces/user/{name}/photos` | Both | Get user photos |
| GET | `/health` | Basic only | Health check |

**Auth Required:**
- **Both**: Basic Auth + x-face-token header
- **Basic only**: Just Basic Auth

## 💻 Development

### Project Structure

```
face-recognition-api/
├── src/                    # Backend
│   ├── api/
│   │   ├── routes.py      # FastAPI endpoints
│   │   └── schemas.py     # Pydantic models
│   ├── services/
│   │   └── face_service.py # Business logic
│   ├── providers/
│   │   ├── insightface_provider.py
│   │   └── aws_rekognition.py
│   ├── database/
│   │   ├── models.py      # SQLAlchemy models
│   │   └── repository.py
│   ├── middleware/
│   │   └── auth.py        # API token authentication
│   ├── config/
│   │   └── settings.py    # Configuration
│   └── main.py            # Application entry
├── web/                   # Frontend
│   ├── src/
│   │   ├── api/
│   │   │   └── faceApi.js # API client
│   │   ├── components/
│   │   │   └── UserPhotosModal.vue
│   │   ├── views/
│   │   │   ├── EnrollView.vue
│   │   │   ├── RecognizeView.vue
│   │   │   └── FacesView.vue
│   │   ├── router/
│   │   │   └── index.js
│   │   ├── App.vue
│   │   └── main.js
│   ├── .env               # Frontend env vars
│   └── vite.config.js
├── alembic/               # Database migrations
├── docker-compose.yaml    # Docker orchestration
├── Dockerfile            # Backend container
├── .env                  # Backend env vars
├── .env.example          # Template
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

### Local Development

**Backend:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start dev server
uvicorn src.main:app --reload
```

**Frontend:**
```bash
cd web
npm install
npm run dev
```

Access:
- Frontend dev server: http://localhost:5173
- Backend API: http://localhost:8000

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "add new column"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Connect to database
docker-compose exec postgres psql -U postgres -d facedb
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy src/

# Run all checks
black . && ruff check . && mypy src/
```

### Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_face_service.py::test_enroll_face

# View coverage
open htmlcov/index.html
```

## 🐛 Troubleshooting

### Authentication Issues

**❌ Problem:** Login prompt on every page reload

**✅ Solution:**
- Ensure you're using `http://face.test` NOT `http://localhost`
- Add `127.0.0.1 face.test` to `/etc/hosts`
- Don't use incognito/private browsing mode
- Browser should cache credentials automatically

**❌ Problem:** Images not loading / 401 errors on images

**✅ Solution:**
- Images are excluded from token auth (Basic Auth only)
- Restart containers: `docker-compose restart`
- Check browser console for specific errors

**❌ Problem:** API calls failing with 401

**✅ Solution:**
- Ensure `VITE_API_TOKEN` in `web/.env` matches `SECRET_KEY` in root `.env`
- Recreate containers: `docker-compose up -d`
- Check browser Network tab for missing `x-face-token` header

### Face Recognition Issues

**❌ Problem:** "No face detected"

**✅ Solutions:**
1. Ensure face is clearly visible, well-lit, frontal
2. Try different detection size: `INSIGHTFACE_DET_SIZE=320` (faster) or `1024` (better)
3. Check image isn't too large (>5MB) or too small (<100px)

**❌ Problem:** Low accuracy / wrong matches

**✅ Solutions:**
1. Increase detection size: `INSIGHTFACE_DET_SIZE=1024`
2. Use better quality enrollment photos
3. Enroll multiple photos per person (2-3 recommended)
4. Adjust threshold: Lower = more lenient (`SIMILARITY_THRESHOLD=0.5`)
5. Use better model: `INSIGHTFACE_MODEL=antelopev2`

**❌ Problem:** Slow recognition (>1 second)

**✅ Solutions:**
1. Decrease detection size: `INSIGHTFACE_DET_SIZE=320`
2. Use `insightface_only` mode
3. Ensure enough RAM (4GB minimum)
4. Check if CPU is overwhelmed

### Docker Issues

**❌ Problem:** Containers won't start

```bash
# Check logs for specific error
docker-compose logs

# Clean restart
docker-compose down
docker-compose up -d

# Rebuild if code changed
docker-compose up -d --build
```

**❌ Problem:** "Port already in use"

```bash
# Find what's using port 80
sudo lsof -i :80

# Kill the process or change port in docker-compose.yaml
ports:
  - "8080:80"  # Use 8080 instead
```

**❌ Problem:** Database connection failed

```bash
# Check if PostgreSQL is healthy
docker-compose ps postgres

# View database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres

# Reset database (⚠️ deletes all data)
docker-compose down -v
docker-compose up -d
```

**❌ Problem:** InsightFace models not downloading

```bash
# Check app logs
docker-compose logs app

# Download manually inside container
docker-compose exec app python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2')"

# Check disk space
df -h
```

### Performance Issues

**Symptoms:** High CPU/RAM usage, slow responses

**Solutions:**
1. Reduce `INSIGHTFACE_DET_SIZE` from 640 to 320
2. Limit concurrent requests
3. Increase Docker memory limit (Docker Desktop settings)
4. Use `insightface_only` mode (faster than hybrid)
5. Close unused applications

## 🚀 Production Deployment

### Prerequisites

1. **Server** with Docker installed
2. **Domain name** pointing to server IP
3. **SSL certificate** (Let's Encrypt automatic via Traefik)
4. **Firewall** configured (ports 80, 443 open)

### Production Configuration

```bash
# .env for production
APP_ENV=production
DEBUG=false

# Your domain
DOMAIN=yourdomain.com

# Strong credentials - CHANGE THESE!
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=$(openssl rand -base64 24)
SECRET_KEY=$(openssl rand -base64 32)

# HTTPS redirect enabled
# Not needed if using Traefik (already configured)

# Email for Let's Encrypt
ACME_EMAIL=admin@yourdomain.com

# Production face recognition
HYBRID_MODE=smart_hybrid
INSIGHTFACE_MODEL=antelopev2
SIMILARITY_THRESHOLD=0.7

# AWS for smart hybrid fallback
AWS_ACCESS_KEY_ID=your-production-key
AWS_SECRET_ACCESS_KEY=your-production-secret
```

### Deployment Steps

```bash
# 1. SSH to server
ssh user@yourdomain.com

# 2. Clone repository
git clone <repository> /opt/face-recognition
cd /opt/face-recognition

# 3. Configure environment
cp .env.example .env
nano .env  # Edit with production settings

# 4. Start services
docker-compose up -d

# 5. Check logs
docker-compose logs -f

# 6. Test HTTPS
curl https://yourdomain.com/health
```

### Security Checklist

- [ ] Changed default credentials
- [ ] Generated strong random `SECRET_KEY` (32+ characters)
- [ ] Configured real email for `ACME_EMAIL`
- [ ] Firewall allows only ports 80, 443
- [ ] Disabled debug mode (`DEBUG=false`)
- [ ] Set `APP_ENV=production`
- [ ] Reviewed `ALLOWED_ORIGINS` (should include your domain)
- [ ] Set up monitoring/logging
- [ ] Configured backup strategy
- [ ] Reviewed face recognition thresholds for your use case
- [ ] Set up log rotation
- [ ] Documented recovery procedures

### Backup Strategy

**Database Backup:**
```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres facedb | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip < backup_20251103.sql.gz | docker-compose exec -T postgres psql -U postgres facedb
```

**Images Backup:**
```bash
# Backup images
tar -czf images_backup_$(date +%Y%m%d).tar.gz data/images/

# Restore images
tar -xzf images_backup_20251103.tar.gz
```

**Full Backup Script:**
```bash
#!/bin/bash
BACKUP_DIR="/backups/face-recognition"
DATE=$(date +%Y%m%d_%H%M%S)

# Database
docker-compose exec postgres pg_dump -U postgres facedb | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Images
tar -czf "$BACKUP_DIR/images_$DATE.tar.gz" data/images/

# Environment
cp .env "$BACKUP_DIR/env_$DATE"

# Keep only last 7 days
find $BACKUP_DIR -mtime +7 -delete
```

### Monitoring

```bash
# View application logs
docker-compose logs -f app

# View all logs
docker-compose logs -f

# Check resource usage
docker stats

# Health check endpoint
curl http://localhost/health

# Watch logs in real-time
docker-compose logs -f --tail=100
```

## Ethical Use

This software processes biometric data. Deployers are responsible for:

- Obtaining informed consent before collecting face data
- Complying with applicable biometric privacy laws (GDPR, BIPA, CCPA, etc.)
- Implementing appropriate data retention and deletion policies
- Using the anti-spoofing features to prevent unauthorized access
- Not using this software for mass surveillance or discriminatory purposes

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Note:** Third-party model weights (InsightFace, Silent-Face-Anti-Spoofing) may have their own license terms. Please verify before commercial use.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For reporting security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Support

- **Bug Reports:** [GitHub Issues](https://github.com/vadimgodev/face-recognition/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vadimgodev/face-recognition/discussions)

## 🙏 Acknowledgments

- **InsightFace** - High-performance face recognition library
- **FastAPI** - Modern Python web framework
- **Vue.js** - Progressive JavaScript framework
- **Traefik** - Cloud-native application proxy
- **PostgreSQL** & **pgvector** - Powerful database with vector support

---

**Built with ❤️ for secure and accurate face recognition**

Last Updated: March 2026
