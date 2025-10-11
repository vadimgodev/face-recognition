# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Face Recognition API - A flexible face recognition system with enrollment and identification capabilities. Built with FastAPI, PostgreSQL, and AWS Rekognition, featuring provider abstraction for easy integration of multiple face recognition services.

## Development Environment

### Python Environment
- Python version: 3.9.6
- Virtual environment location: `.venv/`
- Activate environment: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

### Docker Setup
The application runs in Docker with the following services:
- **app**: FastAPI application (port 8000)
- **postgres**: PostgreSQL 16 with pgvector extension (port 5432)
- **redis**: Redis 7 for caching (port 6379)
- **nginx**: Reverse proxy (port 80)

Start all services: `docker-compose up -d`
Stop services: `docker-compose down`
View logs: `docker-compose logs -f app`

### Common Commands

**Development:**
- Install dependencies: `pip install -r requirements.txt`
- Run app locally: `uvicorn src.main:app --reload`
- Run in Docker: `docker-compose up -d`

**Code Quality:**
- Format code: `black .`
- Lint: `ruff check .`
- Type check: `mypy src/`
- Run all checks: `black . && ruff check . && mypy src/`

**Testing:**
- Run tests: `pytest`
- With coverage: `pytest --cov=src --cov-report=html`
- Single test: `pytest tests/test_file.py::test_function`

**Database:**
- Create migration: `alembic revision --autogenerate -m "description"`
- Apply migrations: `alembic upgrade head`
- Rollback: `alembic downgrade -1`
- Connect to DB: `docker-compose exec postgres psql -U postgres -d facedb`

## Architecture

### Directory Structure
```
src/
├── api/          # FastAPI routes and endpoints
├── services/     # Business logic (enrollment, recognition)
├── providers/    # Face recognition provider abstractions
├── storage/      # PyFilesystem2 storage abstraction
├── database/     # SQLAlchemy models and repositories
└── config/       # Configuration management (Pydantic Settings)
```

### Key Design Patterns

**Provider Abstraction:**
- Abstract `FaceProvider` interface in `src/providers/base.py`
- Concrete implementations: `AWSRekognitionProvider`, etc.
- Factory pattern for provider instantiation based on config
- Current provider: AWS Rekognition
- Future providers: Azure Face API, Google Cloud Vision

**Storage Abstraction:**
- Uses PyFilesystem2 for unified storage interface
- Supports local filesystem and S3 (configurable via `STORAGE_BACKEND`)
- Storage factory in `src/storage/`

**Database Layer:**
- SQLAlchemy async ORM with PostgreSQL
- pgvector extension for embedding storage (when providers support it)
- Repository pattern for data access
- Alembic for migrations

**Configuration:**
- Pydantic Settings for type-safe configuration
- Environment variables via `.env` file
- Config defined in `src/config/settings.py`

### Core Features

1. **Face Enrollment** (`POST /api/v1/faces/enroll`)
   - Upload face image with user metadata
   - Store in provider's face collection
   - Save metadata in PostgreSQL

2. **Face Recognition** (`POST /api/v1/faces/recognize`)
   - Upload image for identification
   - Query provider for matches
   - Return user information with confidence scores

3. **Face Management**
   - List faces: `GET /api/v1/faces`
   - Get by ID: `GET /api/v1/faces/{id}`
   - Delete: `DELETE /api/v1/faces/{id}`

### Important Notes

- AWS Rekognition stores face data in Collections (you don't get raw embeddings)
- For providers that expose embeddings, store them in PostgreSQL using pgvector
- Images can be stored locally or in S3 based on `STORAGE_BACKEND` setting
- Redis is used for caching recognition results and rate limiting
- All API endpoints are documented in OpenAPI/Swagger at `/docs`
