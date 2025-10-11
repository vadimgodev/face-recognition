# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2025-03-29

### Added
- Face enrollment and recognition API with FastAPI
- Provider abstraction supporting AWS Rekognition and InsightFace
- Smart hybrid recognition mode (local InsightFace + cloud AWS fallback)
- Anti-spoofing with liveness detection
- Web UI built with Vue.js 3
- Webcam daemon for door access control
- PostgreSQL with pgvector for embedding storage
- Redis caching for recognition results
- S3 and local filesystem storage backends
- Docker Compose deployment with Nginx reverse proxy
- Two-layer authentication (Basic Auth + API Token)
- HTTPS support via Let's Encrypt
- Batch enrollment script
- Comprehensive API documentation via OpenAPI/Swagger
