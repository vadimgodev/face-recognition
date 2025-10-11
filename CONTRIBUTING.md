# Contributing to Face Recognition API

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose 2.0+
- Node.js 20+ (for frontend development)

### Local Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-api.git
cd face-recognition-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Start infrastructure (database, redis)
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn src.main:app --reload
```

### Frontend Development

```bash
cd web
npm install
npm run dev
```

## Code Style

This project enforces consistent code style using automated tools:

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

Configuration is in `pyproject.toml`. Please ensure all checks pass before submitting a PR.

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_file.py::test_function
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Ensure all code quality checks pass (`black`, `ruff`, `mypy`)
4. Ensure all tests pass (`pytest`)
5. Add tests for new functionality
6. Update documentation if needed
7. Submit a pull request with a clear description of your changes

## Biometric Data Policy

**This is critical for a face recognition project:**

- **NEVER** commit real face photos or biometric data to the repository
- All image files (`*.jpg`, `*.jpeg`, `*.png`) are excluded via `.gitignore`
- Use publicly available face datasets (e.g., [LFW](http://vis-www.cs.umass.edu/lfw/)) for testing
- If your contribution involves test images, use synthetic/generated faces or clearly consented samples
- See `sample_data/README.md` for guidance on test data

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- For security vulnerabilities, see `SECURITY.md`
- Include reproduction steps, expected behavior, and actual behavior in bug reports

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
