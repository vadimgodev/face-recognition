FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# InsightFace requires: libGL (OpenGL), libglib2.0 (GLib)
# Webcam requires: v4l-utils, libv4l-dev (Video4Linux)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use GPU requirements on x86_64 (NVIDIA), CPU requirements on ARM64 (Mac/other)
COPY requirements.txt requirements-gpu.txt ./
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir -r requirements-gpu.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy model download script
COPY scripts/download_models.py /tmp/download_models.py

# Download and prepare InsightFace models during build
# This ensures models are baked into the image with correct directory structure
ENV INSIGHTFACE_MODEL=antelopev2
RUN python3 /tmp/download_models.py && rm /tmp/download_models.py

# Copy DNN face detection models (baked into image)
COPY models/ ./models/

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY webcam_daemon.py .

# Create data directory for local storage
RUN mkdir -p /app/data/images

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
