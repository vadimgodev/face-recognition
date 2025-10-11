from datetime import datetime
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as faces_router, webcam_router
from src.api.schemas import HealthCheckResponse, ErrorResponse
from src.config.settings import settings
from src.database.base import engine
from src.providers.factory import get_face_provider
from src.middleware.auth import APITokenMiddleware
from src.cache.redis_client import get_redis_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup validation: Check critical configuration requirements
    logger.info("Running startup validation checks...")
    try:
        from src.utils.startup_validation import validate_startup_requirements
        validate_startup_requirements(fail_on_error=True)
    except RuntimeError as e:
        logger.error(f"Startup validation failed: {e}")
        raise

    # Initialize Redis cache
    logger.info("Initializing Redis cache...")
    try:
        redis_client = get_redis_client()
        await redis_client.initialize()
        logger.info("✅ Redis cache initialized")
    except Exception as e:
        logger.warning(f"Redis initialization failed (will continue without cache): {e}")

    # Startup: Initialize all face recognition collections (sharded)
    logger.info("Initializing Face Recognition API...")
    try:
        provider = get_face_provider()

        # Initialize all sharded collections
        results = await provider.initialize_all_collections()

        initialized = results.get("initialized", [])
        failed = results.get("failed", [])

        if initialized:
            logger.info(f"Initialized {len(initialized)} collection(s):")
            for collection_id in initialized:
                logger.info(f"  - {collection_id}")

        if failed:
            logger.error(f"Failed to initialize {len(failed)} collection(s):")
            for failure in failed:
                logger.error(f"  - {failure['collection_id']}: {failure['error']}")

        if not failed:
            logger.info("All collections ready!")
    except Exception as e:
        logger.error(f"Failed to initialize collections: {e}", exc_info=True)

    # Warm up models to avoid slow first request
    logger.info("Warming up models...")
    try:
        import asyncio

        # Warm up liveness detection models if enabled
        if settings.liveness_enabled:
            logger.info("Preloading liveness detection models...")
            from src.providers.silent_face_liveness import get_liveness_provider

            liveness_provider = get_liveness_provider(
                device_id=settings.liveness_device_id,
                model_dir=settings.liveness_model_dir,
                detector_path=settings.liveness_detector_path,
            )
            # Trigger lazy loading by accessing the predictor
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: liveness_provider._get_predictor()
            )
            logger.info("✅ Liveness detection models loaded")

        # Warm up face recognition provider (InsightFace or AWS)
        if settings.face_provider == "insightface" or settings.use_hybrid_recognition:
            logger.info("Preloading InsightFace models...")
            from src.providers.factory import get_insightface_provider

            insightface_provider = get_insightface_provider()
            # Trigger lazy loading
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: insightface_provider._get_app()
            )
            logger.info("✅ InsightFace models loaded")

        logger.info("✅ All models warmed up and ready!")
    except Exception as e:
        logger.warning(f"Model warmup encountered errors (non-fatal): {e}", exc_info=True)

    yield

    # Shutdown: Close Redis connections
    try:
        redis_client = get_redis_client()
        await redis_client.close()
        logger.info("Redis connections closed")
    except Exception as e:
        logger.warning(f"Error closing Redis connections: {e}")

    # Shutdown: Close database connections
    await engine.dispose()
    logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Face Recognition API",
    description="A flexible face recognition system with enrollment and identification capabilities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API Token Authentication
app.add_middleware(APITokenMiddleware)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc) if settings.debug else None,
        ).model_dump(),
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the API is running",
)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
    )


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API root",
    description="Get API information",
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Face Recognition API",
        "version": "0.1.0",
        "description": "A flexible face recognition system",
        "docs": "/docs",
        "health": "/health",
    }


# Include routers
app.include_router(faces_router)
app.include_router(webcam_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
