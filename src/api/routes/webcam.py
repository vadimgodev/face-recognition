from __future__ import annotations

import asyncio
import base64
import json
import time

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1/webcam", tags=["webcam"])


# Webcam task state (encapsulated to avoid bare globals)
class _WebcamState:
    task: asyncio.Task | None = None


_webcam_state = _WebcamState()


@router.post(
    "/start",
    summary="Start webcam capture",
    description="Start the webcam capture service for face recognition",
)
async def start_webcam():
    """Start webcam capture service."""
    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if not settings.webcam_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webcam is not enabled in settings (WEBCAM_ENABLED=false)",
        )

    if _webcam_state.task is not None and not _webcam_state.task.done():
        return {
            "success": True,
            "message": "Webcam service is already running",
            "status": "running",
        }

    # Start webcam service
    _webcam_state.task = asyncio.create_task(webcam_service.run_capture_loop())

    return {
        "success": True,
        "message": "Webcam service started successfully",
        "status": "running",
        "camera_id": settings.webcam_device_id,
        "fps": settings.webcam_fps,
        "cooldown_seconds": settings.webcam_success_cooldown_seconds,
    }


@router.post(
    "/stop",
    summary="Stop webcam capture",
    description="Stop the webcam capture service",
)
async def stop_webcam():
    """Stop webcam capture service."""
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if _webcam_state.task is None or _webcam_state.task.done():
        return {
            "success": True,
            "message": "Webcam service is not running",
            "status": "stopped",
        }

    # Stop webcam service
    webcam_service.stop()

    # Wait for task to complete (with timeout)
    try:
        await asyncio.wait_for(_webcam_state.task, timeout=5.0)
    except TimeoutError:
        # Force cancel if it doesn't stop gracefully
        _webcam_state.task.cancel()

    _webcam_state.task = None

    return {
        "success": True,
        "message": "Webcam service stopped successfully",
        "status": "stopped",
    }


@router.get(
    "/status",
    summary="Get webcam status",
    description="Get the current status of the webcam capture service",
)
async def get_webcam_status():
    """Get webcam service status."""
    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    is_running = _webcam_state.task is not None and not _webcam_state.task.done()

    status_info = {
        "success": True,
        "status": "running" if is_running else "stopped",
        "enabled": settings.webcam_enabled,
        "camera_id": settings.webcam_device_id,
        "fps": settings.webcam_fps,
        "cooldown_seconds": settings.webcam_success_cooldown_seconds,
    }

    if is_running:
        status_info["in_cooldown"] = webcam_service.is_in_cooldown()
        status_info["cooldown_remaining"] = webcam_service.get_cooldown_remaining()
        if webcam_service.last_recognized_user:
            status_info["last_recognized_user"] = webcam_service.last_recognized_user

    return status_info


@router.get(
    "/stream",
    summary="Stream webcam feed",
    description="Server-Sent Events stream of webcam frames and recognition results",
)
async def stream_webcam():
    """
    Stream webcam frames and recognition results via Server-Sent Events.

    This endpoint provides a real-time video feed with recognition overlays
    for development and monitoring purposes.
    """
    import cv2

    from src.config.settings import settings
    from src.services.webcam_service import get_webcam_service

    webcam_service = get_webcam_service()

    if not settings.webcam_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webcam is not enabled",
        )

    async def event_generator():
        """Generate SSE events with frames and recognition results."""
        cap = cv2.VideoCapture(settings.webcam_device_id)

        if not cap.isOpened():
            yield f"event: error\ndata: {json.dumps({'error': 'Could not open camera'})}\n\n"
            return

        try:
            loop = asyncio.get_running_loop()
            while True:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                # Encode frame as JPEG (offloaded — cv2 would block the event loop)
                _, buffer = await loop.run_in_executor(None, cv2.imencode, ".jpg", frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Create event data
                event_data = {
                    "frame": frame_b64,
                    "timestamp": time.time(),
                    "camera_id": settings.webcam_device_id,
                    "in_cooldown": webcam_service.is_in_cooldown(),
                    "cooldown_remaining": webcam_service.get_cooldown_remaining(),
                }

                if webcam_service.last_recognized_user:
                    event_data["last_recognized_user"] = webcam_service.last_recognized_user

                # Send event
                yield f"event: frame\ndata: {json.dumps(event_data)}\n\n"

                # Control frame rate
                await asyncio.sleep(1.0 / settings.webcam_fps)

        except asyncio.CancelledError:
            pass
        finally:
            cap.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
