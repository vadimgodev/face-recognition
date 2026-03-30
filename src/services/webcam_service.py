"""Webcam capture service for face recognition."""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO

import cv2
import httpx
import numpy as np
from PIL import Image

from src.config.settings import settings
from src.services.door_service import door_service
from src.utils.access_logger import access_logger

logger = logging.getLogger(__name__)


class WebcamService:
    """Service for capturing and processing webcam frames."""

    def __init__(
        self,
        camera_id: int = 0,
        api_base_url: str = None,
    ):
        """
        Initialize webcam service.

        Args:
            camera_id: Webcam device ID
            api_base_url: Base URL for the face recognition API (uses settings if None)
        """
        self.camera_id = camera_id
        if api_base_url is None:
            api_base_url = settings.webcam_api_url
        self.api_base_url = api_base_url
        self.recognize_url = f"{api_base_url}/api/v1/faces/recognize-multiple"

        self.capture_interval = 1.0 / settings.webcam_fps  # Convert FPS to interval
        self.cooldown_seconds = settings.webcam_success_cooldown_seconds

        self.is_running = False
        self.last_success_time: float | None = None
        self.last_recognized_user: str | None = None

        # Initialize video capture
        self.cap: cv2.VideoCapture | None = None

        # InsightFace for local face detection (optional, lighter than full recognition)
        self._init_face_detector()

        # Liveness detection (anti-spoofing)
        self._init_liveness_detector()

    def _init_face_detector(self):
        """Initialize face detector for pre-filtering frames."""
        try:
            import warnings

            import insightface

            # Suppress warnings during detector initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Use lightweight detection only
                self.face_detector = insightface.app.FaceAnalysis(
                    name=settings.insightface_model,
                    providers=["CPUExecutionProvider"],
                )
                self.face_detector.prepare(
                    ctx_id=-1,  # CPU
                    det_size=(320, 320),  # Smaller size for faster detection
                )
            logger.info("Face detector initialized successfully")
        except Exception:
            # This is not critical - we can still work without local face detection
            # All frames will be sent to the API for recognition
            logger.info("Running without local face detection (all frames will be sent to API)")
            self.face_detector = None

    def _init_liveness_detector(self):
        """Initialize liveness detector for anti-spoofing."""
        if not settings.liveness_enabled:
            logger.info("Liveness detection disabled in settings")
            self.liveness_provider = None
            return

        try:
            from src.providers.silent_face_liveness import get_liveness_provider

            self.liveness_provider = get_liveness_provider(
                device_id=settings.liveness_device_id,
                model_dir=settings.liveness_model_dir,
                detector_path=settings.liveness_detector_path,
            )
            logger.info(
                f"✅ Liveness detection enabled ({self.liveness_provider.provider_name}, "
                f"threshold: {settings.liveness_threshold})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize liveness detector: {e}", exc_info=True)
            # SECURITY: Fail hard if liveness is required
            if settings.liveness_enabled:
                raise RuntimeError(
                    f"SECURITY CRITICAL: Liveness detection is ENABLED in settings but failed to initialize. "
                    f"Cannot start webcam service without liveness protection. "
                    f"Error: {e}\n"
                    f"Fix: Ensure model files exist in {settings.liveness_model_dir} and {settings.liveness_detector_path}, "
                    f"or set LIVENESS_ENABLED=false to disable liveness detection."
                ) from e
            # If liveness is explicitly disabled, allow None
            logger.warning(
                "Liveness detection disabled, continuing without anti-spoofing protection"
            )
            self.liveness_provider = None

    def start_capture(self) -> bool:
        """
        Start video capture from webcam.

        Returns:
            True if capture started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_id}")
                return False

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Warm up camera - discard first few frames
            # First frames are often black/corrupted while camera initializes
            logger.info(f"Camera {self.camera_id} opened, warming up...")
            for i in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read warm-up frame {i+1}/5")
                time.sleep(0.1)  # 100ms between warm-up frames

            logger.info(f"Camera {self.camera_id} ready")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera capture: {e}")
            return False

    def stop_capture(self):
        """Stop video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera capture stopped")

    def capture_frame(self) -> np.ndarray | None:
        """
        Capture a single frame from webcam.

        Returns:
            Frame as numpy array (BGR format) or None if capture failed
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera not opened")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        return frame

    def has_face(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains a face.

        Args:
            frame: Frame as numpy array (BGR format)

        Returns:
            True if face detected, False otherwise
        """
        if self.face_detector is None:
            # If no detector, assume face is present
            return True

        try:
            # Convert BGR to RGB for InsightFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.get(rgb_frame)
            return len(faces) > 0
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return True  # Assume face present on error

    async def check_liveness(self, image_bytes: bytes) -> tuple[bool, float, str]:
        """
        Check if image contains a real live person (anti-spoofing).

        Args:
            image_bytes: Image data as bytes

        Returns:
            Tuple of (is_real, confidence, spoofing_type)
        """
        if self.liveness_provider is None:
            # SECURITY: Verify this is expected
            if settings.liveness_enabled:
                # Provider is None but liveness is enabled - this indicates initialization failure
                # This should never happen if _init_liveness_detector() worked correctly
                raise RuntimeError(
                    "SECURITY CRITICAL: Liveness detection is ENABLED but provider is None. "
                    "This indicates initialization failure. Cannot proceed without liveness check."
                )
            # Liveness detection explicitly disabled, assume real
            return True, 1.0, "real"

        try:
            result = await self.liveness_provider.check_liveness(
                image_bytes=image_bytes,
                threshold=settings.liveness_threshold,
            )

            return result.is_real, result.confidence, result.spoofing_type.value

        except Exception as e:
            logger.error(f"Liveness check error: {e}")
            # FAIL CLOSED - block on errors for maximum security
            # This prevents unclear/low-quality frames from bypassing liveness check
            logger.warning("🚫 Liveness check failed due to error - BLOCKING frame")
            return False, 0.0, "error"

    def frame_to_jpeg_bytes(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """
        Convert frame to JPEG bytes.

        Args:
            frame: Frame as numpy array (BGR format)
            quality: JPEG quality (0-100)

        Returns:
            JPEG image as bytes
        """
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return buffer.getvalue()

    async def recognize_face(self, image_bytes: bytes) -> dict | None:
        """
        Send image to recognition API.

        Args:
            image_bytes: JPEG image as bytes

        Returns:
            Recognition result dict or None if request failed
        """
        try:
            # Get auth credentials from environment
            from os import environ

            basic_auth_user = environ.get("BASIC_AUTH_USERNAME", "")
            basic_auth_pass = environ.get("BASIC_AUTH_PASSWORD", "")
            api_token = environ.get("SECRET_KEY", settings.secret_key)

            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
                data = {
                    "max_results_per_face": 1,
                    "confidence_threshold": 0.6,
                }

                response = await client.post(
                    self.recognize_url,
                    files=files,
                    data=data,
                    auth=(basic_auth_user, basic_auth_pass),  # Basic Auth for Traefik
                    headers={"x-face-token": api_token},  # API token
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            # Log the actual error response from the API
            try:
                error_detail = e.response.text
                logger.error(f"Recognition API returned {e.response.status_code}: {error_detail}")
            except Exception:
                logger.error(f"Recognition API request failed: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"Recognition API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during recognition: {e}")
            return None

    def is_in_cooldown(self) -> bool:
        """Check if service is in cooldown period."""
        if self.last_success_time is None:
            return False
        elapsed = time.time() - self.last_success_time
        return elapsed < self.cooldown_seconds

    def get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self.last_success_time is None:
            return 0.0
        elapsed = time.time() - self.last_success_time
        remaining = self.cooldown_seconds - elapsed
        return max(0.0, remaining)

    async def process_recognition_result(self, result: dict):
        """
        Process recognition result and trigger door unlock if authorized.

        Args:
            result: Recognition result from multi-face API
        """
        if not result.get("success", False):
            # Recognition failed
            access_logger.log_recognition_event(
                result="failure",
                confidence=0.0,
                execution_time_ms=int(result.get("execution_time", 0) * 1000),
                camera_id=self.camera_id,
            )
            return

        detected_faces = result.get("detected_faces", [])
        if not detected_faces:
            # No faces detected
            access_logger.log_recognition_event(
                result="failure",
                confidence=0.0,
                execution_time_ms=int(result.get("execution_time", 0) * 1000),
                camera_id=self.camera_id,
            )
            return

        # Find best match across all detected faces
        best_match = None
        best_similarity = 0.0
        best_match_face_data = None
        best_match_processor = None

        for detected_face in detected_faces:
            matches = detected_face.get("matches", [])
            if matches:
                # Get the best match for this face (first one, already sorted by similarity)
                match = matches[0]
                similarity = match.get("similarity", 0.0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = match
                    best_match_face_data = match.get("face", {})
                    best_match_processor = match.get(
                        "processor", result.get("processor", "unknown")
                    )

        # Log info about all detected faces
        total_faces = len(detected_faces)
        faces_with_matches = sum(1 for f in detected_faces if f.get("total_matches", 0) > 0)
        logger.info(
            f"Multi-face: detected {total_faces} face(s), " f"{faces_with_matches} recognized"
        )

        if not best_match:
            # No matches found across all faces
            access_logger.log_recognition_event(
                result="failure",
                confidence=0.0,
                execution_time_ms=int(result.get("execution_time", 0) * 1000),
                camera_id=self.camera_id,
            )
            return

        # Get user info from best match
        user_name = best_match_face_data.get("user_name", "Unknown")
        user_email = best_match_face_data.get("user_email")

        # Try to unlock door
        unlock_success, door_action = await door_service.unlock_if_authorized(
            user_name, best_similarity
        )

        # Log the event with match-specific processor (shows AWS usage)
        access_logger.log_recognition_event(
            result="success",
            confidence=best_similarity,
            execution_time_ms=int(result.get("execution_time", 0) * 1000),
            user_name=user_name,
            user_email=user_email,
            processor=best_match_processor,  # This will show "antelopev2" or "antelopev2+aws"
            door_action=door_action,
            camera_id=self.camera_id,
            detection_time_ms=int(result.get("detection_time", 0) * 1000),
            recognition_time_ms=int(result.get("recognition_time", 0) * 1000),
        )

        # If door unlocked successfully, start cooldown
        if door_action == "unlocked":
            self.last_success_time = time.time()
            self.last_recognized_user = user_name
            logger.info(
                f"Starting cooldown for {self.cooldown_seconds}s after recognizing {user_name}"
            )

    async def run_capture_loop(self):
        """
        Main capture loop - sequential processing with cooldown.

        This runs continuously until stopped, capturing frames,
        detecting faces, sending to recognition API, and processing results.
        """
        if not self.start_capture():
            logger.error("Failed to start camera capture")
            return

        self.is_running = True
        logger.info("Webcam capture loop started")

        try:
            while self.is_running:
                loop_start_time = time.time()

                # Check cooldown period
                if self.is_in_cooldown():
                    cooldown_remaining = self.get_cooldown_remaining()
                    if settings.access_log_include_cooldown_events:
                        access_logger.log_cooldown_event(
                            cooldown_remaining_seconds=cooldown_remaining,
                            last_recognized_user=self.last_recognized_user,
                        )
                    await asyncio.sleep(0.5)
                    continue

                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # Check if frame contains a face
                if not self.has_face(frame):
                    await asyncio.sleep(self.capture_interval)
                    continue

                # Convert frame to JPEG bytes
                image_bytes = self.frame_to_jpeg_bytes(frame, quality=85)

                # Check liveness (anti-spoofing) before sending to recognition API
                if self.liveness_provider is not None:
                    is_real, liveness_confidence, spoofing_type = await self.check_liveness(
                        image_bytes
                    )

                    if not is_real:
                        # Spoofing attempt or detection error - log and skip this frame
                        if spoofing_type == "error":
                            # Detection error (face not detected clearly)
                            logger.warning(
                                "🚫 Liveness check failed (detection error) - Frame BLOCKED "
                                "(fail-closed security policy)"
                            )
                            access_logger.log_recognition_event(
                                result="liveness_detection_failed",
                                confidence=0.0,
                                execution_time_ms=0,
                                camera_id=self.camera_id,
                                notes="Face not detected with sufficient confidence for liveness check",
                            )
                        else:
                            # Actual spoofing detected
                            logger.warning(
                                f"🚫 Spoofing detected! Type: {spoofing_type}, "
                                f"Confidence: {liveness_confidence:.3f}, "
                                f"Threshold: {settings.liveness_threshold}"
                            )
                            access_logger.log_recognition_event(
                                result="spoofing_detected",
                                confidence=liveness_confidence,
                                execution_time_ms=0,
                                camera_id=self.camera_id,
                                notes=f"Spoofing type: {spoofing_type}",
                            )
                        # Skip recognition for this frame
                        await asyncio.sleep(self.capture_interval)
                        continue

                    # Log successful liveness check (optional, can be noisy)
                    logger.debug(
                        f"✅ Liveness check passed (confidence: {liveness_confidence:.3f})"
                    )

                # Send to recognition API (synchronous - wait for response)
                result = await self.recognize_face(image_bytes)

                # Process result
                if result:
                    await self.process_recognition_result(result)

                # Wait remaining interval time
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, self.capture_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
            access_logger.log_error(f"Capture loop error: {e}")
        finally:
            self.stop_capture()
            self.is_running = False
            logger.info("Webcam capture loop stopped")

    def stop(self):
        """Stop the capture loop."""
        self.is_running = False


# Global webcam service instance (initialized lazily to avoid event loop issues)
webcam_service = None


def get_webcam_service():
    """Get or create the global webcam service instance."""
    global webcam_service
    if webcam_service is None:
        webcam_service = WebcamService(
            camera_id=settings.webcam_device_id,
            api_base_url=settings.webcam_api_url,
        )
    return webcam_service
