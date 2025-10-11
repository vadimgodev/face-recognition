#!/usr/bin/env python3
"""Standalone webcam capture daemon for face recognition."""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.config.settings import settings
from src.services.webcam_service import get_webcam_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class WebcamDaemon:
    """Daemon for running webcam capture service."""

    def __init__(self, camera_id: int = 0, mode: str = "daemon"):
        """
        Initialize webcam daemon.

        Args:
            camera_id: Webcam device ID
            mode: Operating mode ('daemon' or 'dev')
        """
        self.camera_id = camera_id
        self.mode = mode
        self.should_stop = False

    def setup_signal_handlers(self, loop):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum):
            logger.info(f"Received signal {signum}, shutting down...")
            self.should_stop = True

        loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(signal.SIGINT))
        loop.add_signal_handler(signal.SIGTERM, lambda: signal_handler(signal.SIGTERM))

    async def run(self):
        """Run the daemon."""
        loop = asyncio.get_event_loop()
        self.setup_signal_handlers(loop)

        logger.info("=" * 60)
        logger.info("Face Recognition Webcam Daemon")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Camera ID: {self.camera_id}")
        logger.info(f"Capture Rate: {settings.webcam_fps} FPS")
        logger.info(
            f"Cooldown After Success: {settings.webcam_success_cooldown_seconds}s"
        )
        logger.info(
            f"Door Unlock Provider: {settings.door_unlock_provider}"
        )
        logger.info(
            f"Unlock Confidence Threshold: {settings.door_unlock_confidence_threshold}"
        )
        logger.info(
            f"Access Log Output: {settings.access_log_output}"
        )
        logger.info("")
        logger.info("Security Settings:")
        if settings.liveness_enabled:
            logger.info(
                f"  ✅ Liveness Detection: ENABLED "
                f"(threshold: {settings.liveness_threshold}, "
                f"provider: {settings.liveness_provider})"
            )
        else:
            logger.info(
                f"  ⚠️  Liveness Detection: DISABLED "
                f"(spoofing attacks may succeed)"
            )
        logger.info("=" * 60)

        # Get webcam service instance
        webcam = get_webcam_service()
        webcam.camera_id = self.camera_id

        # Create capture task
        capture_task = asyncio.create_task(webcam.run_capture_loop())

        # Wait for shutdown signal
        try:
            while not self.should_stop and not capture_task.done():
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        # Stop webcam service
        webcam.stop()

        # Wait for capture task to complete
        if not capture_task.done():
            try:
                await asyncio.wait_for(capture_task, timeout=5.0)
            except asyncio.TimeoutError:
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass

        logger.info("Daemon shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Webcam capture daemon for face recognition"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=settings.webcam_device_id,
        help="Webcam device ID (default: from settings)",
    )
    parser.add_argument(
        "--mode",
        choices=["daemon", "dev"],
        default="daemon",
        help="Operating mode: daemon (production) or dev (development)",
    )

    args = parser.parse_args()

    if not settings.webcam_enabled:
        logger.error("Webcam is not enabled in settings (WEBCAM_ENABLED=false)")
        logger.error("Set WEBCAM_ENABLED=true in .env to enable webcam capture")
        sys.exit(1)

    # Create and run daemon
    daemon = WebcamDaemon(camera_id=args.camera, mode=args.mode)

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
