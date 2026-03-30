"""
Startup validation for critical system requirements.

This module provides validation functions that should be run at application
startup to ensure all required components are properly configured.
"""

import logging
from pathlib import Path

from src.config.settings import settings

logger = logging.getLogger(__name__)


def validate_liveness_configuration() -> tuple[bool, list[str]]:
    """
    Validate liveness detection configuration and model files.

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if configuration is valid
        - error_messages: List of validation error messages (empty if valid)
    """
    errors = []

    # If liveness is disabled, no validation needed
    if not settings.liveness_enabled:
        logger.info("Liveness detection disabled, skipping validation")
        return True, []

    logger.info("Validating liveness detection configuration...")

    # Check provider
    if settings.liveness_provider not in ["silent_face"]:
        errors.append(
            f"Invalid liveness provider: {settings.liveness_provider}. " f"Supported: silent_face"
        )

    # Check threshold
    if not (0.0 <= settings.liveness_threshold <= 1.0):
        errors.append(
            f"Invalid liveness threshold: {settings.liveness_threshold}. "
            f"Must be between 0.0 and 1.0"
        )

    # Validate model directories exist
    model_dir = Path(settings.liveness_model_dir)
    detector_path = Path(settings.liveness_detector_path)

    if not model_dir.exists():
        errors.append(
            f"Liveness model directory not found: {model_dir}. "
            f"Set LIVENESS_MODEL_DIR or create the directory."
        )
    elif not model_dir.is_dir():
        errors.append(f"Liveness model path is not a directory: {model_dir}")

    if not detector_path.exists():
        errors.append(
            f"Liveness detector path not found: {detector_path}. "
            f"Set LIVENESS_DETECTOR_PATH or create the directory."
        )
    elif not detector_path.is_dir():
        errors.append(f"Liveness detector path is not a directory: {detector_path}")

    # Validate face detector availability (InsightFace OR RetinaFace)
    # Primary: InsightFace buffalo (automatically downloaded)
    # Fallback: RetinaFace caffemodel (optional legacy detector)
    has_insightface = False
    has_retinaface = False

    # Check if InsightFace is available (preferred detector)
    try:
        import insightface  # noqa: F401

        has_insightface = True
        logger.info("✅ InsightFace available for face detection (primary)")
    except ImportError:
        logger.warning("InsightFace not available, checking for RetinaFace fallback...")

    # Check for RetinaFace models (fallback detector)
    if detector_path.exists():
        caffemodel = detector_path / "Widerface-RetinaFace.caffemodel"
        deploy_prototxt = detector_path / "deploy.prototxt"

        if caffemodel.exists() and deploy_prototxt.exists():
            # Validate file sizes if RetinaFace is present
            caffemodel_size = caffemodel.stat().st_size
            deploy_size = deploy_prototxt.stat().st_size

            if caffemodel_size == 0:
                errors.append(f"Face detector caffemodel is empty: {caffemodel}")
            elif caffemodel_size < 1_000_000:  # Less than 1MB
                errors.append(
                    f"Face detector caffemodel seems too small: {caffemodel} "
                    f"({caffemodel_size} bytes). "
                    f"Expected size: ~5-10MB. File may be corrupted or incomplete."
                )

            if deploy_size == 0:
                errors.append(f"Face detector prototxt is empty: {deploy_prototxt}")

            # Only mark as available if no errors with the files
            if caffemodel_size > 0 and deploy_size > 0:
                has_retinaface = True
                logger.info("✅ RetinaFace models available (fallback detector)")

    # Require at least one detector
    if not has_insightface and not has_retinaface:
        errors.append(
            f"No face detector available for liveness detection. "
            f"Install InsightFace (pip install insightface) OR "
            f"download Widerface-RetinaFace.caffemodel and place it in {detector_path}"
        )

    # Validate anti-spoofing models exist
    if model_dir.exists():
        pth_files = list(model_dir.glob("*.pth"))
        if not pth_files:
            errors.append(
                f"No anti-spoofing models (.pth files) found in {model_dir}. "
                f"Download MiniFASNet models and place them in this directory."
            )
        else:
            logger.info(f"Found {len(pth_files)} anti-spoofing model(s)")

            # Validate each model file
            for pth_file in pth_files:
                if pth_file.stat().st_size == 0:
                    errors.append(f"Anti-spoofing model is empty: {pth_file}")
                elif pth_file.stat().st_size < 100_000:  # Less than 100KB
                    errors.append(
                        f"Anti-spoofing model seems too small: {pth_file} "
                        f"({pth_file.stat().st_size} bytes)"
                    )

    # Return validation result
    is_valid = len(errors) == 0

    if is_valid:
        logger.info("✅ Liveness detection configuration is valid")
    else:
        logger.error("❌ Liveness detection configuration validation failed:")
        for error in errors:
            logger.error(f"   - {error}")

    return is_valid, errors


def validate_startup_requirements(fail_on_error: bool = True) -> bool:
    """
    Validate all startup requirements.

    Args:
        fail_on_error: If True, raise exception on validation failure

    Returns:
        True if all validations passed

    Raises:
        RuntimeError: If validation fails and fail_on_error is True
    """
    logger.info("=" * 70)
    logger.info("Running startup validation checks...")
    logger.info("=" * 70)

    all_valid = True
    all_errors = []

    # Validate liveness configuration
    liveness_valid, liveness_errors = validate_liveness_configuration()
    if not liveness_valid:
        all_valid = False
        all_errors.extend(liveness_errors)

    # Add more validation checks here as needed
    # - Database connectivity
    # - AWS credentials (if using AWS provider)
    # - Storage backend availability
    # etc.

    logger.info("=" * 70)

    if all_valid:
        logger.info("✅ All startup validation checks passed")
        logger.info("=" * 70)
        return True
    else:
        logger.error("❌ Startup validation failed with errors:")
        for error in all_errors:
            logger.error(f"   - {error}")
        logger.info("=" * 70)

        if fail_on_error:
            raise RuntimeError(
                f"Startup validation failed with {len(all_errors)} error(s). "
                f"Fix the issues above or disable liveness detection (LIVENESS_ENABLED=false)."
            )

        return False


if __name__ == "__main__":
    # Allow running this module directly for validation
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        validate_startup_requirements(fail_on_error=True)
        print("\n✅ All validation checks passed!")
    except RuntimeError as e:
        print(f"\n❌ Validation failed: {e}")
        exit(1)
