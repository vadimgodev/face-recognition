import logging

from src.config.settings import settings
from src.providers.aws_rekognition import AWSRekognitionProvider
from src.providers.base import FaceProvider
from src.providers.insightface_provider import InsightFaceProvider

logger = logging.getLogger(__name__)

# Module-level caches for provider singletons
# This prevents reloading heavy models (InsightFace) on every request
_insightface_cache: InsightFaceProvider = None
_aws_cache: AWSRekognitionProvider = None


class ProviderFactory:
    """Factory for creating face recognition provider instances."""

    _providers = {
        "aws_rekognition": AWSRekognitionProvider,
        "insightface": InsightFaceProvider,
        # Future providers can be added here:
        # "azure_face": AzureFaceProvider,
        # "google_vision": GoogleVisionProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str = None) -> FaceProvider:
        """
        Create a face provider instance.

        Uses cached singletons for known providers to avoid reloading heavy models.

        Args:
            provider_name: Name of the provider (defaults to settings.face_provider)

        Returns:
            FaceProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name is None:
            provider_name = settings.face_provider

        if provider_name == "insightface":
            return get_insightface_provider()
        elif provider_name == "aws_rekognition":
            return get_aws_provider()

        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported provider: {provider_name}. " f"Available providers: {available}"
            )
        return provider_class()

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())


# Convenience functions for getting providers
def get_face_provider() -> FaceProvider:
    """Get the configured face provider instance."""
    return ProviderFactory.create_provider()


def get_insightface_provider() -> InsightFaceProvider:
    """
    Get InsightFace provider instance (singleton).

    Returns cached instance to avoid reloading heavy models on every request.
    Model loading takes 3-5 seconds, so caching provides massive speedup.
    """
    global _insightface_cache

    if _insightface_cache is None:
        logger.info(
            f"Initializing InsightFace provider (model: {settings.insightface_model}, "
            f"det_size: {settings.insightface_det_size})"
        )
        _insightface_cache = InsightFaceProvider(
            model_name=settings.insightface_model,
            det_size=(settings.insightface_det_size, settings.insightface_det_size),
            ctx_id=settings.insightface_ctx_id,  # CPU for now, can add GPU support later
        )
        logger.info("InsightFace provider cached successfully")

    return _insightface_cache


def get_aws_provider() -> AWSRekognitionProvider:
    """
    Get AWS Rekognition provider instance (singleton).

    Returns cached instance for consistency.
    """
    global _aws_cache

    if _aws_cache is None:
        logger.info("Initializing AWS Rekognition provider")
        _aws_cache = AWSRekognitionProvider()
        logger.info("AWS Rekognition provider cached successfully")

    return _aws_cache
