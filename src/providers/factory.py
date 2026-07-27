import logging
from typing import cast

from src.config.settings import settings
from src.providers import registry
from src.providers.aws_rekognition import AWSRekognitionProvider
from src.providers.base import FaceProvider
from src.providers.insightface_provider import InsightFaceProvider
from src.providers.interfaces import CloudMatchProvider, EmbeddingProvider

logger = logging.getLogger(__name__)

_local_cache: EmbeddingProvider | None = None
_cloud_cache: CloudMatchProvider | None = None


class ProviderFactory:
    """Factory for creating face recognition provider instances."""

    _providers: dict[str, type[FaceProvider]] = {
        "aws_rekognition": AWSRekognitionProvider,
        "insightface": InsightFaceProvider,
        # Future providers can be added here:
        # "azure_face": AzureFaceProvider,
        # "google_vision": GoogleVisionProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str | None = None) -> FaceProvider:
        """
        Create a face provider instance.

        Uses cached singletons for known providers to avoid reloading heavy models.

        Args:
            provider_name: Name of the provider (defaults to the provider that
                matches settings.recognition_mode)

        Returns:
            FaceProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name is None:
            provider_name = (
                "aws_rekognition" if settings.recognition_mode == "cloud" else "insightface"
            )

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


def get_local_provider() -> EmbeddingProvider:
    """Get the configured local embedding provider instance (singleton)."""
    global _local_cache

    if _local_cache is None:
        provider_class = registry.resolve_local(settings.local_provider)
        logger.info(
            f"Initializing local provider (model: {settings.insightface_model}, "
            f"det_size: {settings.insightface_det_size})"
        )
        _local_cache = provider_class(
            model_name=settings.insightface_model,
            det_size=(settings.insightface_det_size, settings.insightface_det_size),
            ctx_id=settings.insightface_ctx_id,  # CPU for now, can add GPU support later
        )
        logger.info("Local provider cached successfully")

    return _local_cache


def get_cloud_provider() -> CloudMatchProvider:
    """Get the configured cloud match provider instance (singleton)."""
    global _cloud_cache

    if _cloud_cache is None:
        provider_class = registry.resolve_cloud(settings.cloud_provider)
        logger.info("Initializing cloud provider")
        _cloud_cache = provider_class()
        logger.info("Cloud provider cached successfully")

    return _cloud_cache


def get_insightface_provider() -> InsightFaceProvider:
    """Get InsightFace provider instance (singleton); thin wrapper over get_local_provider()."""
    return cast(InsightFaceProvider, get_local_provider())


def get_aws_provider() -> AWSRekognitionProvider:
    """Get AWS Rekognition provider instance (singleton); thin wrapper over get_cloud_provider()."""
    return cast(AWSRekognitionProvider, get_cloud_provider())


def clear_provider_cache() -> None:
    """Reset all cached provider singletons."""
    global _local_cache, _cloud_cache
    _local_cache = None
    _cloud_cache = None
