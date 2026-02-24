"""
Centralized provider configuration for MCP Server.

Loads Azure credentials and service configurations from environment
variables and provides singleton instances of provider configs.
"""

from functools import lru_cache
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from mmct.config.providers import (
    VideoAgentProviderConfig,
    IngestionProviderConfig,
    ImageAgentProviderConfig,
)
from mmct.providers.azure import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AISearchChapterProvider,
    AISearchKeyframesProvider,
    AISearchObjectCollectionProvider,
    AzureStorageProvider,
    AzureSpeechServiceProvider,
)
from mmct.providers.local import ClipImageEmbeddingProvider
from dotenv import load_dotenv, find_dotenv

_credentials = None


def get_credentials():
    """Get or create Azure credentials."""
    global _credentials
    if _credentials is None:
        _credentials = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())
        logger.info("Azure credentials initialized")
    return _credentials


class ProviderEnvSettings(BaseSettings):
    """
    Pydantic settings for loading provider configuration from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(), env_file_encoding="utf-8", extra="ignore"
    )

    # LLM Settings
    llm_endpoint: str = Field(...)
    llm_deployment_name: str = Field(...)
    llm_model_name: str = Field(...)
    llm_api_version: str = Field(...)

    # Embedding Service Settings
    embedding_service_endpoint: str = Field(...)
    embedding_service_deployment_name: str = Field(...)
    embedding_service_api_version: str = Field(...)

    # Search Service Settings
    search_endpoint: str = Field(...)
    chapter_index_name: str = Field(...)
    keyframes_index_name: str = Field(...)
    object_collection_index_name: str = Field(...)

    # Storage Settings
    storage_account_name: str = Field(...)
    keyframe_container_name: str = Field(...)

    # Speech Service Settings
    speech_service_resource_id: str = Field(...)
    speech_service_region: str = Field(...)


@lru_cache(maxsize=1)
def get_settings() -> ProviderEnvSettings:
    """Get settings singleton."""
    return ProviderEnvSettings()


@lru_cache(maxsize=1)
def get_image_agent_provider() -> ImageAgentProviderConfig:
    """Get ImageAgentProviderConfig singleton instance."""
    credentials = get_credentials()
    settings = get_settings()

    logger.info("Initializing ImageAgentProviderConfig")

    provider = ImageAgentProviderConfig(
        llm_provider=AzureLLMProvider(
            endpoint=settings.llm_endpoint,
            deployment_name=settings.llm_deployment_name,
            model_name=settings.llm_model_name,
            api_version=settings.llm_api_version,
            credentials=credentials,
        )
    )

    logger.info("ImageAgentProviderConfig initialized successfully")
    return provider


@lru_cache(maxsize=1)
def get_video_agent_provider() -> VideoAgentProviderConfig:
    """Get VideoAgentProviderConfig singleton instance."""
    credentials = get_credentials()
    settings = get_settings()

    logger.info("Initializing VideoAgentProviderConfig")

    provider = VideoAgentProviderConfig(
        llm_provider=AzureLLMProvider(
            endpoint=settings.llm_endpoint,
            deployment_name=settings.llm_deployment_name,
            model_name=settings.llm_model_name,
            api_version=settings.llm_api_version,
            credentials=credentials,
        ),
        embedding_provider=AzureEmbeddingProvider(
            endpoint=settings.embedding_service_endpoint,
            deployment_name=settings.embedding_service_deployment_name,
            api_version=settings.embedding_service_api_version,
            credentials=credentials,
        ),
        image_embedding_provider=ClipImageEmbeddingProvider(),
        vectordb_chapter=AISearchChapterProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.chapter_index_name,
            credentials=credentials,
        ),
        vectordb_object_registry=AISearchObjectCollectionProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.object_collection_index_name,
            credentials=credentials,
        ),
        vectordb_keyframes=AISearchKeyframesProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.keyframes_index_name,
            credentials=credentials,
        ),
        storage_provider=AzureStorageProvider(
            storage_account_name=settings.storage_account_name,
            keyframe_container_name=settings.keyframe_container_name,
            credentials=credentials,
        ),
    )

    logger.info("VideoAgentProviderConfig initialized successfully")
    return provider


@lru_cache(maxsize=1)
def get_ingestion_provider() -> IngestionProviderConfig:
    """Get IngestionProviderConfig singleton instance."""
    credentials = get_credentials()
    settings = get_settings()

    logger.info("Initializing IngestionProviderConfig")

    provider = IngestionProviderConfig(
        llm_provider=AzureLLMProvider(
            endpoint=settings.llm_endpoint,
            deployment_name=settings.llm_deployment_name,
            model_name=settings.llm_model_name,
            api_version=settings.llm_api_version,
            credentials=credentials,
        ),
        embedding_provider=AzureEmbeddingProvider(
            endpoint=settings.embedding_service_endpoint,
            deployment_name=settings.embedding_service_deployment_name,
            api_version=settings.embedding_service_api_version,
            credentials=credentials,
        ),
        image_embedding_provider=ClipImageEmbeddingProvider(),
        vectordb_chapter=AISearchChapterProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.chapter_index_name,
            credentials=credentials,
        ),
        vectordb_object_registry=AISearchObjectCollectionProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.object_collection_index_name,
            credentials=credentials,
        ),
        vectordb_keyframes=AISearchKeyframesProvider(
            endpoint=settings.search_endpoint,
            index_name=settings.keyframes_index_name,
            credentials=credentials,
        ),
        storage_provider=AzureStorageProvider(
            storage_account_name=settings.storage_account_name,
            keyframe_container_name=settings.keyframe_container_name,
            credentials=credentials,
        ),
        transcription_provider=AzureSpeechServiceProvider(
            speech_service_resource_id=settings.speech_service_resource_id,
            speech_service_region=settings.speech_service_region,
            llm_provider=AzureLLMProvider(
                endpoint=settings.llm_endpoint,
                deployment_name=settings.llm_deployment_name,
                model_name=settings.llm_model_name,
                api_version=settings.llm_api_version,
                credentials=credentials,
            ),
            credentials=credentials,
        ),
    )

    logger.info("IngestionProviderConfig initialized successfully")
    return provider
