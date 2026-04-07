"""Centralised provider configuration for MMCT Agent.

Loads Azure credentials and service configuration from environment variables
and exposes singleton getters for all provider types used across the application.

Provider hierarchy:
  - VideoAgentProviderConfig   : LLM + embeddings + vector search + blob storage
  - ImageAgentProviderConfig   : LLM only (vision calls handled by the LLM provider)
  - IngestionProviderConfig    : All of the above + transcription + graph store
  - get_neo4j_query_provider() : Neo4j graph query interface (catalog, video listing)
  - get_text_embedding_provider() / get_image_embedding_provider(): local embedding models
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
from mmct.providers.custom_providers import (
    Neo4jGraphStoreProvider,
    FastEmbedQdrantCLIPEmbeddingProvider,
)
from mmct.v5.query.neo4j_provider import Neo4jQueryProvider

from app.config.credentials import resolve_credentials


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class ProviderEnvSettings(BaseSettings):
    """Environment-driven configuration for all MMCT providers.

    All fields are loaded from environment variables (or a .env file).
    Required fields have no default; the application will fail fast on startup
    if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_endpoint: str = Field(...)
    llm_deployment_name: str = Field(...)
    llm_model_name: str = Field(...)
    llm_api_version: str = Field(...)

    # Text embedding
    embedding_service_endpoint: str = Field(...)
    embedding_service_deployment_name: str = Field(...)
    embedding_service_api_version: str = Field(...)

    # Azure AI Search
    search_endpoint: str = Field(...)
    chapter_index_name: str = Field(...)
    keyframes_index_name: str = Field(...)
    object_collection_index_name: str = Field(...)

    # Blob storage
    storage_account_name: str = Field(...)
    keyframe_container_name: str = Field(...)
    transcript_container_name: str = Field(default="video-transcript-lively")

    # Azure Speech
    speech_service_resource_id: str = Field(...)
    speech_service_region: str = Field(...)

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")


@lru_cache(maxsize=1)
def get_settings() -> ProviderEnvSettings:
    """Return the singleton settings instance."""
    return ProviderEnvSettings()


# ---------------------------------------------------------------------------
# Provider singletons — agent providers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_image_agent_provider() -> ImageAgentProviderConfig:
    """Return the singleton ImageAgentProviderConfig.

    Configures the LLM provider used for image analysis tasks.
    """
    credentials = resolve_credentials()
    settings = get_settings()
    logger.info("Initialising ImageAgentProviderConfig")
    provider = ImageAgentProviderConfig(
        llm_provider=AzureLLMProvider(
            endpoint=settings.llm_endpoint,
            deployment_name=settings.llm_deployment_name,
            model_name=settings.llm_model_name,
            api_version=settings.llm_api_version,
            credentials=credentials,
        )
    )
    logger.info("ImageAgentProviderConfig ready")
    return provider


@lru_cache(maxsize=1)
def get_video_agent_provider() -> VideoAgentProviderConfig:
    """Return the singleton VideoAgentProviderConfig.

    Configures all providers required for video question-answering:
    LLM, text/image embeddings, vector search indexes, and blob storage.
    """
    credentials = resolve_credentials()
    settings = get_settings()
    logger.info("Initialising VideoAgentProviderConfig")
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
        image_embedding_provider=FastEmbedQdrantCLIPEmbeddingProvider(),
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
    logger.info("VideoAgentProviderConfig ready")
    return provider


@lru_cache(maxsize=1)
def get_ingestion_provider() -> IngestionProviderConfig:
    """Return the singleton IngestionProviderConfig.

    Configures all providers needed for the video ingestion pipeline:
    LLM, embeddings, vector search, blob storage, speech transcription,
    and the Neo4j graph store.
    """
    credentials = resolve_credentials()
    settings = get_settings()
    logger.info("Initialising IngestionProviderConfig")
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
        image_embedding_provider=FastEmbedQdrantCLIPEmbeddingProvider(),
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
            credentials=credentials,
            llm_provider=AzureLLMProvider(
                endpoint=settings.llm_endpoint,
                deployment_name=settings.llm_deployment_name,
                model_name=settings.llm_model_name,
                api_version=settings.llm_api_version,
                credentials=credentials,
            ),
        ),
        graph_store_provider=Neo4jGraphStoreProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) if settings.neo4j_password else None,
    )
    logger.info("IngestionProviderConfig ready")
    return provider


# ---------------------------------------------------------------------------
# Provider singletons — graph query & local embeddings
# ---------------------------------------------------------------------------

_neo4j_query_provider: Optional[Neo4jQueryProvider] = None
_text_embedding_provider = None
_image_embedding_provider = None


def get_neo4j_query_provider() -> Neo4jQueryProvider:
    """Return the singleton Neo4j query provider.

    Used for graph-backed operations: video catalog generation, video listing,
    and temporal graph search.

    Raises:
        ValueError: If NEO4J_PASSWORD is not configured.
    """
    global _neo4j_query_provider
    if _neo4j_query_provider is None:
        settings = get_settings()
        if not settings.neo4j_password:
            raise ValueError(
                "Neo4j password not configured. Set the NEO4J_PASSWORD environment variable."
            )
        _neo4j_query_provider = Neo4jQueryProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        logger.info(f"Neo4j query provider initialised: {settings.neo4j_uri}")
    return _neo4j_query_provider


def get_text_embedding_provider():
    """Return the singleton local text embedding provider (BGE-small, 384-dim).

    Uses the same model as the ingestion pipeline to ensure embedding space
    consistency between indexed content and query vectors.
    """
    global _text_embedding_provider
    if _text_embedding_provider is None:
        from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
        _text_embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        logger.info("FastEmbedBGEsmallEmbeddingProvider initialised (384-dim)")
    return _text_embedding_provider


def get_image_embedding_provider():
    """Return the singleton local image embedding provider (QdrantCLIP, 512-dim).

    Uses the same model as the ingestion pipeline to ensure embedding space
    consistency between indexed keyframes and query images.
    """
    global _image_embedding_provider
    if _image_embedding_provider is None:
        _image_embedding_provider = FastEmbedQdrantCLIPEmbeddingProvider()
        logger.info("FastEmbedQdrantCLIPEmbeddingProvider initialised (512-dim)")
    return _image_embedding_provider
