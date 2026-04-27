"""
Centralized provider configuration for MMCTAgent.

Loads Azure service credentials and configuration from environment variables
and exposes singleton factory functions for all concrete providers available
in this repo.

Credential strategy: ChainedTokenCredential tries AzureCliCredential first
(local dev / CI with az login), then falls back to DefaultAzureCredential
(managed identity, workload identity, environment variables, etc.).
"""

from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Optional

import os
from azure.identity import ChainedTokenCredential, AzureCliCredential, DefaultAzureCredential, ManagedIdentityCredential
from dotenv import find_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mmct.providers.base import BaseTranscriptionProvider
from mmct.providers.azure_providers import (
    AzureLLMProvider,
    AzureReasoningLLMProvider,
    AzureEmbeddingProvider,
    AzureSpeechServiceProvider,
    AzureStorageProvider,
    WhisperTranscriptionProvider,
)
from mmct.providers.custom_providers import (
    Neo4jQueryProvider,
    FastEmbedQdrantCLIPEmbeddingProvider,
    FastEmbedBGEsmallEmbeddingProvider,
    FastEmbedArcticEmbeddingProvider,
)
from mmct.providers.custom_providers.neo4j_graph_store_provider import Neo4jGraphStoreProvider
from mmct.image_pipeline.config import ImageAgentProviderConfig


@dataclass(frozen=True)
class IngestionProviders:
    """Provider bundle consumed by the ingestion pipeline StepContext."""

    llm_provider: AzureLLMProvider
    transcription_provider: BaseTranscriptionProvider
    storage_provider: AzureStorageProvider
    graph_store_provider: Optional[Neo4jGraphStoreProvider] = None
    llm_config: Optional[dict] = None


@dataclass(frozen=True)
class QueryPipelineProviders:
    """Provider bundle consumed by graph/state query pipelines."""

    model_client: Any
    neo4j_provider: Optional[Neo4jQueryProvider]
    storage_provider: Optional[Any] = None
    image_llm_provider: Optional[Any] = None


class ProviderEnvSettings(BaseSettings):
    """Pydantic settings for loading provider configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_endpoint: Optional[str] = Field(default=None)
    llm_deployment_name: Optional[str] = Field(default=None)
    llm_model_name: Optional[str] = Field(default=None)
    llm_api_version: Optional[str] = Field(default="2024-08-01-preview")
    llm_api_key: Optional[str] = Field(default=None)

    # Embedding Service Settings
    embedding_service_endpoint: Optional[str] = Field(default=None)
    embedding_service_deployment_name: Optional[str] = Field(default=None)
    embedding_service_api_version: Optional[str] = Field(default="2024-08-01-preview")
    embedding_service_api_key: Optional[str] = Field(default=None)

    # Storage Settings
    storage_account_name: Optional[str] = Field(default=None)
    keyframe_container_name: Optional[str] = Field(default=None)
    storage_access_key: Optional[str] = Field(default=None)

    # Speech Service Settings
    speech_service_resource_id: Optional[str] = Field(default=None)
    speech_service_region: Optional[str] = Field(default=None)
    speech_service_api_key: Optional[str] = Field(default=None)

    # Transcription Settings
    transcription_provider: str = Field(default="azure", validation_alias="TRANSCRIPTION_PROVIDER")
    whisper_deployment_name: Optional[str] = Field(default=None)
    transcription_api_key: Optional[str] = Field(default=None)

    # Neo4j Graph Database Settings
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")

    # Access Control (ACL) Settings
    acl_enabled: bool = Field(default=False)


@lru_cache(maxsize=1)
def get_settings() -> ProviderEnvSettings:
    """Return cached ProviderEnvSettings loaded from .env."""
    return ProviderEnvSettings()


def resolve_credentials():
    """Resolve Azure credentials using a chained strategy.

    Tries AzureCliCredential first (local dev / CI with ``az login``),
    then falls back to DefaultAzureCredential (managed identity,
    workload identity, environment variables, etc.).

    When ``AZURE_CLIENT_ID`` is set (e.g. in Container Apps with a
    user-assigned managed identity), it is passed explicitly so that
    ManagedIdentityCredential targets the correct identity instead of
    defaulting to the system-assigned one.
    """
    managed_identity_client_id = os.environ.get("AZURE_CLIENT_ID")
    return ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            managed_identity_client_id=managed_identity_client_id,
        ),
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> AzureLLMProvider:
    """Return a singleton AzureLLMProvider."""
    settings = get_settings()
    logger.info("Initializing AzureLLMProvider")

    # Construct provider args, using API key if provided and not a placeholder
    kwargs = {
        "endpoint": settings.llm_endpoint,
        "deployment_name": settings.llm_deployment_name,
        "model_name": settings.llm_model_name,
        "api_version": settings.llm_api_version,
    }

    api_key = settings.llm_api_key
    if api_key and not api_key.startswith("<your") and api_key.strip():
        kwargs["api_key"] = api_key
    else:
        logger.debug("LLM_API_KEY not provided or placeholder — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return AzureLLMProvider(**kwargs)


@lru_cache(maxsize=1)
def get_reasoning_llm_provider() -> AzureReasoningLLMProvider:
    """Return a singleton AzureReasoningLLMProvider (for o1/o3-mini style models).

    Reuses the same LLM endpoint and deployment as get_llm_provider().
    Configure a separate deployment if needed via LLM_DEPLOYMENT_NAME.
    """
    settings = get_settings()
    logger.info("Initializing AzureReasoningLLMProvider")

    # Construct provider args, using API key if provided and not a placeholder
    kwargs = {
        "endpoint": settings.llm_endpoint,
        "deployment_name": settings.llm_deployment_name,
        "model_name": settings.llm_model_name,
        "api_version": settings.llm_api_version,
    }

    api_key = settings.llm_api_key
    if api_key and not api_key.startswith("<your") and api_key.strip():
        kwargs["api_key"] = api_key
    else:
        logger.debug("LLM_API_KEY not provided or placeholder — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return AzureReasoningLLMProvider(**kwargs)


@lru_cache(maxsize=1)
def get_embedding_provider() -> AzureEmbeddingProvider:
    """Return a singleton AzureEmbeddingProvider."""
    settings = get_settings()
    logger.info("Initializing AzureEmbeddingProvider")

    kwargs = {
        "endpoint": settings.embedding_service_endpoint,
        "deployment_name": settings.embedding_service_deployment_name,
        "api_version": settings.embedding_service_api_version,
    }

    api_key = settings.embedding_service_api_key
    if api_key and not api_key.startswith("<your") and api_key.strip():
        kwargs["api_key"] = api_key
    else:
        logger.debug("EMBEDDING_SERVICE_API_KEY not provided — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return AzureEmbeddingProvider(**kwargs)


@lru_cache(maxsize=1)
def get_storage_provider() -> AzureStorageProvider:
    """Return a singleton AzureStorageProvider."""
    settings = get_settings()
    logger.info("Initializing AzureStorageProvider")

    kwargs = {
        "storage_account_name": settings.storage_account_name,
        "keyframe_container_name": settings.keyframe_container_name,
    }

    storage_key = settings.storage_access_key
    if storage_key and not storage_key.startswith("<your") and storage_key.strip():
        # Passing as connection string (provider currently expects this)
        kwargs["blob_connection_string"] = storage_key
    else:
        logger.debug("STORAGE_ACCESS_KEY not provided — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return AzureStorageProvider(**kwargs)


@lru_cache(maxsize=1)
def get_speech_provider() -> AzureSpeechServiceProvider:
    """Return a singleton AzureSpeechServiceProvider."""
    settings = get_settings()
    logger.info("Initializing AzureSpeechServiceProvider")

    kwargs = {
        "speech_service_resource_id": settings.speech_service_resource_id,
        "speech_service_region": settings.speech_service_region,
        "llm_provider": get_llm_provider(),
    }

    api_key = settings.speech_service_api_key
    if api_key and not api_key.startswith("<your") and api_key.strip():
        kwargs["api_key"] = api_key
    else:
        logger.debug("SPEECH_SERVICE_API_KEY not provided — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return AzureSpeechServiceProvider(**kwargs)


@lru_cache(maxsize=1)
def get_whisper_provider() -> WhisperTranscriptionProvider:
    """Return a singleton WhisperTranscriptionProvider."""
    settings = get_settings()
    logger.info("Initializing WhisperTranscriptionProvider")

    kwargs = {
        "endpoint": settings.llm_endpoint,
        "api_version": settings.llm_api_version,
        "deployment_name": settings.whisper_deployment_name,
    }

    api_key = settings.transcription_api_key
    if api_key and not api_key.startswith("<your") and api_key.strip():
        kwargs["api_key"] = api_key
    else:
        logger.debug("TRANSCRIPTION_API_KEY not provided — falling back to identity")
        kwargs["credentials"] = resolve_credentials()

    return WhisperTranscriptionProvider(**kwargs)


@lru_cache(maxsize=1)
def get_neo4j_query_provider() -> Optional[Neo4jQueryProvider]:
    """Return a singleton Neo4jQueryProvider, or None if NEO4J_PASSWORD is unset.

    Covers both v4 and v5 query pipeline use cases. Returns None when no Neo4j
    password is configured so callers can gracefully degrade.
    """
    settings = get_settings()
    if not settings.neo4j_password:
        logger.warning("NEO4J_PASSWORD not set — Neo4jQueryProvider not initialized")
        return None
    logger.info("Initializing Neo4jQueryProvider")
    return Neo4jQueryProvider(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


@lru_cache(maxsize=1)
def get_neo4j_graph_store_provider() -> Optional[Neo4jGraphStoreProvider]:
    """Return a singleton Neo4jGraphStoreProvider, or None if NEO4J_PASSWORD is unset."""
    settings = get_settings()
    if not settings.neo4j_password:
        logger.warning("NEO4J_PASSWORD not set — Neo4jGraphStoreProvider not initialized")
        return None
    logger.info("Initializing Neo4jGraphStoreProvider")
    return Neo4jGraphStoreProvider(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


@lru_cache(maxsize=1)
def get_ingestion_providers() -> IngestionProviders:
    """Return a provider bundle suitable for `mmct.video_pipeline.core.ingestion.IngestionPipeline`.
    
    Dynamically selects between Whisper and Azure Speech Service based on the
    TRANSCRIPTION_PROVIDER environment variable (defaults to 'azure').
    """
    settings = get_settings()
    
    # Select transcription provider based on settings
    if settings.transcription_provider.lower() == "whisper":
        transcription_provider = get_whisper_provider()
        logger.info("Using WhisperTranscriptionProvider for ingestion")
    else:
        transcription_provider = get_speech_provider()
        logger.info("Using AzureSpeechServiceProvider (default) for ingestion")

    return IngestionProviders(
        llm_provider=get_llm_provider(),
        transcription_provider=transcription_provider,
        storage_provider=get_storage_provider(),
        graph_store_provider=get_neo4j_graph_store_provider(),
        llm_config={},
    )


@lru_cache(maxsize=1)
def get_clip_provider() -> FastEmbedQdrantCLIPEmbeddingProvider:
    """Return a singleton FastEmbedQdrantCLIPEmbeddingProvider (512-dim image embeddings)."""
    logger.info("Initializing FastEmbedQdrantCLIPEmbeddingProvider")
    return FastEmbedQdrantCLIPEmbeddingProvider()


@lru_cache(maxsize=1)
def get_bge_provider() -> FastEmbedBGEsmallEmbeddingProvider:
    """Return a singleton FastEmbedBGEsmallEmbeddingProvider (384-dim text embeddings)."""
    logger.info("Initializing FastEmbedBGEsmallEmbeddingProvider")
    return FastEmbedBGEsmallEmbeddingProvider()


@lru_cache(maxsize=1)
def get_arctic_provider() -> FastEmbedArcticEmbeddingProvider:
    """Return a singleton FastEmbedArcticEmbeddingProvider (384-dim object embeddings)."""
    logger.info("Initializing FastEmbedArcticEmbeddingProvider")
    return FastEmbedArcticEmbeddingProvider()


@lru_cache(maxsize=1)
def get_query_pipeline_providers() -> QueryPipelineProviders:
    """Return the standard provider bundle for graph/state query pipelines."""
    llm_provider = get_llm_provider()
    return QueryPipelineProviders(
        model_client=llm_provider.get_autogen_client(),
        neo4j_provider=get_neo4j_query_provider(),
        storage_provider=get_storage_provider(),
        image_llm_provider=llm_provider,
    )


@lru_cache(maxsize=1)
def get_image_agent_provider() -> ImageAgentProviderConfig:
    """Return a singleton ImageAgentProviderConfig."""
    logger.info("Initializing ImageAgentProviderConfig")
    return ImageAgentProviderConfig(llm_provider=get_llm_provider())
