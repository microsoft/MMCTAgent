"""
FastAPI dependency injection wrappers around provider_config singletons.

All functions are synchronous thin wrappers. The lru_cache decorators in
provider_config ensure at most one provider instance per type per process lifetime.
"""

from functools import lru_cache

from config.provider_config import (
    IngestionProviders,
    QueryPipelineProviders,
    get_ingestion_providers,
    get_llm_provider,
    get_query_pipeline_providers,
)
from mmct.image_pipeline.config import ImageAgentProviderConfig


@lru_cache(maxsize=1)
def _build_image_agent_provider_config() -> ImageAgentProviderConfig:
    return ImageAgentProviderConfig(llm_provider=get_llm_provider())


def get_ingestion_providers_dep() -> IngestionProviders:
    """FastAPI Depends-compatible factory — returns the cached IngestionProviders bundle."""
    return get_ingestion_providers()


def get_query_pipeline_providers_dep() -> QueryPipelineProviders:
    """FastAPI Depends-compatible factory — returns the cached QueryPipelineProviders bundle."""
    return get_query_pipeline_providers()


def get_image_agent_provider_config_dep() -> ImageAgentProviderConfig:
    """FastAPI Depends-compatible factory — returns the cached ImageAgentProviderConfig."""
    return _build_image_agent_provider_config()
