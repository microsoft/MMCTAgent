"""Pydantic configuration models for provider dependency injection."""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Type, Optional

from mmct.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseStorageProvider,
    BaseTranscriptionProvider,
    BaseImageEmbeddingProvider,
    BaseGraphDBProvider,
    BaseGraphStoreProvider,
)
from mmct.providers.base.chapter_vector_db_provider import BaseChapterVectorDBProvider
from mmct.providers.base.keyframes_vector_db_provider import BaseKeyframesVectorDBProvider
from mmct.providers.base.object_collection_vector_db_provider import BaseObjectCollectionVectorDBProvider

# Sprint 2: New provider base classes
from mmct.providers.base.synopsis_vector_db_provider import BaseSynopsisVectorDBProvider
from mmct.providers.base.temporal_vector_db_provider import BaseTemporalVectorDBProvider

# Sprint 1: Event and object vector DB providers
from mmct.providers.base.event_vector_db_provider import BaseEventVectorDBProvider
from mmct.providers.base.object_vector_db_provider import BaseObjectVectorDBProvider


# All valid provider base classes
PROVIDER_TYPES: Dict[str, Type] = {
    "BaseLLMProvider": BaseLLMProvider,
    "BaseEmbeddingProvider": BaseEmbeddingProvider,
    "BaseStorageProvider": BaseStorageProvider,
    "BaseTranscriptionProvider": BaseTranscriptionProvider,
    "BaseImageEmbeddingProvider": BaseImageEmbeddingProvider,
    "BaseChapterVectorDBProvider": BaseChapterVectorDBProvider,
    "BaseKeyframesVectorDBProvider": BaseKeyframesVectorDBProvider,
    "BaseObjectCollectionVectorDBProvider": BaseObjectCollectionVectorDBProvider,
    # Sprint 2: New provider types
    "BaseSynopsisVectorDBProvider": BaseSynopsisVectorDBProvider,
    "BaseTemporalVectorDBProvider": BaseTemporalVectorDBProvider,
    # Sprint 1: Graph and vector DB provider types
    "BaseGraphDBProvider": BaseGraphDBProvider,
    "BaseGraphStoreProvider": BaseGraphStoreProvider,
    "BaseEventVectorDBProvider": BaseEventVectorDBProvider,
    "BaseObjectVectorDBProvider": BaseObjectVectorDBProvider,
}


# Helper: Validate one provider
def _validate_provider(value, expected_type, field_name):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Invalid provider for '{field_name}'. "
            f"Expected {expected_type.__name__}, got {type(value).__name__}."
        )
    return value

# Base Class: Auto-generates validators for all provider fields
class ProviderConfigBase(BaseModel):

    @classmethod
    def __get_validators__(cls):
        # yield BaseModel's validators first
        yield from super().__get_validators__()

        # Auto-generate validators for each provider field
        for field_name, annotation in cls.__annotations__.items():

            # Check if annotation is one of the provider base classes
            for provider_name, provider_type in PROVIDER_TYPES.items():
                if annotation is provider_type:
                    # Create validator and bind it to the field
                    def make_validator(f_name, f_type):
                        @field_validator(f_name)
                        def validator(v):
                            return _validate_provider(v, f_type, f_name)
                        return validator

                    validator_func = make_validator(field_name, provider_type)
                    setattr(cls, f"validate_{field_name}", validator_func)
                    break

    class Config:
        arbitrary_types_allowed = True


# ImageAgent Provider Config
class ImageAgentProviderConfig(ProviderConfigBase):
    llm_provider: BaseLLMProvider = Field(...)


# VideoAgent Provider Config
class VideoAgentProviderConfig(ProviderConfigBase):
    llm_provider: BaseLLMProvider = Field(...)
    embedding_provider: BaseEmbeddingProvider = Field(...)
    image_embedding_provider: BaseImageEmbeddingProvider = Field(...)
    vectordb_chapter: BaseChapterVectorDBProvider = Field(...)
    vectordb_object_registry: BaseObjectCollectionVectorDBProvider = Field(...)
    vectordb_keyframes: BaseKeyframesVectorDBProvider = Field(...)
    storage_provider: BaseStorageProvider = Field(...)
    
    # Sprint 2: Optional new providers (backward compatible)
    vectordb_synopsis: Optional[BaseSynopsisVectorDBProvider] = Field(default=None)
    vectordb_temporal: Optional[BaseTemporalVectorDBProvider] = Field(default=None)
    llm_config: Optional[Dict] = Field(default=None)


# Sprint 1: Temporal Graph Provider Config
class TemporalGraphProviderConfig(ProviderConfigBase):
    """Configuration for temporal graph providers.
    
    Supports graph-based storage and vector search for events and objects
    in video temporal knowledge graphs.
    """
    graph_db_provider: BaseGraphDBProvider = Field(
        ..., description="Graph database provider for node/edge operations"
    )
    event_vector_provider: Optional[BaseEventVectorDBProvider] = Field(
        default=None, description="Optional event vector search provider"
    )
    object_vector_provider: Optional[BaseObjectVectorDBProvider] = Field(
        default=None, description="Optional object vector search provider"
    )
    embedding_provider: Optional[BaseEmbeddingProvider] = Field(
        default=None, description="Optional embedding provider for vector generation"
    )


# IngestionPipeline Provider Config
class IngestionProviderConfig(ProviderConfigBase):
    llm_provider: BaseLLMProvider = Field(...)
    embedding_provider: BaseEmbeddingProvider = Field(...)
    image_embedding_provider: BaseImageEmbeddingProvider = Field(...)
    vectordb_chapter: BaseChapterVectorDBProvider = Field(...)
    vectordb_object_registry: BaseObjectCollectionVectorDBProvider = Field(...)
    vectordb_keyframes: BaseKeyframesVectorDBProvider = Field(...)
    storage_provider: BaseStorageProvider = Field(...)
    transcription_provider: BaseTranscriptionProvider = Field(...)
    
    # Graph store provider for uploading graphs to Neo4j or other stores
    graph_store_provider: Optional[BaseGraphStoreProvider] = Field(
        default=None, description="Graph store provider for uploading graphs"
    )
    
    # Sprint 2: Optional new providers (backward compatible)
    vectordb_synopsis: Optional[BaseSynopsisVectorDBProvider] = Field(default=None)
    vectordb_temporal: Optional[BaseTemporalVectorDBProvider] = Field(default=None)
    llm_config: Optional[Dict] = Field(default=None)
   
