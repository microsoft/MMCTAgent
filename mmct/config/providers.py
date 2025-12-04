"""Pydantic configuration models for provider dependency injection."""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Type

from mmct.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseStorageProvider,
    BaseTranscriptionProvider,
    BaseImageEmbeddingProvider,
)
from mmct.providers.base.chapter_vector_db_provider import BaseChapterVectorDBProvider
from mmct.providers.base.keyframes_vector_db_provider import BaseKeyframesVectorDBProvider
from mmct.providers.base.object_collection_vector_db_provider import BaseObjectCollectionVectorDBProvider


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
   
