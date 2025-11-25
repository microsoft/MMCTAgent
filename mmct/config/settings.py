from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# Load environment variables once at module level
load_dotenv(find_dotenv())


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    llm_provider: str = Field(default="azure")
    llm_endpoint: str
    llm_deployment_name: str
    llm_api_version: str = Field(default="2024-08-01-preview")
    llm_model_name: str
    llm_use_managed_identity: bool = Field(default=True)
    llm_api_key: Optional[str] = Field(default=None)
    llm_vision_deployment_name: Optional[str] = Field(default=None)
    llm_vision_api_version: Optional[str] = Field(default=None)
    llm_timeout: int = Field(default=200)
    llm_max_retries: int = Field(default=2)
    llm_temperature: float = Field(default=0.0)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class SearchConfig(BaseSettings):
    """Search provider configuration."""

    search_provider: str = Field(default="azure_ai_search")
    search_endpoint: Optional[str] = Field(default=None)
    search_api_key: Optional[str] = Field(default=None)
    search_use_managed_identity: bool = Field(default=True)
    search_index_name: str = Field(default="default")
    search_timeout: int = Field(default=30)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class EmbeddingConfig(BaseSettings):
    """Embedding provider configuration."""

    embedding_provider: str = Field(default="azure")
    embedding_service_endpoint: str
    embedding_service_deployment_name: str
    embedding_service_api_version: str = Field(default="2024-08-01-preview")
    embedding_service_api_key: Optional[str] = Field(default=None)
    embedding_use_managed_identity: bool = Field(default=True)
    embedding_timeout: int = Field(default=200)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class ImageEmbeddingConfig(BaseSettings):
    """Image embedding provider configuration for CLIP models."""

    image_embedding_model_name: str = Field(default="openai/clip-vit-base-patch32")
    image_embedding_device: str = Field(default="auto")
    image_embedding_max_size: int = Field(default=224)
    image_embedding_batch_size: int = Field(default=8)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    def to_provider_config(self) -> dict:
        """Convert to provider configuration dictionary."""
        return {
            "model_name": self.image_embedding_model_name,
            "device": self.image_embedding_device,
            "max_image_size": self.image_embedding_max_size,
            "batch_size": self.image_embedding_batch_size
        }


class TranscriptionConfig(BaseSettings):
    """Transcription provider configuration."""

    transcription_provider: str = Field(default="azure")
    whisper_endpoint: Optional[str] = Field(default=None)
    speech_service_deployment_name: Optional[str] = Field(default=None)
    speech_service_api_version: str = Field(default="2024-08-01-preview")
    speech_service_key: Optional[str] = Field(default=None)
    speech_service_region: Optional[str] = Field(default=None)
    speech_service_resource_id: Optional[str] = Field(default=None)
    speech_use_managed_identity: bool = Field(default=True)
    speech_timeout: int = Field(default=200)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class StorageConfig(BaseSettings):
    """Storage configuration."""

    storage_provider: str = Field(default="azure")
    storage_connection_string: Optional[str] = Field(default=None)
    storage_account_name: Optional[str] = Field(default=None)
    storage_container_name: str = Field(default="default")
    storage_account_url: Optional[str] = Field(default=None)
    storage_use_managed_identity: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class VisionConfig(BaseSettings):
    """Vision provider configuration."""

    vision_provider: str = Field(default="azure")

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class SecurityConfig(BaseSettings):
    """Security configuration."""

    keyvault_url: Optional[str] = Field(default=None)
    enable_secrets_manager: bool = Field(default=False)
    managed_identity_client_id: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    log_enable_json: bool = Field(default=False)
    log_enable_file: bool = Field(default=False)
    log_max_file_size: str = Field(default="10 MB")
    log_retention_days: int = Field(default=7)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class MMCTConfig(BaseSettings):
    """Main configuration class with all nested configs."""

    # Application settings
    app_name: str = Field(default="MMCT Agent")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # Nested configuration objects
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    image_embedding: ImageEmbeddingConfig = Field(default_factory=ImageEmbeddingConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


# Create single settings instance
settings = MMCTConfig()