from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import os


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    
    provider: str = Field(default="azure", env="LLM_PROVIDER")
    endpoint: str = Field(..., env="LLM_ENDPOINT")
    deployment_name: str = Field(..., env="LLM_DEPLOYMENT_NAME")
    api_version: str = Field(default="2024-08-01-preview", env="LLM_API_VERSION")
    model_name: str = Field(..., env="LLM_MODEL_NAME")
    use_managed_identity: bool = Field(default=True, env="LLM_USE_MANAGED_IDENTITY")
    api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    embedding_deployment_name: Optional[str] = Field(default=None, env="EMBEDDING_SERVICE_DEPLOYMENT_NAME")
    vision_deployment_name: Optional[str] = Field(default=None, env="LLM_VISION_DEPLOYMENT_NAME")
    vision_api_version: Optional[str] = Field(default=None, env="LLM_VISION_API_VERSION")
    timeout: int = Field(default=200, env="LLM_TIMEOUT")
    max_retries: int = Field(default=2, env="LLM_MAX_RETRIES")
    temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs):
        # Force load environment variables before validation
        load_dotenv(find_dotenv())
        
        # If no explicit values provided, use environment variables
        if not kwargs:
            kwargs = {
                'endpoint': os.getenv("LLM_ENDPOINT"),
                'deployment_name': os.getenv("LLM_DEPLOYMENT_NAME"),
                'model_name': os.getenv("LLM_MODEL_NAME"),
                'provider': os.getenv("LLM_PROVIDER", "azure"),
                'api_version': os.getenv("LLM_API_VERSION", "2024-08-01-preview"),
                'use_managed_identity': os.getenv("LLM_USE_MANAGED_IDENTITY", "true").lower() == "true",
                'api_key': os.getenv("LLM_API_KEY"),
                'embedding_deployment_name': os.getenv("EMBEDDING_SERVICE_DEPLOYMENT_NAME"),
                'vision_deployment_name': os.getenv("LLM_VISION_DEPLOYMENT_NAME"),
                'vision_api_version': os.getenv("LLM_VISION_API_VERSION"),
                'timeout': int(os.getenv("LLM_TIMEOUT", "200")),
                'max_retries': int(os.getenv("LLM_MAX_RETRIES", "2")),
                'temperature': float(os.getenv("LLM_TEMPERATURE", "0.0")),
            }
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        super().__init__(**kwargs)


class SearchConfig(BaseSettings):
    """Search provider configuration."""
    
    provider: str = Field(default="azure_ai_search", env="SEARCH_PROVIDER")
    endpoint: Optional[str] = Field(default=None, env="SEARCH_ENDPOINT")
    api_key: Optional[str] = Field(default=None, env="SEARCH_API_KEY")
    use_managed_identity: bool = Field(default=True, env="SEARCH_USE_MANAGED_IDENTITY")
    index_name: str = Field(default="default", env="SEARCH_INDEX_NAME")
    timeout: int = Field(default=30, env="SEARCH_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs):
        # Force load environment variables before validation
        load_dotenv(find_dotenv())
        
        # If no explicit values provided, use environment variables
        if not kwargs:
            kwargs = {
                'provider': os.getenv("SEARCH_PROVIDER", "azure_ai_search"),
                'endpoint': os.getenv("SEARCH_ENDPOINT"),
                'api_key': os.getenv("SEARCH_API_KEY"),
                'use_managed_identity': os.getenv("SEARCH_USE_MANAGED_IDENTITY", "true").lower() == "true",
                'index_name': os.getenv("SEARCH_INDEX_NAME", "default"),
                'timeout': int(os.getenv("SEARCH_TIMEOUT", "30")),
            }
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        super().__init__(**kwargs)


class EmbeddingConfig(BaseSettings):
    """Embedding provider configuration."""
    
    provider: str = Field(default="azure", env="EMBEDDING_PROVIDER")
    endpoint: str = Field(..., env="EMBEDDING_SERVICE_ENDPOINT")
    deployment_name: str = Field(..., env="EMBEDDING_SERVICE_DEPLOYMENT_NAME")
    api_version: str = Field(default="2024-08-01-preview", env="EMBEDDING_SERVICE_API_VERSION")
    api_key: Optional[str] = Field(default=None, env="EMBEDDING_SERVICE_API_KEY")
    use_managed_identity: bool = Field(default=True, env="EMBEDDING_USE_MANAGED_IDENTITY")
    timeout: int = Field(default=200, env="EMBEDDING_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs):
        # Force load environment variables before validation
        load_dotenv(find_dotenv())
        
        # If no explicit values provided, use environment variables with fallbacks to LLM config
        if not kwargs:
            kwargs = {
                'provider': os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "azure")),
                'endpoint': os.getenv("EMBEDDING_SERVICE_ENDPOINT") or os.getenv("LLM_ENDPOINT"),
                'deployment_name': os.getenv("EMBEDDING_SERVICE_DEPLOYMENT_NAME") or os.getenv("LLM_DEPLOYMENT_NAME"),
                'api_version': os.getenv("EMBEDDING_SERVICE_API_VERSION", os.getenv("LLM_API_VERSION", "2024-08-01-preview")),
                'api_key': os.getenv("EMBEDDING_SERVICE_API_KEY", os.getenv("LLM_API_KEY")),
                'use_managed_identity': os.getenv("EMBEDDING_USE_MANAGED_IDENTITY", os.getenv("LLM_USE_MANAGED_IDENTITY", "true")).lower() == "true",
                'timeout': int(os.getenv("EMBEDDING_TIMEOUT", "200")),
            }
            # Remove None values but ensure required fields are provided
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            # Validate required fields and provide fallbacks
            if not kwargs.get('endpoint'):
                raise ValueError("EMBEDDING_SERVICE_ENDPOINT or LLM_ENDPOINT must be provided")
            if not kwargs.get('deployment_name'):
                raise ValueError("EMBEDDING_SERVICE_DEPLOYMENT_NAME or LLM_DEPLOYMENT_NAME must be provided")
        
        super().__init__(**kwargs)


class ImageEmbeddingConfig(BaseSettings):
    """Image embedding provider configuration for CLIP models."""

    model_name: str = Field(default="openai/clip-vit-base-patch32", env="IMAGE_EMBEDDING_MODEL_NAME")
    device: str = Field(default="auto", env="IMAGE_EMBEDDING_DEVICE")
    max_image_size: int = Field(default=224, env="IMAGE_EMBEDDING_MAX_SIZE")
    batch_size: int = Field(default=8, env="IMAGE_EMBEDDING_BATCH_SIZE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        extra="ignore",
        case_sensitive=False
    )

    def __init__(self, **kwargs):
        # Force load environment variables before validation
        from dotenv import load_dotenv, find_dotenv
        import os

        load_dotenv(find_dotenv())

        # If no explicit values provided, use environment variables with defaults
        if not kwargs:
            kwargs = {
                'model_name': os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32"),
                'device': os.getenv("IMAGE_EMBEDDING_DEVICE", "auto"),
                'max_image_size': int(os.getenv("IMAGE_EMBEDDING_MAX_SIZE", "224")),
                'batch_size': int(os.getenv("IMAGE_EMBEDDING_BATCH_SIZE", "8")),
            }

        super().__init__(**kwargs)

    def to_provider_config(self) -> dict:
        """Convert to provider configuration dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_image_size": self.max_image_size,
            "batch_size": self.batch_size
        }


class TranscriptionConfig(BaseSettings):
    """Transcription provider configuration."""

    provider: str = Field(default="azure", env="TRANSCRIPTION_PROVIDER")
    # Whisper API settings
    endpoint: Optional[str] = Field(default=None, env="WHISPER_ENDPOINT")
    deployment_name: Optional[str] = Field(default=None, env="SPEECH_SERVICE_DEPLOYMENT_NAME")
    api_version: str = Field(default="2024-08-01-preview", env="SPEECH_SERVICE_API_VERSION")
    api_key: Optional[str] = Field(default=None, env="SPEECH_SERVICE_KEY")
    # Azure Speech SDK (STT) settings
    region: Optional[str] = Field(default=None, env="SPEECH_SERVICE_REGION")
    resource_id: Optional[str] = Field(default=None, env="SPEECH_SERVICE_RESOURCE_ID")
    # Common settings
    use_managed_identity: bool = Field(default=True, env="SPEECH_USE_MANAGED_IDENTITY")
    timeout: int = Field(default=200, env="SPEECH_TIMEOUT")

    def __init__(self, **kwargs):
        # Force load environment variables before validation
        load_dotenv(find_dotenv())

        # If no explicit values provided, use environment variables
        if not kwargs:
            kwargs = {
                'provider': os.getenv("TRANSCRIPTION_PROVIDER", "azure"),
                'endpoint': os.getenv("WHISPER_ENDPOINT"),
                'deployment_name': os.getenv("SPEECH_SERVICE_DEPLOYMENT_NAME"),
                'api_version': os.getenv("SPEECH_SERVICE_API_VERSION", "2024-08-01-preview"),
                'api_key': os.getenv("SPEECH_SERVICE_KEY"),
                'region': os.getenv("SPEECH_SERVICE_REGION"),
                'resource_id': os.getenv("SPEECH_SERVICE_RESOURCE_ID"),
                'use_managed_identity': os.getenv("SPEECH_USE_MANAGED_IDENTITY", "true").lower() == "true",
                'timeout': int(os.getenv("SPEECH_TIMEOUT", "200")),
            }

        super().__init__(**kwargs)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        extra="ignore",
        case_sensitive=False
    )


class StorageConfig(BaseSettings):
    """Storage configuration."""

    provider: str = Field(default="azure", env="STORAGE_PROVIDER")
    connection_string: Optional[str] = Field(default=None, env="STORAGE_CONNECTION_STRING")
    account_name: Optional[str] = Field(default=None, env="STORAGE_ACCOUNT_NAME")
    container_name: str = Field(default="default", env="STORAGE_CONTAINER_NAME")
    account_url: Optional[str] = Field(default=None, env="STORAGE_ACCOUNT_URL")
    use_managed_identity: bool = Field(default=True, env="STORAGE_USE_MANAGED_IDENTITY")

    def __init__(self, **kwargs):
        # Force load environment variables before validation
        load_dotenv(find_dotenv())

        # If no explicit values provided, use environment variables
        if not kwargs:
            kwargs = {
                'provider': os.getenv("STORAGE_PROVIDER", "azure"),
                'connection_string': os.getenv("STORAGE_CONNECTION_STRING"),
                'account_name': os.getenv("STORAGE_ACCOUNT_NAME"),
                'container_name': os.getenv("STORAGE_CONTAINER_NAME"),
                'account_url': os.getenv("STORAGE_ACCOUNT_URL"),
                'use_managed_identity': os.getenv("STORAGE_USE_MANAGED_IDENTITY"),
            }

        super().__init__(**kwargs)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        extra="ignore",
        case_sensitive=False
    )


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    keyvault_url: Optional[str] = Field(default=None, env="KEYVAULT_URL")
    enable_secrets_manager: bool = Field(default=False, env="ENABLE_SECRETS_MANAGER")
    managed_identity_client_id: Optional[str] = Field(default=None, env="MANAGED_IDENTITY_CLIENT_ID")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    enable_json: bool = Field(default=False, env="LOG_ENABLE_JSON")
    enable_file_logging: bool = Field(default=False, env="LOG_ENABLE_FILE")
    max_file_size: str = Field(default="10 MB", env="LOG_MAX_FILE_SIZE")
    retention_days: int = Field(default=7, env="LOG_RETENTION_DAYS")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )


class MMCTConfig(BaseSettings):
    """Main configuration class."""
    
    # Application settings
    app_name: str = Field(default="MMCT Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        validate_assignment=True, 
        extra="ignore",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs):
        # Force load environment variables before initializing
        load_dotenv(find_dotenv())
        
        super().__init__(**kwargs)
        # Initialize cached configurations
        self._llm = None
        self._search = None
        self._embedding = None
        self._transcription = None
        self._storage = None
        self._security = None
        self._logging = None
    
    @property
    def llm(self) -> LLMConfig:
        if self._llm is None:
            # Force load environment variables first
            load_dotenv(find_dotenv(), override=True)
            self._llm = LLMConfig()
        return self._llm
    
    @property
    def search(self) -> SearchConfig:
        if self._search is None:
            load_dotenv(find_dotenv(), override=True)
            self._search = SearchConfig()
        return self._search
    
    @property
    def embedding(self) -> EmbeddingConfig:
        if self._embedding is None:
            load_dotenv(find_dotenv(), override=True)
            self._embedding = EmbeddingConfig()
        return self._embedding
    
    @property
    def transcription(self) -> TranscriptionConfig:
        if self._transcription is None:
            load_dotenv(find_dotenv(), override=True)
            self._transcription = TranscriptionConfig()
        return self._transcription
    
    @property
    def storage(self) -> StorageConfig:
        if self._storage is None:
            load_dotenv(find_dotenv(), override=True)
            self._storage = StorageConfig()
        return self._storage
    
    @property
    def security(self) -> SecurityConfig:
        if self._security is None:
            load_dotenv(find_dotenv(), override=True)
            self._security = SecurityConfig()
        return self._security
    
    @property
    def logging(self) -> LoggingConfig:
        if self._logging is None:
            load_dotenv(find_dotenv(), override=True)
            self._logging = LoggingConfig()
        return self._logging