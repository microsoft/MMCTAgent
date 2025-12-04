"""Settings management using Pydantic."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    api_key: str = Field(default="", description="Azure OpenAI API key")
    endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    deployment: str = Field(default="", description="Default deployment name")
    api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )
    
    # Additional deployments for multi-model support
    deployments: dict[str, str] = Field(
        default_factory=dict,
        description="Named deployments mapping (name -> deployment_id)",
    )
    
    # Request settings
    timeout_seconds: float = Field(default=60.0, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Initial retry delay")
    
    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Ensure endpoint doesn't have trailing slash."""
        return v.rstrip("/") if v else v
    
    def get_deployment(self, name: str | None = None) -> str:
        """Get a deployment ID by name.
        
        Args:
            name: Deployment name. Uses default if None.
            
        Returns:
            Deployment ID.
        """
        if name is None:
            return self.deployment
        return self.deployments.get(name, self.deployment)


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MMCT_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    include_timestamp: bool = Field(default=True, description="Include timestamps")
    include_trace_id: bool = Field(default=True, description="Include trace IDs")
    output: str = Field(default="stderr", description="Output destination")


class MemorySettings(BaseSettings):
    """Memory configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MMCT_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    default_strategy: str = Field(
        default="adaptive",
        description="Default memory strategy",
    )
    max_tokens: int = Field(default=4000, description="Maximum context tokens")
    window_size: int = Field(default=20, description="Sliding window size")
    summarization_threshold: int = Field(
        default=3000,
        description="Token threshold for summarization",
    )
    persist_path: str = Field(
        default="./memory_logs",
        description="Path for memory persistence",
    )


class SwarmSettings(BaseSettings):
    """Swarm configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MMCT_SWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    max_iterations: int = Field(default=20, description="Maximum swarm iterations")
    max_agent_iterations: int = Field(
        default=5,
        description="Max consecutive runs per agent",
    )
    timeout_seconds: float = Field(default=300.0, description="Swarm timeout")
    persist_memory: bool = Field(default=False, description="Enable memory persistence")
    context_transform_enabled: bool = Field(
        default=True,
        description="Enable context transformation on handoff",
    )


class Settings(BaseSettings):
    """Main settings aggregating all configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Sub-settings (loaded from environment)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    swarm: SwarmSettings = Field(default_factory=SwarmSettings)
    
    # Convenience accessors for common Azure OpenAI settings
    @property
    def azure_openai_api_key(self) -> str:
        """Get Azure OpenAI API key."""
        return self.azure_openai.api_key
    
    @property
    def azure_openai_endpoint(self) -> str:
        """Get Azure OpenAI endpoint."""
        return self.azure_openai.endpoint
    
    @property
    def azure_openai_deployment(self) -> str:
        """Get default Azure OpenAI deployment."""
        return self.azure_openai.deployment
    
    @property
    def azure_openai_api_version(self) -> str:
        """Get Azure OpenAI API version."""
        return self.azure_openai.api_version
    
    def create_azure_client(
        self,
        deployment: str | None = None,
    ) -> Any:
        """Create an Azure OpenAI client with current settings.
        
        Args:
            deployment: Optional deployment name override.
            
        Returns:
            Configured AzureOpenAIClient.
        """
        from mmct_agent.llm import AzureOpenAIClient, LLMConfig
        
        return AzureOpenAIClient(
            api_key=self.azure_openai.api_key,
            endpoint=self.azure_openai.endpoint,
            deployment=deployment or self.azure_openai.deployment,
            api_version=self.azure_openai.api_version,
            config=LLMConfig(
                timeout_seconds=self.azure_openai.timeout_seconds,
                max_retries=self.azure_openai.max_retries,
                retry_delay_seconds=self.azure_openai.retry_delay_seconds,
            ),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings instance.
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings, clearing cache.
    
    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
