"""Configuration for MCP Server."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MCPServerConfig:
    """Configuration for MCP Server."""

    # Azure AI Search settings
    search_endpoint: str = os.getenv("SEARCH_SERVICE_ENDPOINT", "")
    index_name: str = os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index")
    search_api_key: Optional[str] = os.getenv("SEARCH_API_KEY")

    # CLIP model settings
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Search settings
    default_top_k: int = 10

    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Create config from environment variables."""
        return cls(
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT", ""),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            clip_model_name=os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "10"))
        )
