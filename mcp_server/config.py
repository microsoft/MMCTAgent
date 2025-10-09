"""Configuration for MCP Server."""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class EmbeddingModel(Enum):
    """Enumeration of supported embedding models."""
    CLIP_VIT_BASE_PATCH32 = "openai/clip-vit-base-patch32"
    COLQWEN_2_5 = "vidore/colqwen2.5-v0.2"


@dataclass
class MCPServerConfig:
    """Configuration for MCP Server."""

    # Azure AI Search settings
    search_endpoint: str = os.getenv("SEARCH_SERVICE_ENDPOINT", "")
    index_name: str = os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index")
    search_api_key: Optional[str] = os.getenv("SEARCH_API_KEY")

    # Embedding model settings
    embedding_model: str = EmbeddingModel.COLQWEN_2_5.value

    # Search settings
    default_top_k: int = 10

    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Create config from environment variables."""
        # Get embedding model from env, default to ColQwen
        embedding_model_env = os.getenv("EMBEDDING_MODEL", "")

        # If env variable is set, use it; otherwise use the enum default
        if embedding_model_env:
            embedding_model = embedding_model_env
        else:
            embedding_model = EmbeddingModel.COLQWEN_2_5.value

        return cls(
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT", ""),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            embedding_model=embedding_model,
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "10"))
        )
