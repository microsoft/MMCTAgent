"""
Configuration dataclasses for video ingestion pipeline.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class EmbeddingModel(Enum):
    """Enumeration of supported embedding models."""
    CLIP_VIT_BASE_PATCH32 = "openai/clip-vit-base-patch32"
    COLQWEN_2_5 = "vidore/colqwen2.5-v0.2"

@dataclass
class KeyframeExtractionConfig:
    """Configuration for keyframe extraction."""
    motion_threshold: float = 0.8
    sample_fps: int = 1
    max_frame_width: int = 800
    debug_mode: bool = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    clip_model_name: str = EmbeddingModel.COLQWEN_2_5.value
    batch_size: int = 8
    device: str = "auto"  # "cpu", "cuda", or "auto"
    max_image_size: int = 224


@dataclass
class BlobStorageConfig:
    """Configuration for blob storage operations."""
    connection_string: Optional[str] = None
    container_name: str = "keyframes"
    upload_batch_size: int = 10
    generate_sas_token: bool = False
    sas_expiry_hours: int = 24


@dataclass
class SearchIndexConfig:
    """Configuration for Azure AI Search index."""
    search_endpoint: str = ""
    index_name: str = "video-keyframes-index"
    embedding_model: str = EmbeddingModel.COLQWEN_2_5.value
    hnsw_m: int = 4
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 500
    batch_upload_size: int = 100

    @property
    def vector_dimensions(self) -> int:
        """Get vector dimensions based on the embedding model."""
        if self.embedding_model.startswith('openai'):
            return 512  # CLIP models
        elif self.embedding_model.startswith('vidore'):
            return 128  # ColQwen models
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}")
