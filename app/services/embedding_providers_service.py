"""Embedding and Neo4j provider singletons for the query pipeline.

Provides lazy-initialized singleton providers for text embeddings, image
embeddings, and the Neo4j graph database used during video query processing.
These providers are shared across query services to avoid redundant initialization.
"""

from typing import Optional

from loguru import logger

from mmct.v5.query.neo4j_provider import Neo4jQueryProvider
from app.config import get_settings


# Singleton instances
_neo4j_provider: Optional[Neo4jQueryProvider] = None
_embedding_provider = None
_image_embedding_provider = None


def get_neo4j_provider() -> Neo4jQueryProvider:
    """Get or create Neo4j provider singleton."""
    global _neo4j_provider
    if _neo4j_provider is None:
        settings = get_settings()

        if not settings.neo4j_password:
            raise ValueError("Neo4j password not configured. Set NEO4J_PASSWORD env variable.")

        _neo4j_provider = Neo4jQueryProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        logger.info(f"Neo4j provider initialized: {settings.neo4j_uri}")

    return _neo4j_provider


def get_text_embedding_provider():
    """Get text embedding provider (FastEmbedBGEsmall, 384-dim).

    This is the same provider used during graph ingestion.
    """
    global _embedding_provider
    if _embedding_provider is None:
        from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider

        _embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        logger.info("FastEmbedBGEsmallEmbeddingProvider initialized (384-dim)")

    return _embedding_provider


def get_image_embedding_provider():
    """Get image embedding provider (QdrantCLIP, 512-dim).

    This is the same provider used during graph ingestion for keyframes.
    """
    global _image_embedding_provider
    if _image_embedding_provider is None:
        from mmct.providers.custom_providers import FastEmbedQdrantCLIPEmbeddingProvider

        _image_embedding_provider = FastEmbedQdrantCLIPEmbeddingProvider()
        logger.info("FastEmbedQdrantCLIPEmbeddingProvider initialized (512-dim)")

    return _image_embedding_provider


