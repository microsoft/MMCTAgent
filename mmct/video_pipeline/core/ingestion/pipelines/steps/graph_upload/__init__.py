"""Graph upload pipeline step.

Generates embeddings and uploads the hierarchical graph to a graph store.

Embedding Generation:
- Text nodes (ChapterGroup, Chapter, Event, Object): BGE-small (384-dim)
- Image nodes (Keyframe): QdrantCLIP (512-dim)

Supported providers:
- Neo4jGraphStoreProvider: Upload to Neo4j with vector indexes
"""

from .step import GraphUploadStep

__all__ = [
    "GraphUploadStep",
]
