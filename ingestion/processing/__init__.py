"""Video processing modules for keyframe extraction and embeddings generation."""

from .keyframe_extractor import KeyframeExtractor, extract_keyframes_from_video
from .clip_embeddings import CLIPEmbeddingsGenerator

__all__ = [
    "KeyframeExtractor",
    "extract_keyframes_from_video",
    "CLIPEmbeddingsGenerator",
]
