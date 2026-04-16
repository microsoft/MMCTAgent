"""Keyframe upload pipeline step.

Uploads keyframe images to blob storage with metadata for the ingestion pipeline.
"""

from .step import KeyframeUploadStep

__all__ = ["KeyframeUploadStep"]
