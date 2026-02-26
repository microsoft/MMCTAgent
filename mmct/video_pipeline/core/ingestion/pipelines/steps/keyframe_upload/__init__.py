"""Keyframe upload pipeline step.

Uploads keyframe images to blob storage with metadata for the dense captioning pipeline.
"""

from .step import KeyframeUploadStep

__all__ = ["KeyframeUploadStep"]
