"""Keyframe extraction with action boundary detection."""

from .step import KeyframeExtractionStep, KeyframeConfig
from .boundary_detector import detect_action_boundaries, BoundaryThresholds

__all__ = [
    "KeyframeExtractionStep",
    "KeyframeConfig",
    "detect_action_boundaries",
    "BoundaryThresholds",
]
