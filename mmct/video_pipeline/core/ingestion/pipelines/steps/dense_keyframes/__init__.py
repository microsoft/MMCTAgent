"""Dense keyframe extraction with action boundary detection."""

from .step import DenseKeyframeExtractionStep, DenseKeyframeConfig
from .boundary_detector import detect_action_boundaries, BoundaryThresholds

__all__ = [
    "DenseKeyframeExtractionStep",
    "DenseKeyframeConfig",
    "detect_action_boundaries",
    "BoundaryThresholds",
]
