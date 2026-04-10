"""Node type definitions.

Each node type is defined in its own file and auto-registers on import.
"""

from mmct.video_pipeline.core.graph.node_types.chapter_group import ChapterGroupNodeType
from mmct.video_pipeline.core.graph.node_types.chapter import ChapterNodeType
from mmct.video_pipeline.core.graph.node_types.transcript import TranscriptNodeType
from mmct.video_pipeline.core.graph.node_types.event import EventNodeType
from mmct.video_pipeline.core.graph.node_types.object import ObjectNodeType
from mmct.video_pipeline.core.graph.node_types.keyframe import KeyframeNodeType

__all__ = [
    "ChapterGroupNodeType",
    "ChapterNodeType",
    "TranscriptNodeType",
    "EventNodeType",
    "ObjectNodeType",
    "KeyframeNodeType",
]
