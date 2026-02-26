"""Node type definitions.

Each node type is defined in its own file and auto-registers on import.
"""

from mmct.graph.node_types.chapter_group import ChapterGroupNodeType
from mmct.graph.node_types.chapter import ChapterNodeType
from mmct.graph.node_types.transcript import TranscriptNodeType
from mmct.graph.node_types.event import EventNodeType
from mmct.graph.node_types.object import ObjectNodeType
from mmct.graph.node_types.keyframe import KeyframeNodeType

__all__ = [
    "ChapterGroupNodeType",
    "ChapterNodeType",
    "TranscriptNodeType",
    "EventNodeType",
    "ObjectNodeType",
    "KeyframeNodeType",
]
