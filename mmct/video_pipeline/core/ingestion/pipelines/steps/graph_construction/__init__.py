"""Graph construction pipeline step.

Builds hierarchical temporal knowledge graphs from chapters, events, objects, and keyframes.

Node Types:
- ChapterGroup: High-level topic groupings
- Chapter: Video segments with temporal boundaries
- Keyframe: Visual frames linked to chapters
- Event: Atomic actions/occurrences
- Object: Entities that appear in events

Edge Types:
- NEXT_GROUP/PREV_GROUP: ChapterGroup temporal sequence
- HAS_CHAPTER/IN_GROUP: ChapterGroup ↔ Chapter hierarchy
- NEXT_CHAPTER/PREV_CHAPTER: Chapter temporal sequence
- HAS_KEYFRAME: Chapter → Keyframe hierarchy
- HAS_EVENT/IN_CHAPTER: Chapter ↔ Event hierarchy
- NEXT_EVENT/PREV_EVENT: Event temporal sequence
- CONTAINS/APPEARS_IN: Event ↔ Object participation
- SIMILAR_TO: Event semantic similarity
- CAUSES/RESULT_OF: Event causal relationships

Components:
- GraphConstructionStep: Main pipeline step
- GraphBuilder: Creates nodes and primary edges
- GraphLinker: Creates SIMILAR_TO edges
- CausalLinker: Creates CAUSES/RESULT_OF edges
"""

from .step import GraphConstructionStep
from .builder import GraphBuilder, GraphBuildResult
from .linker import GraphLinker, GraphLinkResult
from .causal_linker import (
    HeuristicCausalDetector,
    CausalDetectionConfig,
    CausalLinkResult,
    CausalCandidate,
    LLMCausalValidator,
    CausalLinker,
)

__all__ = [
    "GraphConstructionStep",
    "GraphBuilder",
    "GraphBuildResult",
    "GraphLinker",
    "GraphLinkResult",
    "HeuristicCausalDetector",
    "CausalDetectionConfig",
    "CausalLinkResult",
    "CausalCandidate",
    "LLMCausalValidator",
    "CausalLinker",
]
