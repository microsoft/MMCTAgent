"""Chapter grouping pipeline step.

Groups semantically similar consecutive chapters using embedding similarity
and temporal proximity, then generates LLM summaries for each group.

Components:
- ChapterGroupingStep: Main pipeline step
- ChapterGrouper: Sliding window grouping algorithm
- GroupSummarizer: LLM-based group summarization

Grouping Algorithm:
1. Extract embeddings from chapters (if available)
2. Sliding window groups consecutive chapters with similarity >= threshold
3. Temporal window constraint limits max chapter distance in a group
4. Generate LLM summaries for hierarchical organization
"""

from .step import ChapterGroupingStep
from .grouper import ChapterGrouper
from .summarizer import GroupSummarizer

__all__ = [
    "ChapterGroupingStep",
    "ChapterGrouper",
    "GroupSummarizer",
]
