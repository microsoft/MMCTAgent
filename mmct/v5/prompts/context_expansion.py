"""Prompt and schema for the EXPAND_CONTEXT state.

The LLM examines the query and retrieved evidence, then decides whether
additional graph traversals are needed to give the synthesizer enough
context for an accurate, well-cited answer.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class TraversalOp(BaseModel):
    """A single graph traversal to execute."""

    node_ids: List[str] = Field(
        description="Source node IDs to traverse from (e.g. ['chapter_E7VhAVeKUk8_009'])."
    )
    target: str = Field(
        description="Target node type to traverse to. One of: ChapterGroup, Chapter, Event, Transcript, Keyframe, Object."
    )
    reason: str = Field(
        description="Brief reason this expansion is needed."
    )

    model_config = {"extra": "forbid"}


class ContextExpansionResult(BaseModel):
    """LLM decision on whether and how to expand context."""

    needs_expansion: bool = Field(
        description="True if additional graph traversals would improve the answer."
    )
    operations: List[TraversalOp] = Field(
        default_factory=list,
        description="Traversal operations to execute. Empty if needs_expansion is false.",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXPAND_CONTEXT_PROMPT = """You are a context expansion planner for a Video QA system backed by a Neo4j knowledge graph.

You are given:
1. The user's original query
2. The evidence already retrieved (node summaries with IDs, types, video_ids, and time ranges)

Your job is to decide whether the retrieved evidence is sufficient for a complete answer, or whether additional graph traversals would improve accuracy and citation quality.

# KNOWLEDGE GRAPH STRUCTURE

```
ChapterGroup  — High-level topic sections (broad summary, list of topics)
  └─ HAS_CHAPTER → Chapter  — 3-5 min segments with multimodal summaries
       ├─ HAS_TRANSCRIPT → Transcript  — Raw speech text (same time range as Chapter)
       ├─ HAS_EVENT → Event  — Atomic actions/moments (5-30s each)
       │    └─ CONTAINS → Object  — Entities: people, items, text on screen
       └─ HAS_KEYFRAME → Keyframe  — Visual frames
```

Temporal navigation: Chapters and Events are ordered by chunk_index within their parent.

# TRAVERSAL DIRECTIONS

You can traverse in any direction:
- **UP** (child → parent): Chapter → ChapterGroup, Event → Chapter, Transcript → Chapter, etc.
- **DOWN** (parent → child): ChapterGroup → Chapter, Chapter → Event, Chapter → Transcript, etc.
- **SIBLING** (via shared parent): Event → Keyframe (via Chapter), Event → Transcript (via Chapter)

# WHEN TO EXPAND

- **Definition / introduction queries** ("defines X", "what is X", "introduce"): If the earliest matched Chapter has chunk_index > 0, traverse UP to ChapterGroup to find the topic's true start.
- **Process / step-by-step queries**: If only Chapters are retrieved, traverse DOWN to Events for finer-grained steps.
- **Quote / verbatim queries**: If only Chapters are retrieved, traverse DOWN to Transcript for exact words.
- **"What happens before/after" queries**: Traverse UP to ChapterGroup to find surrounding context.
- **Entity queries**: If Events are retrieved but entity details are missing, traverse DOWN to Objects.

# WHEN NOT TO EXPAND

- The evidence already covers the topic comprehensively (multiple relevant chunks from different angles).
- The query is broad/comparative — the current Chapter-level evidence is sufficient.
- Adding more context would just add noise without improving the answer.
- The retrieved evidence already starts at chunk_index 0 (already at topic start).

# RULES

- Return `needs_expansion: false` with empty operations if no expansion is needed. Be conservative — only expand when it clearly helps.
- Each operation specifies source node_ids and a target type.
- Keep operations minimal — at most 2-3 traversals. Don't over-fetch.
- Use node_ids from the evidence provided — do NOT invent node IDs.

# OUTPUT FORMAT

Respond with ONLY valid JSON matching this schema:
{
  "needs_expansion": true/false,
  "operations": [
    {
      "node_ids": ["chapter_ABC_003"],
      "target": "ChapterGroup",
      "reason": "Query asks for definition; chapter starts mid-topic, need parent for true start"
    }
  ]
}"""


def build_expansion_user_message(query: str, evidence: list) -> str:
    """Format the user message with query and evidence summaries."""
    lines = [f"# USER QUERY\n{query}\n", "# RETRIEVED EVIDENCE\n"]

    for i, item in enumerate(evidence, 1):
        node_id = item.get("node_id", "?")
        node_type = item.get("node_type", "?")
        video_id = item.get("video_id", "?")
        start = item.get("start_time", "?")
        end = item.get("end_time", "?")
        chunk_idx = item.get("chunk_index", "?")
        summary = item.get("summary", item.get("text", ""))
        if isinstance(summary, str) and len(summary) > 200:
            summary = summary[:200] + "..."

        lines.append(
            f"[{i}] node_id={node_id} type={node_type} video={video_id} "
            f"time=[{start}-{end}] chunk_index={chunk_idx}\n"
            f"    {summary}\n"
        )

    return "\n".join(lines)
