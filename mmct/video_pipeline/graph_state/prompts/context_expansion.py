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
        description="Target node type to traverse to. One of: SiblingChapters, Event, Transcript."
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
  └─ HAS_CHAPTER → Chapter  — 3-5 min segments with timestamped descriptions
       ├─ HAS_TRANSCRIPT → Transcript  — Raw speech text (same time range as Chapter)
       ├─ HAS_EVENT → Event  — Atomic actions/moments (5-30s each)
       │    └─ CONTAINS → Object  — Entities: people, items, text on screen
       └─ HAS_KEYFRAME → Keyframe  — Visual frames
```

Temporal navigation: Chapters and Events are ordered by chunk_index within their parent.

# AVAILABLE OPERATIONS

1. **SIBLINGS** (target: "SiblingChapters"): Given a Chapter node_id, fetch ALL sibling Chapters that share the same parent ChapterGroup. Use this to get surrounding context — the chapters before and after the matched one.
2. **DOWN to Event** (target: "Event"): Given a Chapter node_id, fetch its child Event nodes for fine-grained step-by-step details.
3. **DOWN to Transcript** (target: "Transcript"): Given a Chapter node_id, fetch raw speech text for exact quotes.

# WHEN TO EXPAND

- **Definition / introduction / evolution queries**: If the earliest matched Chapter has chunk_index > 0, use SIBLINGS to get the full topic context (preceding and following chapters in the group).
- **"What happens before/after" queries**: Use SIBLINGS to get surrounding chapters.
- **Quote / verbatim queries**: If only Chapters are retrieved, traverse DOWN to Transcript for exact words.
- **ONLY use DOWN to Event when Chapter timestamped descriptions lack sufficient detail** — e.g., the query asks for specific step-by-step actions and the chapter summary is too vague. Chapters already contain [Xs] timestamped lines, so Events are rarely needed.

# WHEN NOT TO EXPAND

- The evidence already covers the topic comprehensively (multiple relevant chunks from different angles).
- The query is broad/comparative — the current Chapter-level evidence is sufficient.
- Adding more context would just add noise without improving the answer.
- The retrieved evidence already starts at chunk_index 0 (already at topic start).

# RULES

- Return `needs_expansion: false` with empty operations if no expansion is needed. Be conservative — only expand when it clearly helps.
- Each operation specifies source node_ids and a target type.
- **At most 1 expansion operation.** Do NOT chain multiple traversals.
- You MAY include **multiple node_ids** in a single operation to expand several chapters at once.
- Use node_ids from the evidence provided — do NOT invent node IDs.

# OUTPUT FORMAT

Respond with ONLY valid JSON matching this schema:
{
  "needs_expansion": true/false,
  "operations": [
    {
      "node_ids": ["chapter_ABC_003", "chapter_DEF_001"],
      "target": "SiblingChapters",
      "reason": "Query asks for evolution across lectures; need surrounding chapters for full context"
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
