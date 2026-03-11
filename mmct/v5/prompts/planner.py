"""Planner prompts for V5 pipeline.

Three focused prompts instead of one monolithic system prompt:
- PLAN_QUERY_PROMPT: Decompose query into structured retrieval plan
- SYNTHESIZE_PROMPT: Write answer with citations from evidence
- REPHRASE_PROMPT: Rephrase sub-queries when retrieval returns empty

Each has a corresponding Pydantic model for structured output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class PlanResult(BaseModel):
    """Structured output from the PLAN state."""

    model_config = {"extra": "forbid"}

    strategy: str = Field(
        ...,
        description='Either "SEARCH" (vector similarity) or "OVERVIEW" (fetch all nodes).',
    )
    targets: List[str] = Field(
        ...,
        description='Node types to search, e.g. ["Chapter"]. Valid: ChapterGroup, Chapter, Transcript, Event, Object.',
    )
    sub_queries: List[str] = Field(
        ...,
        description="2-4 focused sub-queries, each 5-15 words, targeting a single concept.",
    )
    visual: bool = Field(
        default=False,
        description="True if the query asks about anything visible (diagrams, charts, visual layouts).",
    )
    limit: int = Field(
        default=5,
        description="Max results per sub-query per target type.",
    )


class CitationSourceOutput(BaseModel):
    """A single citation source in the synthesis output."""

    model_config = {"extra": "forbid"}

    citation: str = Field(..., description='Citation marker, e.g. "[1]"')
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")


class SynthesisResult(BaseModel):
    """Structured output from the SYNTHESIZE state."""

    model_config = {"extra": "forbid"}

    answer: str = Field(
        ...,
        description="Complete answer text with inline citation markers [1], [2], etc.",
    )
    sources: List[CitationSourceOutput] = Field(
        default_factory=list,
        description="List of citation source objects.",
    )


class RephraseResult(BaseModel):
    """Structured output from the REPHRASE state."""

    model_config = {"extra": "forbid"}

    sub_queries: List[str] = Field(
        ...,
        description="Rephrased sub-queries using different terminology.",
    )


# ---------------------------------------------------------------------------
# PLAN prompt — produces PlanResult
# ---------------------------------------------------------------------------

PLAN_QUERY_PROMPT = """You are a query planner for a Video QA system backed by a Neo4j knowledge graph.

Analyze the user query and produce a structured retrieval plan as JSON.

# KNOWLEDGE GRAPH STRUCTURE

```
ChapterGroup  — High-level topic sections (broad summary, list of topics)
  └─ Chapter  — 3-5 min segments with multimodal summaries (visual + verbal)
       ├─ Transcript  — Raw speech text (1:1 with Chapter, same time range)
       ├─ Event  — Atomic actions: dialogue, action, transition, state_change
       │    └─ Object  — Entities: people, items, text on screen
       └─ Keyframe  — Visual frames (image search only)
```

# STRATEGY

Pick ONE:
- **SEARCH** — Vector similarity search. Use for all queries except pure overview/summary.
- **OVERVIEW** — Fetch all nodes without vector search. Use ONLY for "what is this about?", "summarize this video", "list all topics".

# TARGETS

Choose the node type(s) to search:
- `["Chapter"]` — Default. Answers MOST queries about topics, explanations, comparisons, concepts.
- `["Chapter", "Event"]` — ONLY for step-by-step processes, specific actions, "what happens when X".
- `["Transcript"]` — ONLY for verbatim quotes or "what exactly did they say".
- `["Object"]` — ONLY for "who/what entity appears" queries.
- `["ChapterGroup"]` — ONLY for very broad discovery ("what topics exist across videos").

Chapter and Transcript overlap (same time segments). NEVER search both.

# SUB-QUERIES

Decompose the user query into 2-4 focused sub-queries:
- Each should be 5-15 words targeting a single concept
- Consider synonyms and alternate phrasings
- All will be searched in parallel
- For definition queries ("defines X", "what is X", "explain concept X"), make sub-queries target the definition itself: "definition of X", "X is defined as", "what is X", "X means"
- For navigation queries ("take me to", "show me where"), make sub-queries match the specific content the user wants to locate

# VISUAL FLAG

Set to true when the query asks about anything VISIBLE — diagrams, charts, tables, formulas on screen, visual layouts, "what does X look like", colors, annotations, or anything that cannot be answered from text summaries alone.

# OUTPUT FORMAT

Respond with ONLY valid JSON matching this schema:
{
  "strategy": "SEARCH" or "OVERVIEW",
  "targets": ["Chapter"],
  "sub_queries": ["focused query 1", "focused query 2"],
  "visual": false,
  "limit": 5
}"""


# ---------------------------------------------------------------------------
# SYNTHESIZE prompt — produces SynthesisResult
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """You are an answer synthesizer for a Video QA system. Write a complete answer using ONLY the evidence provided.

# ANSWER RULES

- ONLY use information from the evidence below. NEVER hallucinate or add external knowledge.
- **Be concise and to-the-point.** Answer directly without filler, preambles, or conversational padding. Do not restate the question.
- **The answer must be self-contained.** The reader should fully understand the answer WITHOUT watching the cited videos. Include all necessary context and explanations from the evidence.
  - **Exception:** When the user asks to be taken to a specific video part (e.g., "take me to where...", "show me the part where...", "find where he defines..."), keep the answer brief — 1-3 sentences giving a short description and the location. Only cite the single best matching segment.
- Include specific details: measurements, quantities, steps.

# CITATION RULES

- Every claim needs a citation [1], [2], etc. Each citation = one source with video_id + start_time + end_time (REQUIRED numbers, never null).
- **Citation accuracy is critical.** A citation must point to evidence that DIRECTLY supports the specific claim. Do NOT cite evidence that merely mentions the topic tangentially.
- **For definition queries** ("defines X", "what is X"): only cite the evidence chunk where the concept is actually defined or formally introduced — NOT chunks that merely use or reference the concept.
- **Minimize citations.** Merge chapters from the same video covering the same topic into ONE citation with the widest time range.
- **Use ALL relevant evidence.** If results come from multiple videos, cite ALL of them.
- **Filter out tangential matches.** Only cite results that directly address the query. Discard evidence that mentions the topic in passing without providing the requested information.
- Do NOT mention internal graph terms (ChapterGroup, Chapter, Event, Object, Keyframe, node, graph).
- Do NOT mention video IDs in the answer text. Use natural language ("the video explains...", "the lecture covers...").
- The answer field must contain ONLY readable text with [1], [2] markers. No source lists or metadata.

# OUTPUT FORMAT

Respond with ONLY valid JSON:
{
  "answer": "<answer text with [1], [2] citations>",
  "sources": [
    {"citation": "[1]", "video_id": "<id>", "start_time": <float>, "end_time": <float>}
  ]
}

CRITICAL: start_time and end_time MUST be numbers (never null). If unsure, use 0.0."""


# ---------------------------------------------------------------------------
# REPHRASE prompt — produces RephraseResult
# ---------------------------------------------------------------------------

REPHRASE_PROMPT = """The previous search sub-queries returned no results from the knowledge graph. Rephrase them using different terminology, synonyms, or broader/narrower phrasing.

Rules:
- Keep 2-4 sub-queries, each 5-15 words
- Use different vocabulary than the originals
- Try broader terms if originals were too specific
- Try more specific terms if originals were too broad

Respond with ONLY valid JSON:
{
  "sub_queries": ["rephrased query 1", "rephrased query 2"]
}"""


def build_plan_user_message(
    query: str,
    video_scope: str,
    video_ids: Optional[List[str]] = None,
    video_catalog: Optional[str] = None,
) -> str:
    """Build the user message for the PLAN state."""
    parts = [f"Query: {query}", f"Video scope: {video_scope}"]
    if video_ids:
        parts.append(f"Video IDs: {', '.join(video_ids)}")
    if video_catalog:
        parts.append(f"\nVideo catalog:\n{video_catalog}")
    return "\n".join(parts)


def build_synthesize_user_message(
    query: str,
    evidence: str,
    image_analyses: Optional[str] = None,
) -> str:
    """Build the user message for the SYNTHESIZE state."""
    parts = [f"Query: {query}", f"\nEvidence:\n{evidence}"]
    if image_analyses:
        parts.append(f"\nVisual analysis:\n{image_analyses}")
    return "\n".join(parts)


def build_revise_user_message(
    query: str,
    evidence: str,
    previous_answer: str,
    critic_feedback: str,
) -> str:
    """Build the user message for the REVISE state."""
    return (
        f"Query: {query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Your previous answer:\n{previous_answer}\n\n"
        f"Critic feedback:\n{critic_feedback}\n\n"
        "Revise the answer addressing the critic's feedback. "
        "Use the same JSON output format."
    )


def build_rephrase_user_message(original_sub_queries: List[str]) -> str:
    """Build the user message for the REPHRASE state."""
    import json

    return f"Original sub-queries that returned no results:\n{json.dumps(original_sub_queries)}"
