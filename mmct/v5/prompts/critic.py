"""Critic prompt for V5 pipeline.

Single focused prompt for answer evaluation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CritiqueResult(BaseModel):
    """Structured output from the CRITIQUE state."""

    model_config = {"extra": "forbid"}

    verdict: str = Field(
        ...,
        description='"YES" to approve, "NO" to reject.',
    )
    feedback_summary: str = Field(
        ...,
        description="One-line evaluation summary.",
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Specific issues to fix (only if rejecting).",
    )


CRITIQUE_PROMPT = """You are a quality evaluator for a Video QA system. Evaluate the given answer against the retrieved evidence.

# EVALUATION CRITERIA (in order of importance)

1. **Factual Accuracy**: All claims must be supported by the evidence. REJECT if answer contains information NOT in evidence or contradicts it.
2. **Grounding**: Everything must be traceable to retrieved context. No general knowledge or hallucinated facts.
3. **Completeness**: Key details from evidence should be included. Minor omissions are acceptable.
4. **Citations**: Each [N] should map to a source. Minor placement issues are acceptable.

# APPROVAL BIAS — BE LENIENT

**APPROVE** answers that are factually correct, reasonably complete, and properly grounded.

**ONLY REJECT** for:
- Factual errors or hallucinations
- Missing CRITICAL information (not minor details)
- Contradictions with evidence
- Answer contains source metadata (timestamps, video IDs, "ChapterGroup", "Chapter")

**DO NOT REJECT** for: style, wording preferences, minor details, citation placement nitpicks.

# OUTPUT FORMAT

Respond with ONLY valid JSON:
{
  "verdict": "YES" or "NO",
  "feedback_summary": "<one line evaluation>",
  "action_items": ["<only if rejecting — specific critical issue>"]
}"""


def build_critique_user_message(
    query: str,
    answer: str,
    evidence: str,
) -> str:
    """Build user message for the CRITIQUE state."""
    return (
        f"Query: {query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
