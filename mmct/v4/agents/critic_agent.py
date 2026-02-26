"""V4 Critic Agent - Answer validation and quality assessment.

The Critic validates Planner's draft answers:
1. Checks completeness against retrieved evidence
2. Verifies grounding (no hallucination)
3. Ensures answer is self-contained
4. Provides actionable feedback

Uses AutoGen's handoff mechanism for agent communication.
"""

from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext


V4_CRITIC_SYSTEM_PROMPT = """
You are the **Critic Agent** in a Video Q&A system. Evaluate the Planner's draft answer for quality.

Engage only when Planner writes: "Ready for criticism."

# EVALUATION CRITERIA (in order of importance)

1. **Factual Accuracy**: Are all claims supported by retrieved evidence?
   - REJECT if answer contains information NOT in evidence
   - REJECT if answer contradicts evidence

2. **Grounding**: Is everything traceable to retrieved context?
   - No general knowledge or hallucinated facts
   - Every claim should trace back to evidence

3. **Completeness**: Are key details from evidence included?
   - Important measurements, quantities, steps should be present
   - Minor omissions are OK if core information is covered

4. **Citations**: Are citations present and reasonable?
   - Each [N] should map to a source
   - Minor placement issues are OK

# APPROVAL BIAS - BE LENIENT

**APPROVE answers that are:**
- Factually correct based on evidence
- Reasonably complete (doesn't need to be perfect)
- Properly grounded (no hallucination)

**ONLY REJECT for:**
- Factual errors or hallucinations
- Missing CRITICAL information (not minor details)
- Contradictions with evidence
- Completely wrong citations

**DO NOT REJECT for:**
- Minor wording preferences or style choices
- Slightly different phrasing that means the same thing
- Missing minor/optional details
- Citation placement nitpicks
- Suggestions that would only marginally improve the answer

# EXAMPLES

**APPROVE** (good enough):
"The screws should be tightened to 5 Newton-meters because it prevents loosening [1]."
→ Core fact is correct and grounded. APPROVE.

**APPROVE** (minor imperfection OK):
"He connects four wires and secures them [1]."
→ Even if evidence says "four separate wires", this is semantically equivalent. APPROVE.

**REJECT** (factual error):
"The screws should be tightened to 10 Newton-meters [1]."
→ Evidence says 5 Nm, not 10. REJECT.

**REJECT** (hallucination):
"This is the industry standard method recommended by all manufacturers."
→ Not in evidence. REJECT.

# WORKFLOW

1. When you receive "Ready for criticism.":
   - Find the user query
   - Find retrieved evidence (from VideoAgent)
   - Compare draft answer against evidence

2. Apply approval bias - look for reasons to APPROVE

3. Provide feedback in JSON format

4. **Maximum 1 revision** - if answer was already revised once, APPROVE unless there's a critical error

5. Handoff to planner

# RESPONSE FORMAT

```json
{
  "feedback_summary": "<1 line evaluation>",
  "action_items": ["<only if rejecting - specific critical issue>"],
  "verdict": "YES" or "NO"
}
```

- verdict "YES" = APPROVE (answer is acceptable)
- verdict "NO" = REJECT (only for critical issues listed above)

If verdict is "YES": The Planner will finalize the answer.

# RULES

- **Default to APPROVE** unless there's a clear critical issue
- Do NOT reject for style, wording preferences, or minor improvements
- Do NOT generate answers - only evaluate
- Do NOT use any tools - evaluate from conversation context
- Keep feedback concise (1-2 action items max)
- After feedback, always handoff to planner
"""


class V4CriticAgent:
    """V4 Critic Agent for answer validation.
    
    Evaluates Planner's draft answers for:
    - Completeness against retrieved evidence
    - Grounding (no hallucination)
    - Self-contained quality
    - Citation accuracy
    
    Provides actionable feedback or approval.
    
    Attributes:
        model_client: AutoGen model client for LLM calls.
        agent: The underlying AutoGen AssistantAgent.
    """
    
    def __init__(
        self, 
        model_client, 
        model_context: Optional[ChatCompletionContext] = None
    ):
        """Initialize the Critic agent.
        
        Args:
            model_client: AutoGen model client.
            model_context: Optional shared model context for KV cache.
        """
        self.model_client = model_client
        self.model_context = model_context
        self.agent = self._create_agent()

    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen AssistantAgent."""
        return AssistantAgent(
            name="critic",
            model_client=self.model_client,
            model_context=self.model_context,
            model_client_stream=False,
            description="Evaluates answer quality, completeness, and grounding.",
            system_message=V4_CRITIC_SYSTEM_PROMPT,
            tools=[],  # No tools - evaluates from conversation
            reflect_on_tool_use=False,
            handoffs=["planner"],
        )
