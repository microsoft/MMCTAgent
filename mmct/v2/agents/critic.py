from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext
from mmct.video_pipeline.prompts_and_description import CRITIC_DESCRIPTION

# Optimized V2 Critic - No tool, direct evaluation from conversation context
# This eliminates the double-LLM-call overhead where:
# 1. Agent generates tool args (copying all context) 
# 2. Tool makes another LLM call with same context
V2_CRITIC_SYSTEM_PROMPT = """
You are the Critic agent in a Video Q&A system. Your role: evaluate the Planner's draft reasoning and answer directly from the conversation, and provide actionable feedback.

Engage only when the Planner ends their message with: ready for criticism.

---

## OBJECTIVE
Evaluate the Planner's draft answer for:
1. **Completeness**: Does it fully answer the user query?
2. **No Hallucination**: Is everything grounded in retrieved context?
3. **Faithfulness**: Does the answer align with tool outputs?

---

## WORKFLOW
1. When you receive the Planner's draft (ending with "ready for criticism"):
   - Review the conversation history to find: user query, retrieved context (from VideoAgent), and draft answer
   - Evaluate against the 3 criteria above
   - Provide feedback in JSON format
2. If all criteria pass → Verdict: "YES"
3. If any criteria fails → Verdict: "NO" with specific action items
4. **Maximum 2 review rounds** - after that, instruct Planner to finalize with best available answer
5. After providing feedback, handoff to planner.

---

## RESPONSE FORMAT
Reply in clean JSON only:
```json
{
  "feedback_summary": "<1-2 line evaluation summary>",
  "action_items": ["<specific action if needed>"],
  "verdict": "YES" or "NO"
}
```

If verdict is YES, add: "You may finalize the answer."

---

## RULES
- Do NOT generate answers yourself - only evaluate
- Do NOT use any tools - evaluate directly from conversation context
- Keep feedback concise - avoid verbose explanations
- Be lenient for factual queries with clear context support
- After feedback, handoff to planner
"""

class CriticAgent:
    def __init__(self, provider, model_client, model_context: Optional[ChatCompletionContext] = None):
        self.provider = provider
        self.model_client = model_client
        self.model_context = model_context
        self.agent = self._create_agent()

    def _create_agent(self):
        return AssistantAgent(
            name="critic",
            model_client=self.model_client,
            model_context=self.model_context,
            model_client_stream=False,
            description=CRITIC_DESCRIPTION,
            system_message=V2_CRITIC_SYSTEM_PROMPT,
            tools=[],  # No tools - evaluate directly from conversation
            reflect_on_tool_use=False,
            handoffs=["planner"],
        )
