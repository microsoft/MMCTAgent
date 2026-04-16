"""Critic agent for the graph (swarm-based) pipeline.

Validates Planner's draft answers:
1. Checks completeness against retrieved evidence
2. Verifies grounding (no hallucination)
3. Ensures answer is self-contained
4. Provides actionable feedback

Uses AutoGen's handoff mechanism for agent communication.
"""

from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext

from mmct.video_pipeline.graph_agent.prompts.critic import CRITIC_SYSTEM_PROMPT


class CriticAgent:
    """Critic agent for answer validation.

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
        model_context: Optional[ChatCompletionContext] = None,
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
            system_message=CRITIC_SYSTEM_PROMPT,
            tools=[],  # Evaluates from conversation context only
            reflect_on_tool_use=False,
            handoffs=["planner"],
        )
