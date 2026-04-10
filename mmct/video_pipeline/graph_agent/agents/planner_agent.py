"""Planner agent for the graph (swarm-based) pipeline.

Orchestrates the query pipeline:
1. Analyzes user queries to determine required granularity levels
2. Creates natural language plans for VideoAgent
3. Synthesizes final answers with citations from retrieved evidence
4. Optionally hands off to Critic for validation

Uses AutoGen's handoff mechanism for agent communication.
"""

import json
from typing import Any, Dict, Optional, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents._assistant_agent import HandoffBase
from autogen_core.model_context import ChatCompletionContext
from autogen_core.tools import FunctionTool
from pydantic import BaseModel

from mmct.video_pipeline.graph_agent.schemas import QueryResponse
from mmct.video_pipeline.graph_agent.prompts.planner import (
    PLANNER_SYSTEM_PROMPT_WITH_CRITIC,
    PLANNER_SYSTEM_PROMPT_WITHOUT_CRITIC,
    _format_prompt,
)

from loguru import logger


# ---------------------------------------------------------------------------
# Counting handoff — enforces per-agent handoff limits at code level
# ---------------------------------------------------------------------------

MAX_HANDOFFS_PER_AGENT = 2

# Module-level mutable counter shared across all CountingHandoff instances.
# Reset via reset_handoff_counter() before each query.
_handoff_counter: Dict[str, int] = {}


def reset_handoff_counter() -> None:
    """Reset handoff counts. Call before each new query."""
    _handoff_counter.clear()


class CountingHandoff(HandoffBase):
    """A Handoff that blocks after MAX_HANDOFFS_PER_AGENT invocations."""

    @property
    def handoff_tool(self):  # type: ignore[override]
        target = self.target
        max_calls = MAX_HANDOFFS_PER_AGENT
        normal_message = self.message

        def _counted_handoff() -> str:
            count = _handoff_counter.get(target, 0)
            if count >= max_calls:
                return (
                    f"HANDOFF_BLOCKED: You have already handed off to {target} "
                    f"{count} times (limit {max_calls}). You MUST now call "
                    f"submit_final_answer with whatever evidence you have."
                )
            _handoff_counter[target] = count + 1
            return normal_message

        return FunctionTool(
            _counted_handoff,
            name=self.name,
            description=self.description,
            strict=True,
        )


SUBMIT_TOOL_NAME = "submit_final_answer"


def submit_final_answer(answer: str, sources: List[Dict[str, Any]]) -> str:
    """Submit the final answer to the user. This ends the conversation.

    Args:
        answer: Human-readable answer with inline citations [1], [2], etc.
                Must contain ONLY the answer text — no source lists, no timestamps,
                no video IDs, no graph terms (ChapterGroup, Chapter, etc.).
        sources: List of source objects. Each object must have:
                 citation (str like "[1]"), video_id (str), start_time (float), end_time (float).
                 Use an empty list [] if no sources.

    Returns:
        Confirmation that the answer was submitted.
    """
    response = {"answer": answer, "sources": sources}
    return json.dumps(response)


class PlannerAgent:
    """Planner agent using AutoGen handoffs.

    Orchestrates the query pipeline:
    1. Creates natural language plans based on query analysis
    2. Delegates to VideoAgent for graph retrieval
    3. Delegates to ImageAgent for visual analysis (if needed)
    4. Synthesizes final answer with citations
    5. Optionally validates with Critic

    Attributes:
        model_client: AutoGen model client for LLM calls.
        use_critic: Whether to use Critic for validation.
        video_catalog: Optional compact catalog of available videos.
        agent: The underlying AutoGen AssistantAgent.
    """

    def __init__(
        self,
        model_client,
        use_critic: bool = True,
        model_context: Optional[ChatCompletionContext] = None,
        video_catalog: Optional[str] = None,
    ):
        """Initialize the Planner agent.

        Args:
            model_client: AutoGen model client.
            use_critic: Whether to enable Critic validation.
            model_context: Optional shared model context for KV cache.
            video_catalog: Optional pre-generated catalog of available videos.
        """
        self.model_client = model_client
        self.use_critic = use_critic
        self.model_context = model_context
        self.video_catalog = video_catalog
        self.agent = self._create_agent()

    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen AssistantAgent."""
        handoff_targets = ["VideoAgent", "ImageAgent"]
        if self.use_critic:
            handoff_targets.append("critic")
            system_message = _format_prompt(PLANNER_SYSTEM_PROMPT_WITH_CRITIC)
        else:
            system_message = _format_prompt(PLANNER_SYSTEM_PROMPT_WITHOUT_CRITIC)

        if self.video_catalog:
            system_message += (
                "\n\n# VIDEO CATALOG\n\n"
                "The following is a summary of the videos available in the knowledge graph. "
                "Use this to inform your retrieval plan.\n\n"
                f"{self.video_catalog}"
            )

        handoffs = [CountingHandoff(target=t) for t in handoff_targets]

        return AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Orchestrator that analyzes queries, creates plans, and synthesizes answers with citations.",
            system_message=system_message,
            tools=[submit_final_answer],
            reflect_on_tool_use=False,
            handoffs=handoffs,
        )
