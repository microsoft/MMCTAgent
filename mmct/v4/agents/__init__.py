"""V4 Agents module for MMCTAgent.

Provides the V4 agent system using AutoGen Swarm:
- V4PlannerAgent: Graph-aware query planning with AutoGen handoffs
- V4VideoAgent: Graph retrieval and keyframe decisions with AutoGen handoffs
- V4ImageAgent: Visual frame analysis with AutoGen handoffs
- V4CriticAgent: Answer validation and quality assessment with AutoGen handoffs
"""

from mmct.v4.agents.planner_agent import (
    V4PlannerAgent,
    V4_PLANNER_SYSTEM_PROMPT,
    V4_PLANNER_SYSTEM_PROMPT_WITH_CRITIC,
    V4_PLANNER_SYSTEM_PROMPT_WITHOUT_CRITIC,
)
from mmct.v4.agents.video_agent import (
    V4VideoAgent,
    V4_VIDEO_AGENT_SYSTEM_PROMPT,
)
from mmct.v4.agents.image_agent import (
    V4ImageAgent,
    V4_IMAGE_AGENT_SYSTEM_PROMPT,
)
from mmct.v4.agents.critic_agent import (
    V4CriticAgent,
    V4_CRITIC_SYSTEM_PROMPT,
)

__all__ = [
    # Agents
    "V4PlannerAgent",
    "V4VideoAgent",
    "V4ImageAgent",
    "V4CriticAgent",
    # System prompts
    "V4_PLANNER_SYSTEM_PROMPT",
    "V4_PLANNER_SYSTEM_PROMPT_WITH_CRITIC",
    "V4_PLANNER_SYSTEM_PROMPT_WITHOUT_CRITIC",
    "V4_VIDEO_AGENT_SYSTEM_PROMPT",
    "V4_IMAGE_AGENT_SYSTEM_PROMPT",
    "V4_CRITIC_SYSTEM_PROMPT",
]
