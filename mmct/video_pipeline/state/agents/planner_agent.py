"""Planner agent for the state machine pipeline.

Wraps the PLAN state handler as a class-based agent.
"""

from loguru import logger

from mmct.video_pipeline.state.llm.client import StructuredLLMClient
from mmct.video_pipeline.state.state_machine import QueryContext, QueryState
from mmct.video_pipeline.state.prompts.planner import (
    PLAN_QUERY_PROMPT,
    PlanResult,
    build_plan_user_message,
)


class PlannerAgent:
    """Decomposes a user query into a structured retrieval plan.

    Args:
        model_client: AutoGen ChatCompletionClient from get_llm_provider().get_autogen_client().
    """

    def __init__(self, model_client):
        self.llm = StructuredLLMClient(model_client)

    async def run(self, ctx: QueryContext) -> QueryState:
        """LLM decomposes query into structured plan.

        Reads ctx.query, ctx.video_scope, ctx.effective_video_ids, ctx.video_catalog.
        Writes ctx.plan.

        Returns:
            VALIDATE_PLAN on success, ERROR on failure.
        """
        user_msg = build_plan_user_message(
            query=ctx.query,
            video_scope=ctx.video_scope,
            video_ids=ctx.effective_video_ids,
            video_catalog=ctx.video_catalog,
        )

        try:
            plan = await self.llm.call_typed(
                system_prompt=PLAN_QUERY_PROMPT,
                user_message=user_msg,
                response_type=PlanResult,
            )
            ctx.plan = plan.model_dump()
            ctx.add_tokens(self.llm.total_prompt_tokens, self.llm.total_completion_tokens)
            self.llm.reset_usage()
            return QueryState.VALIDATE_PLAN

        except Exception as e:
            logger.error(f"[{ctx.request_id}] PLAN failed: {e}")
            ctx.add_tokens(self.llm.total_prompt_tokens, self.llm.total_completion_tokens)
            self.llm.reset_usage()
            ctx.answer = f"Failed to plan query: {e}"
            return QueryState.ERROR
