"""Critic agent for the state machine pipeline.

Wraps the CRITIQUE state handler as a class-based agent.
"""

import json

from loguru import logger

from mmct.video_pipeline.state.llm.client import StructuredLLMClient
from mmct.video_pipeline.state.state_machine import (
    QueryContext,
    QueryState,
    MAX_REVISE_ATTEMPTS,
)
from mmct.video_pipeline.state.prompts.critic import (
    CRITIQUE_PROMPT,
    CritiqueResult,
    build_critique_user_message,
)
from mmct.video_pipeline.state._helpers import format_evidence_compact


class CriticAgent:
    """Evaluates answer quality and decides whether to revise or submit.

    Args:
        model_client: AutoGen ChatCompletionClient from get_llm_provider().get_autogen_client().
    """

    def __init__(self, model_client):
        self.llm = StructuredLLMClient(model_client)

    async def run(self, ctx: QueryContext) -> QueryState:
        """LLM evaluates answer quality.

        Reads ctx.query, ctx.answer, ctx.evidence, ctx.image_analyses, ctx.revise_attempts.
        Writes ctx.critic_feedback.

        Returns:
            SUBMIT if answer is acceptable or max revisions reached.
            REVISE if the answer needs improvement.
        """
        evidence_str = format_evidence_compact(ctx.evidence)
        image_str = (
            json.dumps(ctx.image_analyses, indent=2, default=str)
            if ctx.image_analyses
            else None
        )
        user_msg = build_critique_user_message(
            query=ctx.query,
            answer=ctx.answer or "",
            evidence=evidence_str,
            image_analyses=image_str,
        )

        try:
            result = await self.llm.call_typed(
                system_prompt=CRITIQUE_PROMPT,
                user_message=user_msg,
                response_type=CritiqueResult,
            )
            ctx.critic_feedback = result.model_dump()
            ctx.add_tokens(self.llm.total_prompt_tokens, self.llm.total_completion_tokens)
            self.llm.reset_usage()

            if result.verdict.upper() == "YES":
                return QueryState.SUBMIT

            if ctx.revise_attempts < MAX_REVISE_ATTEMPTS:
                return QueryState.REVISE

            return QueryState.SUBMIT

        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Critique failed: {e} — submitting anyway")
            ctx.add_tokens(self.llm.total_prompt_tokens, self.llm.total_completion_tokens)
            self.llm.reset_usage()
            return QueryState.SUBMIT
