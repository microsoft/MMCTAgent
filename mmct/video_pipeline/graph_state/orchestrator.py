"""State machine orchestrator — deterministic query pipeline.

This module drives the video query pipeline through explicit states, calling 
LLMs only for high-intelligence tasks (planning, synthesis, revision) and 
executing retrieval/analysis tools programmatically.
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger

from mmct.video_pipeline.graph_state._helpers import format_evidence_compact, dedup_sources
from mmct.video_pipeline.graph_state.state_machine import (
    QueryContext,
    QueryState,
    StateTransitionError,
    MAX_RETRIEVE_ATTEMPTS,
    MAX_SUB_QUERIES,
    VALID_TARGETS,
    VALID_STRATEGIES,
    MIN_SUB_QUERIES,
)
from mmct.video_pipeline.graph_state.llm.client import StructuredLLMClient
from mmct.video_pipeline.graph_state.prompts.planner import (
    PLAN_QUERY_PROMPT,
    SYNTHESIZE_PROMPT,
    REPHRASE_PROMPT,
    PlanResult,
    SynthesisResult,
    RephraseResult,
    build_plan_user_message,
    build_synthesize_user_message,
    build_revise_user_message,
    build_rephrase_user_message,
)
from mmct.video_pipeline.graph_state.prompts.context_expansion import (
    EXPAND_CONTEXT_PROMPT,
    ContextExpansionResult,
    build_expansion_user_message,
)
from mmct.video_pipeline.graph_state.tools.retrieval import RetrievalExecutor
from mmct.video_pipeline.graph_state.tools.video_discovery import VideoDiscoveryExecutor
from mmct.video_pipeline.graph_state.tools.image_analysis import ImageAnalysisExecutor
from mmct.video_pipeline.graph_state.query.neo4j_provider import Neo4jQueryProvider
from mmct.video_pipeline.graph_state.agents.planner_agent import PlannerAgent
from mmct.video_pipeline.graph_state.agents.critic_agent import CriticAgent

_log = logger.bind(component="state")
class StateOrchestrator:
    """State machine-driven query orchestrator.

    Drives the pipeline through deterministic states, calling LLMs
    only for intelligence tasks and executing tools programmatically.

    Attributes:
        model_client: AutoGen ChatCompletionClient for LLM tasks.
        neo4j_provider: Provider for graph database access.
        storage_provider: Optional blob storage provider for keyframe downloads.
        image_llm_provider: Optional LLM provider for image (ViT) analysis.
        use_critic (bool): Whether to run the critic evaluation step.
        video_catalog (str, optional): Pre-generated catalog metadata for the planner.
    """

    def __init__(
        self,
        model_client: Any,
        neo4j_provider: Neo4jQueryProvider,
        storage_provider: Optional[Any] = None,
        image_llm_provider: Optional[Any] = None,
        use_critic: bool = True,
        video_catalog: Optional[str] = None,
        max_turns: int = 20,
    ):
        """Initializes the StateOrchestrator.

        Args:
            model_client: AutoGen model client.
            neo4j_provider: Neo4jQueryProvider instance.
            storage_provider: Optional storage provider.
            image_llm_provider: Optional vision-capable LLM provider.
            use_critic: If True, enables the critic revision cycle.
            video_catalog: Optional pre-generated metadata for the planner context.
            max_turns: Conserved for interface parity with GraphOrchestrator (unused).
        """
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.storage_provider = storage_provider
        self.image_llm_provider = image_llm_provider
        self.use_critic = use_critic
        self.video_catalog = video_catalog

        self._retrieval = RetrievalExecutor(neo4j_provider)
        self._discovery = VideoDiscoveryExecutor(neo4j_provider)
        self._image_analyzer = ImageAnalysisExecutor(
            image_llm_provider=image_llm_provider,
            storage_provider=storage_provider,
        )
        self._planner = PlannerAgent(model_client)
        self._critic = CriticAgent(model_client)

    async def query(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Processes a query through the state machine pipeline.

        Args:
            user_query: The natural language question to answer.
            video_id: Single video ID scope.
            video_ids: Multiple video ID scopes.
            request_id: Caller-provided request identifier for tracing.

        Returns:
            Dict[str, Any]: Dictionary with answer, sources, and usage metrics.
        """
        start = time.time()
        rid = request_id or "no-rid"

        _log.info(f"{'=' * 60}")
        _log.info(f"[{rid}] State Machine Pipeline")
        _log.info(f"{'=' * 60}")
        _log.info(f"[{rid}] Query: {user_query}")

        ctx = QueryContext(
            query=user_query,
            video_id=video_id,
            video_ids=video_ids,
            use_critic=self.use_critic,
            request_id=rid,
            video_catalog=self.video_catalog,
        )

        try:
            response = await self._run_state_machine(ctx)
            elapsed = time.time() - start
            response["elapsed_seconds"] = elapsed
            response["token_usage"] = ctx.token_usage

            _log.info(f"{'=' * 60}")
            _log.info(f"[{rid}] Query completed in {elapsed:.2f}s")
            _log.info(f"{'=' * 60}")
            return response
        finally:
            self._image_analyzer.cleanup()

    async def query_stream(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        request_id: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Processes a query with real-time streaming state events.

        Args:
            user_query: The question to answer.
            video_id: Optional single video ID scope.
            video_ids: Optional list of video ID scopes.
            request_id: Optional correlation ID.

        Yields:
            Dict[str, Any]: State transition messages or the final result.
        """
        start = time.time()
        rid = request_id or "no-rid"

        ctx = QueryContext(
            query=user_query,
            video_id=video_id,
            video_ids=video_ids,
            use_critic=self.use_critic,
            request_id=rid,
            video_catalog=self.video_catalog,
        )

        llm = StructuredLLMClient(self.model_client)
        state = QueryState.PARSE_INPUT

        try:
            while state != QueryState.SUBMIT:
                yield {
                    "type": "message",
                    "agent": "orchestrator",
                    "content": f"State: {state.name}",
                }

                state = await self._execute_state(state, ctx, llm)

            response = self._build_response(ctx)
            elapsed = time.time() - start
            ctx.add_tokens(llm.total_prompt_tokens, llm.total_completion_tokens)
            response["elapsed_seconds"] = elapsed
            response["token_usage"] = ctx.token_usage
            yield {"type": "final", "data": response}

        except Exception as e:
            logger.error(f"[{rid}] Pipeline error: {e}")
            yield {
                "type": "final",
                "data": {
                    "answer": f"Pipeline error: {e}",
                    "sources": [],
                    "token_usage": ctx.token_usage,
                    "elapsed_seconds": time.time() - start,
                },
            }
        finally:
            self._image_analyzer.cleanup()

    async def close(self) -> None:
        """Closes the underlying Neo4j provider connection."""
        if self.neo4j_provider:
            await self.neo4j_provider.close()

    async def _run_state_machine(self, ctx: QueryContext) -> Dict[str, Any]:
        """Drives the main state machine loop until completion."""
        llm = StructuredLLMClient(self.model_client)
        state = QueryState.PARSE_INPUT

        while state != QueryState.SUBMIT:
            state = await self._execute_state(state, ctx, llm)

        ctx.add_tokens(llm.total_prompt_tokens, llm.total_completion_tokens)
        return self._build_response(ctx)

    async def _execute_state(
        self, state: QueryState, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """Executes the logic for a single pipeline state and determine next state."""
        _log.bind(request_id=ctx.request_id, state=state.name, video_id=ctx.video_id, query=ctx.query).info(
            f"[{ctx.request_id}] {state.name}"
        )
        ctx.log_state(state)

        match state:
            case QueryState.PARSE_INPUT:
                return self._state_parse_input(ctx)
            case QueryState.PLAN:
                return await self._planner.run(ctx)
            case QueryState.VALIDATE_PLAN:
                return self._state_validate_plan(ctx)
            case QueryState.DISCOVER_VIDEOS:
                return await self._state_discover_videos(ctx)
            case QueryState.RETRIEVE:
                return await self._state_retrieve(ctx)
            case QueryState.CHECK_EVIDENCE:
                return self._state_check_evidence(ctx)
            case QueryState.EXPAND_CONTEXT:
                return await self._state_expand_context(ctx, llm)
            case QueryState.REPHRASE:
                return await self._state_rephrase(ctx, llm)
            case QueryState.ANALYZE_IMAGES:
                return await self._state_analyze_images(ctx)
            case QueryState.SYNTHESIZE:
                return await self._state_synthesize(ctx, llm)
            case QueryState.CRITIQUE:
                return await self._critic.run(ctx)
            case QueryState.REVISE:
                return await self._state_revise(ctx, llm)
            case QueryState.ERROR:
                return QueryState.SUBMIT
            case _:
                raise StateTransitionError(state, f"Unhandled state: {state}")

    def _state_parse_input(self, ctx: QueryContext) -> QueryState:
        """Determines video scope and ID constraints from input parameters."""
        if ctx.video_id:
            ctx.video_scope = "single"
            ctx.effective_video_ids = [ctx.video_id]
        elif ctx.video_ids:
            ctx.video_scope = "multi"
            ctx.effective_video_ids = ctx.video_ids
        else:
            ctx.video_scope = "cross"
            ctx.effective_video_ids = None

        _log.info(f"[{ctx.request_id}] PARSE_INPUT — scope={ctx.video_scope}")
        return QueryState.PLAN

    def _state_validate_plan(self, ctx: QueryContext) -> QueryState:
        """Performs programmatic validation and clamping of the retrieval plan."""
        plan = ctx.plan
        if not plan:
            ctx.answer = "No plan generated."
            return QueryState.ERROR

        if plan.get("strategy") not in VALID_STRATEGIES:
            plan["strategy"] = "SEARCH"

        plan["targets"] = [t for t in plan.get("targets", ["Chapter"]) if t in VALID_TARGETS]
        if not plan["targets"]:
            plan["targets"] = ["Chapter"]

        sqs = plan.get("sub_queries", [ctx.query])
        if len(sqs) < MIN_SUB_QUERIES:
            sqs = [ctx.query]
        elif len(sqs) > MAX_SUB_QUERIES:
            sqs = sqs[:MAX_SUB_QUERIES]
        plan["sub_queries"] = sqs

        plan["limit"] = max(1, min(plan.get("limit", 5), 20))

        if not plan.get("visual"):
            _VISUAL_KEYWORDS = [
                "color", "colour", "shape", "size", "appearance",
                "left", "right", "above", "below", "behind", "between",
                "how many", "count", "number of",
                "wearing", "holding", "carrying",
                "look like", "looks like", "appear", "appears",
                "position", "located", "placement",
            ]
            query_lower = ctx.query.lower()
            if any(kw in query_lower for kw in _VISUAL_KEYWORDS):
                plan["visual"] = True
                _log.info(
                    f"[{ctx.request_id}] VALIDATE_PLAN — "
                    "visual flag forced ON by keyword safety net"
                )

        ctx.plan = plan

        if ctx.video_scope == "cross" and ctx.effective_video_ids is None:
            return QueryState.DISCOVER_VIDEOS
        return QueryState.RETRIEVE

    async def _state_discover_videos(self, ctx: QueryContext) -> QueryState:
        """Identifies relevant video IDs for cross-video scope queries."""
        ranked = await self._discovery.discover(query=ctx.query, limit=8)

        if not ranked:
            logger.warning(f"[{ctx.request_id}] No videos discovered")
            ctx.answer = "No relevant videos found in the knowledge graph."
            ctx.sources = []
            return QueryState.SUBMIT

        _SUMMARY_KEYWORDS = {"summary", "recap", "review of weeks", "summary of weeks"}

        selected = []
        demoted = []
        for vid, score, title in ranked:
            is_summary = any(kw in title.lower() for kw in _SUMMARY_KEYWORDS)
            if is_summary:
                demoted.append((vid, score, title))
            else:
                selected.append(vid)

        MAX_RETRIEVAL_VIDEOS = 3
        if len(selected) > MAX_RETRIEVAL_VIDEOS:
            selected = selected[:MAX_RETRIEVAL_VIDEOS]
        elif len(selected) < MAX_RETRIEVAL_VIDEOS and demoted:
            for vid, _, _ in demoted:
                if len(selected) >= MAX_RETRIEVAL_VIDEOS:
                    break
                selected.append(vid)

        ctx.effective_video_ids = selected
        _log.info(
            f"[{ctx.request_id}] DISCOVER_VIDEOS — "
            f"discovered {len(ranked)} → selected {len(selected)} videos: {selected}"
        )
        if demoted:
            _log.info(
                f"[{ctx.request_id}] DISCOVER_VIDEOS — "
                f"demoted summary videos: {[(v, t) for v, _, t in demoted]}"
            )
        return QueryState.RETRIEVE

    async def _state_retrieve(self, ctx: QueryContext) -> QueryState:
        """Executes programmatic retrieval of video metadata and features."""
        plan = ctx.plan
        ctx.retrieve_attempts += 1

        if plan["strategy"] == "OVERVIEW":
            vid = (ctx.effective_video_ids or [""])[0]
            if vid:
                _log.info(f"[{ctx.request_id}] RETRIEVE — OVERVIEW for video={vid}")
                ctx.evidence = await self._retrieval.overview(
                    video_id=vid,
                    level=plan["targets"][0] if plan["targets"] else "ChapterGroup",
                    limit=plan.get("limit", 50),
                )
        else:
            _log.info(
                f"[{ctx.request_id}] RETRIEVE — SEARCH targets={plan['targets']} "
                f"sub_queries={len(plan.get('sub_queries', []))} "
                f"video_ids={ctx.effective_video_ids}"
            )
            ctx.evidence = await self._retrieval.search(
                sub_queries=plan["sub_queries"],
                targets=plan["targets"],
                video_ids=ctx.effective_video_ids,
                limit=plan.get("limit", 5),
            )

        _log.info(
            f"[{ctx.request_id}] RETRIEVE — retrieved {len(ctx.evidence)} evidence items"
        )
        for i, e in enumerate(ctx.evidence):
            ntype = e.get("node_type", "?")
            vid = e.get("video_id", "?")
            score = e.get("score", "")
            score_str = f" score={score:.4f}" if isinstance(score, float) else ""
            desc = e.get("summary") or e.get("transcript") or e.get("description") or ""
            if len(desc) > 120:
                desc = desc[:120] + "..."
            _log.info(f"[{ctx.request_id}] RETRIEVE — [{i+1}] {ntype} {vid}{score_str}: {desc}")

        if plan.get("visual") and ctx.evidence:
            _log.info(f"[{ctx.request_id}] RETRIEVE — searching keyframes (visual=True)")
            ctx.keyframes = await self._retrieval.search_keyframes(
                query=ctx.query,
                video_ids=ctx.effective_video_ids,
                limit=5,
            )
            _log.info(f"[{ctx.request_id}] RETRIEVE — found {len(ctx.keyframes)} keyframes")

        return QueryState.CHECK_EVIDENCE

    def _state_check_evidence(self, ctx: QueryContext) -> QueryState:
        """Assesses evidence sufficiency and routing to expansion or rephrasing."""
        has_evidence = len(ctx.evidence) > 0

        if has_evidence:
            _log.info(
                f"[{ctx.request_id}] CHECK_EVIDENCE — "
                f"{len(ctx.evidence)} results — routing to context expansion"
            )
            return QueryState.EXPAND_CONTEXT

        if ctx.retrieve_attempts < MAX_RETRIEVE_ATTEMPTS:
            _log.info(
                f"[{ctx.request_id}] CHECK_EVIDENCE — "
                "empty results — will rephrase and retry"
            )
            return QueryState.REPHRASE

        _log.info(
            f"[{ctx.request_id}] CHECK_EVIDENCE — "
            f"empty after {ctx.retrieve_attempts} attempts — synthesizing with no evidence"
        )
        return QueryState.SYNTHESIZE

    async def _state_expand_context(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """Determines through the LLM whether additional graph context is required."""
        user_msg = build_expansion_user_message(ctx.query, ctx.evidence)

        try:
            result = await llm.call_typed(
                system_prompt=EXPAND_CONTEXT_PROMPT,
                user_message=user_msg,
                response_type=ContextExpansionResult,
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] EXPAND_CONTEXT LLM error: {e}")
            return self._route_after_expansion(ctx)

        if not result.needs_expansion or not result.operations:
            _log.info(f"[{ctx.request_id}] EXPAND_CONTEXT — no expansion needed")
            return self._route_after_expansion(ctx)

        valid_targets = {
            "SiblingChapters", "ChapterGroup", "Chapter",
            "Event", "Transcript", "Keyframe", "Object",
        }
        for op in result.operations[:1]:
            if op.target not in valid_targets:
                logger.warning(f"[{ctx.request_id}] Invalid expansion target: {op.target}")
                continue
            if not op.node_ids:
                continue

            _log.info(
                f"[{ctx.request_id}] EXPAND_CONTEXT — "
                f"traversing {len(op.node_ids)} node(s) → {op.target} ({op.reason})"
            )
            try:
                if op.target == "SiblingChapters":
                    expanded = await self._retrieval.get_sibling_chapters(
                        chapter_ids=op.node_ids,
                        limit=20,
                    )
                else:
                    expanded = await self._retrieval.traverse(
                        node_ids=op.node_ids,
                        target=op.target,
                        limit=20,
                    )
                if expanded:
                    existing_ids = {e.get("node_id") for e in ctx.evidence}
                    new_items = [e for e in expanded if e.get("node_id") not in existing_ids]
                    ctx.evidence.extend(new_items)
                    _log.info(
                        f"[{ctx.request_id}] EXPAND_CONTEXT — "
                        f"+{len(new_items)} new results from {op.target}"
                    )
            except Exception as e:
                logger.warning(f"[{ctx.request_id}] Traversal error: {e}")

        return self._route_after_expansion(ctx)

    def _route_after_expansion(self, ctx: QueryContext) -> QueryState:
        """Determines route after expansion (ANALYZE_IMAGES vs SYNTHESIZE)."""
        if ctx.plan and ctx.plan.get("visual") and ctx.keyframes:
            return QueryState.ANALYZE_IMAGES
        return QueryState.SYNTHESIZE

    async def _state_rephrase(self, ctx: QueryContext, llm: StructuredLLMClient) -> QueryState:
        """Rephrases sub-queries for refined retrieval attempts."""
        original_sqs = ctx.plan.get("sub_queries", [])
        user_msg = build_rephrase_user_message(original_sqs)

        try:
            result = await llm.call_typed(
                system_prompt=REPHRASE_PROMPT,
                user_message=user_msg,
                response_type=RephraseResult,
            )
            ctx.plan["sub_queries"] = result.sub_queries[:MAX_SUB_QUERIES]
            _log.info(
                f"[{ctx.request_id}] REPHRASE — "
                f"rephrased into {len(result.sub_queries)} queries"
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Rephrase failed: {e}")

        return QueryState.RETRIEVE

    async def _state_analyze_images(self, ctx: QueryContext) -> QueryState:
        """Calls the Image Analysis executor for perceptual frame understanding."""
        if not ctx.keyframes:
            return QueryState.SYNTHESIZE

        _log.info(f"[{ctx.request_id}] ANALYZE_IMAGES — analyzing {len(ctx.keyframes)} keyframes")
        ctx.image_analyses = await self._image_analyzer.analyze_keyframes(
            keyframes=ctx.keyframes,
            query=ctx.query,
        )
        _log.info(
            f"[{ctx.request_id}] ANALYZE_IMAGES — "
            f"completed {len(ctx.image_analyses)} image analyses"
        )
        return QueryState.SYNTHESIZE

    async def _state_synthesize(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """Synthesizes the final answer with evidence and citations using the LLM."""
        evidence_str = format_evidence_compact(ctx.evidence)
        image_str = (
            json.dumps(ctx.image_analyses, indent=2, default=str)
            if ctx.image_analyses
            else None
        )

        user_msg = build_synthesize_user_message(
            query=ctx.query,
            evidence=evidence_str,
            image_analyses=image_str,
        )

        try:
            result = await llm.call_typed(
                system_prompt=SYNTHESIZE_PROMPT,
                user_message=user_msg,
                response_type=SynthesisResult,
            )
            ctx.answer = result.answer
            ctx.sources = [s.model_dump() for s in result.sources]
            ctx.answer, ctx.sources = dedup_sources(ctx.answer, ctx.sources)
            _log.info(
                f"[{ctx.request_id}] SYNTHESIZE — "
                f"answer generated: {len(result.answer)} chars, {len(ctx.sources)} citations"
            )
        except Exception as e:
            logger.error(f"[{ctx.request_id}] Synthesis failed: {e}")
            ctx.answer = "Failed to synthesize answer from evidence."
            ctx.sources = []

        if ctx.use_critic:
            return QueryState.CRITIQUE
        return QueryState.SUBMIT

    async def _state_revise(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """Revises the synthesized answer based on critical feedback."""
        ctx.revise_attempts += 1
        evidence_str = format_evidence_compact(ctx.evidence)
        feedback_str = json.dumps(ctx.critic_feedback, indent=2) if ctx.critic_feedback else ""

        user_msg = build_revise_user_message(
            query=ctx.query,
            evidence=evidence_str,
            previous_answer=ctx.answer or "",
            critic_feedback=feedback_str,
        )

        try:
            result = await llm.call_typed(
                system_prompt=SYNTHESIZE_PROMPT,
                user_message=user_msg,
                response_type=SynthesisResult,
            )
            ctx.answer = result.answer
            ctx.sources = [s.model_dump() for s in result.sources]
            ctx.answer, ctx.sources = dedup_sources(ctx.answer, ctx.sources)
            _log.info(
                f"[{ctx.request_id}] REVISE — "
                f"revised answer: {len(result.answer)} chars, {len(ctx.sources)} citations"
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Revision failed: {e}")

        return QueryState.SUBMIT

    def _build_response(self, ctx: QueryContext) -> Dict[str, Any]:
        """Constructs the standard result dictionary from the query context."""
        return {
            "answer": ctx.answer or "No answer generated.",
            "sources": ctx.sources,
        }


async def process_query(
    user_query: str,
    video_id: Optional[str] = None,
    video_ids: Optional[List[str]] = None,
    request_id: str = "",
    **orchestrator_kwargs,
) -> Dict[str, Any]:
    """Convenience functional entry point for the state machine pipeline.

    Args:
        user_query: Natural language question.
        video_id: Optional ID to restrict search to a single video.
        video_ids: Optional list of IDs to restrict search scope.
        request_id: Optional correlation ID.
        **orchestrator_kwargs: Dependencies passed to StateOrchestrator.

    Returns:
        Dict[str, Any]: Structured query result.
    """
    orch = StateOrchestrator(**orchestrator_kwargs)
    try:
        return await orch.query(
            user_query=user_query,
            video_id=video_id,
            video_ids=video_ids,
            request_id=request_id,
        )
    finally:
        await orch.close()
