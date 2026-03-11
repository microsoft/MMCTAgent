"""V5 Orchestrator — State machine-driven query pipeline.

Replaces AutoGen Swarm with a programmatic state machine.
Each state is a method; transitions are explicit match/case.
LLM calls only where intelligence is needed.
"""

import json
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from mmct.v5.state_machine import (
    QueryContext,
    QueryState,
    StateTransitionError,
    MAX_RETRIEVE_ATTEMPTS,
    MAX_REVISE_ATTEMPTS,
    MAX_SUB_QUERIES,
    MIN_SUB_QUERIES,
    VALID_TARGETS,
    VALID_STRATEGIES,
)
from mmct.v5.llm.client import StructuredLLMClient
from mmct.v5.prompts.planner import (
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
from mmct.v5.prompts.critic import (
    CRITIQUE_PROMPT,
    CritiqueResult,
    build_critique_user_message,
)
from mmct.v5.prompts.context_expansion import (
    EXPAND_CONTEXT_PROMPT,
    ContextExpansionResult,
    build_expansion_user_message,
)
from mmct.v5.executors.retrieval import RetrievalExecutor
from mmct.v5.executors.video_discovery import VideoDiscoveryExecutor
from mmct.v5.executors.image_analysis import ImageAnalysisExecutor
from mmct.v5.query.neo4j_provider import Neo4jQueryProvider


class _Colors:
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _print_state(state: QueryState, rid: str, detail: str = "") -> None:
    detail_str = f" — {detail}" if detail else ""
    print(
        f"{_Colors.GRAY}[{_ts()}]{_Colors.RESET} [{rid}] "
        f"{_Colors.MAGENTA}[state]{_Colors.RESET} "
        f"{_Colors.BOLD}{state.name}{_Colors.RESET}{detail_str}",
        flush=True,
    )


class V5Orchestrator:
    """State machine-driven query orchestrator.

    Drives the pipeline through deterministic states, calling LLMs
    only for intelligence tasks and executing tools programmatically.
    """

    def __init__(
        self,
        model_client,
        neo4j_provider: Neo4jQueryProvider,
        embedding_provider,
        image_embedding_provider=None,
        storage_provider=None,
        image_llm_provider=None,
        use_critic: bool = True,
        video_catalog: Optional[str] = None,
    ):
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.storage_provider = storage_provider
        self.image_llm_provider = image_llm_provider
        self.use_critic = use_critic
        self.video_catalog = video_catalog

        self._retrieval = RetrievalExecutor(
            neo4j_provider, embedding_provider, image_embedding_provider
        )
        self._discovery = VideoDiscoveryExecutor(neo4j_provider, embedding_provider)
        self._image_analyzer = ImageAnalysisExecutor(
            image_llm_provider=image_llm_provider,
            storage_provider=storage_provider,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Process a query through the state machine pipeline."""
        start = time.time()
        rid = request_id or "no-rid"

        print(f"\n{'=' * 60}", flush=True)
        print(f"[{rid}] {_Colors.CYAN}V5 Query Pipeline{_Colors.RESET}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"{_Colors.GRAY}[{_ts()}]{_Colors.RESET} [{rid}] Query: {user_query}", flush=True)

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

            print(f"\n{'=' * 60}", flush=True)
            print(f"[{rid}] {_Colors.GREEN}Query completed in {elapsed:.2f}s{_Colors.RESET}", flush=True)
            print(f"{'=' * 60}\n", flush=True)
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
        """Process a query with streaming state transition events."""
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

            # Final submit
            response = self._build_response(ctx)
            elapsed = time.time() - start
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

    # ------------------------------------------------------------------
    # State Machine Loop
    # ------------------------------------------------------------------

    async def _run_state_machine(self, ctx: QueryContext) -> Dict[str, Any]:
        llm = StructuredLLMClient(self.model_client)
        state = QueryState.PARSE_INPUT

        while state != QueryState.SUBMIT:
            state = await self._execute_state(state, ctx, llm)

        ctx.add_tokens(llm.total_prompt_tokens, llm.total_completion_tokens)
        return self._build_response(ctx)

    async def _execute_state(
        self, state: QueryState, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """Execute a single state and return the next state."""
        _print_state(state, ctx.request_id)
        ctx.log_state(state)

        match state:
            case QueryState.PARSE_INPUT:
                return self._state_parse_input(ctx)

            case QueryState.PLAN:
                return await self._state_plan(ctx, llm)

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
                return await self._state_critique(ctx, llm)

            case QueryState.REVISE:
                return await self._state_revise(ctx, llm)

            case QueryState.ERROR:
                return QueryState.SUBMIT

            case _:
                raise StateTransitionError(state, f"Unhandled state: {state}")

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _state_parse_input(self, ctx: QueryContext) -> QueryState:
        """Determine video scope from input params. No LLM."""
        if ctx.video_id:
            ctx.video_scope = "single"
            ctx.effective_video_ids = [ctx.video_id]
        elif ctx.video_ids:
            ctx.video_scope = "multi"
            ctx.effective_video_ids = ctx.video_ids
        else:
            ctx.video_scope = "cross"
            ctx.effective_video_ids = None

        _print_state(QueryState.PARSE_INPUT, ctx.request_id, f"scope={ctx.video_scope}")
        return QueryState.PLAN

    async def _state_plan(self, ctx: QueryContext, llm: StructuredLLMClient) -> QueryState:
        """LLM decomposes query into structured plan."""
        user_msg = build_plan_user_message(
            query=ctx.query,
            video_scope=ctx.video_scope,
            video_ids=ctx.effective_video_ids,
            video_catalog=ctx.video_catalog,
        )

        try:
            plan = await llm.call_typed(
                system_prompt=PLAN_QUERY_PROMPT,
                user_message=user_msg,
                response_type=PlanResult,
            )
            ctx.plan = plan.model_dump()
            _print_state(
                QueryState.PLAN,
                ctx.request_id,
                f"strategy={plan.strategy} targets={plan.targets} "
                f"sub_queries={len(plan.sub_queries)} visual={plan.visual}",
            )
            return QueryState.VALIDATE_PLAN

        except Exception as e:
            logger.error(f"[{ctx.request_id}] PLAN failed: {e}")
            ctx.answer = f"Failed to plan query: {e}"
            return QueryState.ERROR

    def _state_validate_plan(self, ctx: QueryContext) -> QueryState:
        """Programmatic validation and clamping of the plan. No LLM."""
        plan = ctx.plan
        if not plan:
            ctx.answer = "No plan generated."
            return QueryState.ERROR

        # Validate strategy
        if plan.get("strategy") not in VALID_STRATEGIES:
            plan["strategy"] = "SEARCH"

        # Validate and clamp targets
        plan["targets"] = [t for t in plan.get("targets", ["Chapter"]) if t in VALID_TARGETS]
        if not plan["targets"]:
            plan["targets"] = ["Chapter"]

        # Clamp sub-queries
        sqs = plan.get("sub_queries", [ctx.query])
        if len(sqs) < MIN_SUB_QUERIES:
            sqs = [ctx.query]
        elif len(sqs) > MAX_SUB_QUERIES:
            sqs = sqs[:MAX_SUB_QUERIES]
        plan["sub_queries"] = sqs

        # Clamp limit
        plan["limit"] = max(1, min(plan.get("limit", 5), 20))

        ctx.plan = plan

        # Decide next state
        if ctx.video_scope == "cross" and ctx.effective_video_ids is None:
            return QueryState.DISCOVER_VIDEOS
        return QueryState.RETRIEVE

    async def _state_discover_videos(self, ctx: QueryContext) -> QueryState:
        """Discover relevant videos for cross-video scope. No LLM."""
        discovered = await self._discovery.discover(query=ctx.query, limit=5)

        if not discovered:
            logger.warning(f"[{ctx.request_id}] No videos discovered")
            ctx.answer = "No relevant videos found in the knowledge graph."
            ctx.sources = []
            return QueryState.SUBMIT

        ctx.effective_video_ids = discovered
        _print_state(
            QueryState.DISCOVER_VIDEOS,
            ctx.request_id,
            f"found {len(discovered)} videos: {discovered}",
        )
        return QueryState.RETRIEVE

    async def _state_retrieve(self, ctx: QueryContext) -> QueryState:
        """Execute retrieval based on validated plan. No LLM."""
        plan = ctx.plan
        ctx.retrieve_attempts += 1

        if plan["strategy"] == "OVERVIEW":
            # Overview: fetch from first video
            vid = (ctx.effective_video_ids or [""])[0]
            if vid:
                ctx.evidence = await self._retrieval.overview(
                    video_id=vid,
                    level=plan["targets"][0] if plan["targets"] else "ChapterGroup",
                    limit=plan.get("limit", 50),
                )
        else:
            # SEARCH: parallel sub-query execution
            ctx.evidence = await self._retrieval.search(
                sub_queries=plan["sub_queries"],
                targets=plan["targets"],
                video_ids=ctx.effective_video_ids,
                limit=plan.get("limit", 5),
            )

        # Also search keyframes if visual
        if plan.get("visual") and ctx.evidence:
            ctx.keyframes = await self._retrieval.search_keyframes(
                query=ctx.query,
                video_ids=ctx.effective_video_ids,
                limit=5,
            )

        return QueryState.CHECK_EVIDENCE

    def _state_check_evidence(self, ctx: QueryContext) -> QueryState:
        """Check if evidence is sufficient. No LLM."""
        has_evidence = len(ctx.evidence) > 0

        if has_evidence:
            _print_state(
                QueryState.CHECK_EVIDENCE,
                ctx.request_id,
                f"{len(ctx.evidence)} results — routing to context expansion",
            )
            return QueryState.EXPAND_CONTEXT

        if ctx.retrieve_attempts < MAX_RETRIEVE_ATTEMPTS:
            _print_state(
                QueryState.CHECK_EVIDENCE,
                ctx.request_id,
                "empty results — will rephrase and retry",
            )
            return QueryState.REPHRASE

        _print_state(
            QueryState.CHECK_EVIDENCE,
            ctx.request_id,
            f"empty after {ctx.retrieve_attempts} attempts — synthesizing with no evidence",
        )
        return QueryState.SYNTHESIZE

    async def _state_expand_context(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """LLM decides whether to expand context via graph traversals."""
        user_msg = build_expansion_user_message(ctx.query, ctx.evidence)

        try:
            result = await llm.call_typed(
                system_prompt=EXPAND_CONTEXT_PROMPT,
                user_message=user_msg,
                response_type=ContextExpansionResult,
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] EXPAND_CONTEXT LLM error: {e} — skipping expansion")
            return self._route_after_expansion(ctx)

        if not result.needs_expansion or not result.operations:
            _print_state(
                QueryState.EXPAND_CONTEXT,
                ctx.request_id,
                "no expansion needed",
            )
            return self._route_after_expansion(ctx)

        # Execute each traversal operation
        valid_targets = {"ChapterGroup", "Chapter", "Event", "Transcript", "Keyframe", "Object"}
        for op in result.operations:
            if op.target not in valid_targets:
                logger.warning(f"[{ctx.request_id}] Invalid expansion target: {op.target}")
                continue
            if not op.node_ids:
                continue

            _print_state(
                QueryState.EXPAND_CONTEXT,
                ctx.request_id,
                f"traversing {len(op.node_ids)} node(s) → {op.target} ({op.reason})",
            )
            try:
                expanded = await self._retrieval.traverse(
                    node_ids=op.node_ids,
                    target=op.target,
                    limit=20,
                )
                if expanded:
                    # Deduplicate against existing evidence
                    existing_ids = {e.get("node_id") for e in ctx.evidence}
                    new_items = [e for e in expanded if e.get("node_id") not in existing_ids]
                    ctx.evidence.extend(new_items)
                    _print_state(
                        QueryState.EXPAND_CONTEXT,
                        ctx.request_id,
                        f"+{len(new_items)} new results from {op.target} traversal",
                    )
            except Exception as e:
                logger.warning(f"[{ctx.request_id}] Traversal error: {e}")

        return self._route_after_expansion(ctx)

    def _route_after_expansion(self, ctx: QueryContext) -> QueryState:
        """Route to ANALYZE_IMAGES or SYNTHESIZE after expansion."""
        if ctx.plan and ctx.plan.get("visual") and ctx.keyframes:
            return QueryState.ANALYZE_IMAGES
        return QueryState.SYNTHESIZE

    async def _state_rephrase(self, ctx: QueryContext, llm: StructuredLLMClient) -> QueryState:
        """LLM rephrases sub-queries for retry."""
        original_sqs = ctx.plan.get("sub_queries", [])
        user_msg = build_rephrase_user_message(original_sqs)

        try:
            result = await llm.call_typed(
                system_prompt=REPHRASE_PROMPT,
                user_message=user_msg,
                response_type=RephraseResult,
            )
            ctx.plan["sub_queries"] = result.sub_queries[:MAX_SUB_QUERIES]
            _print_state(
                QueryState.REPHRASE,
                ctx.request_id,
                f"rephrased to {len(result.sub_queries)} sub-queries",
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Rephrase failed: {e}")

        return QueryState.RETRIEVE

    async def _state_analyze_images(self, ctx: QueryContext) -> QueryState:
        """Analyze keyframe images. LLM (ViT) per keyframe."""
        if not ctx.keyframes:
            return QueryState.SYNTHESIZE

        ctx.image_analyses = await self._image_analyzer.analyze_keyframes(
            keyframes=ctx.keyframes,
            query=ctx.query,
        )
        return QueryState.SYNTHESIZE

    async def _state_synthesize(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """LLM writes answer with citations from evidence."""
        evidence_str = json.dumps(ctx.evidence, indent=2, default=str) if ctx.evidence else "No evidence retrieved."
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
            _print_state(
                QueryState.SYNTHESIZE,
                ctx.request_id,
                f"answer={len(result.answer)} chars, {len(result.sources)} citations",
            )
        except Exception as e:
            logger.error(f"[{ctx.request_id}] Synthesis failed: {e}")
            ctx.answer = "Failed to synthesize answer from evidence."
            ctx.sources = []

        if ctx.use_critic:
            return QueryState.CRITIQUE
        return QueryState.SUBMIT

    async def _state_critique(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """LLM evaluates answer quality."""
        evidence_str = json.dumps(ctx.evidence, indent=2, default=str) if ctx.evidence else ""
        user_msg = build_critique_user_message(
            query=ctx.query,
            answer=ctx.answer or "",
            evidence=evidence_str,
        )

        try:
            result = await llm.call_typed(
                system_prompt=CRITIQUE_PROMPT,
                user_message=user_msg,
                response_type=CritiqueResult,
            )
            ctx.critic_feedback = result.model_dump()
            _print_state(
                QueryState.CRITIQUE,
                ctx.request_id,
                f"verdict={result.verdict} — {result.feedback_summary}",
            )

            if result.verdict.upper() == "YES":
                return QueryState.SUBMIT

            if ctx.revise_attempts < MAX_REVISE_ATTEMPTS:
                return QueryState.REVISE

            _print_state(
                QueryState.CRITIQUE,
                ctx.request_id,
                "max revisions reached — submitting anyway",
            )
            return QueryState.SUBMIT

        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Critique failed: {e} — submitting anyway")
            return QueryState.SUBMIT

    async def _state_revise(
        self, ctx: QueryContext, llm: StructuredLLMClient
    ) -> QueryState:
        """LLM revises answer based on critic feedback."""
        ctx.revise_attempts += 1
        evidence_str = json.dumps(ctx.evidence, indent=2, default=str) if ctx.evidence else ""
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
            _print_state(
                QueryState.REVISE,
                ctx.request_id,
                f"revised answer={len(result.answer)} chars",
            )
        except Exception as e:
            logger.warning(f"[{ctx.request_id}] Revision failed: {e}")

        return QueryState.SUBMIT

    # ------------------------------------------------------------------
    # Response Builder
    # ------------------------------------------------------------------

    def _build_response(self, ctx: QueryContext) -> Dict[str, Any]:
        return {
            "answer": ctx.answer or "No answer generated.",
            "sources": ctx.sources,
        }

    async def close(self):
        if self.neo4j_provider:
            await self.neo4j_provider.close()
