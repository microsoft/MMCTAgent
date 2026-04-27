"""Unified entrypoint for graph and state query pipelines."""

from enum import Enum
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional

from mmct.acl import AccessCheckCallback, UserIdentifierContext, user_identifier_scope
from mmct.utils.error_handler import ConfigurationException
from mmct.video_pipeline.graph_agent.orchestrator import GraphOrchestrator
from mmct.video_pipeline.graph_state.orchestrator import StateOrchestrator


class QueryPipelineMode(str, Enum):
    """Available query pipeline implementations."""

    GRAPH_AGENT = "graph_agent"
    GRAPH_STATE = "graph_state"


class VideoQueryPipeline:
    """Unified entry point for querying video content using MMCT pipelines.

    The VideoQueryPipeline acts as a high-level wrapper that orchestrates 
    video-based question answering using either an agentic swarm (`GRAPH_AGENT`) 
    or a deterministic state machine (`GRAPH_STATE`). It handles dependency 
    injection and provides both asynchronous and streaming query interfaces.

    Attributes:
        mode (QueryPipelineMode): The active orchestration strategy (Agent vs State).
    """

    def __init__(
        self,
        mode: Annotated[QueryPipelineMode, "Selects graph vs state query pipeline"],
        *,
        model_client: Annotated[Any, "AutoGen-compatible chat completion client"] = None,
        neo4j_provider: Annotated[Any, "Unified Neo4j query provider"] = None,
        storage_provider: Annotated[Any, "Optional blob storage provider"] = None,
        image_llm_provider: Annotated[Any, "Optional vision-capable LLM provider"] = None,
        use_critic: Annotated[bool, "Enable answer critique/revision pass"] = True,
        max_turns: Annotated[int, "Maximum graph swarm turns"] = 20,
        video_catalog: Annotated[Optional[str], "Optional planner catalog context"] = None,
        use_provider_defaults: Annotated[
            bool,
            "Hydrate missing dependencies from config.provider_config",
        ] = False,
        acl_callback: Annotated[
            Optional[AccessCheckCallback],
            "Per-deployment access-check callback. Required when ACL_ENABLED=true.",
        ] = None,
    ) -> None:
        """Initializes the VideoQueryPipeline with selected mode and providers.

        Args:
            mode: The pipeline execution mode (GRAPH_AGENT or GRAPH_STATE).
            model_client: Client for generating chat completions. Required if 
                `use_provider_defaults` is False.
            neo4j_provider: Provider for interacting with the Neo4j video graph.
                Required if `use_provider_defaults` is False.
            storage_provider: Provider for retrieving stored assets (keyframes, etc.).
            image_llm_provider: Provider for vision-language tasks (e.g., GPT-4V).
            use_critic: Whether to run a critic/revision cycle on the final answer.
            max_turns: Maximum conversation turns for agent-based reasoning.
            video_catalog: Pre-populated catalog of videos for the planner.
            use_provider_defaults: If True, automatically fetches providers from
                centralized configuration if not explicitly provided.
            acl_callback: Per-deployment access-check callback with signature
                ``async (video_ids: list[str], user_identifier_context: dict)
                -> AccessCheckResult``. Must be supplied when
                ACL_ENABLED=true; must NOT be supplied when ACL_ENABLED=false.

        Raises:
            ConfigurationException: If required dependencies are missing and
                cannot be hydrated from defaults, if ACL_ENABLED=true but no
                ``acl_callback`` is supplied, or if ``acl_callback`` is
                supplied while ACL_ENABLED=false.
        """
        self.mode = QueryPipelineMode(mode)

        from config.provider_config import get_settings
        self._acl_enabled = get_settings().acl_enabled
        if self._acl_enabled and acl_callback is None:
            raise ConfigurationException(
                "ACL_ENABLED=true but no acl_callback provided to VideoQueryPipeline"
            )
        if not self._acl_enabled and acl_callback is not None:
            raise ConfigurationException(
                "acl_callback was supplied to VideoQueryPipeline but ACL_ENABLED=false. "
                "Either set ACL_ENABLED=true or remove the acl_callback argument."
            )
        self._acl_callback = acl_callback

        if use_provider_defaults:
            from config.provider_config import get_query_pipeline_providers
            defaults = get_query_pipeline_providers()
            model_client = model_client or defaults.model_client
            neo4j_provider = neo4j_provider or defaults.neo4j_provider
            storage_provider = storage_provider or defaults.storage_provider
            image_llm_provider = image_llm_provider or defaults.image_llm_provider

        missing = [
            name
            for name, value in (
                ("model_client", model_client),
                ("neo4j_provider", neo4j_provider),
            )
            if value is None
        ]
        if missing:
            raise ConfigurationException(
                "Missing required query pipeline dependencies: "
                + ", ".join(missing)
                + ". Pass them explicitly or set `use_provider_defaults=True`."
            )

        orchestrator_cls = (
            GraphOrchestrator
            if self.mode == QueryPipelineMode.GRAPH_AGENT
            else StateOrchestrator
        )

        # Store providers for health checks
        self._neo4j_provider = neo4j_provider
        self._model_client = model_client
        self._image_llm_provider = image_llm_provider
        self._storage_provider = storage_provider

        # __init__ enforces (acl_enabled iff acl_callback is not None), so
        # passing acl_callback through directly accurately reflects whether
        # filtering is active.
        self._orchestrator = orchestrator_cls(
            model_client=model_client,
            neo4j_provider=neo4j_provider,
            storage_provider=storage_provider,
            image_llm_provider=image_llm_provider,
            use_critic=use_critic,
            max_turns=max_turns,
            video_catalog=video_catalog,
            acl_callback=acl_callback,
        )

    async def query(
        self,
        user_query: Annotated[str, "Natural language question about video content"],
        video_id: Annotated[Optional[str], "Single video scope"] = None,
        video_ids: Annotated[Optional[List[str]], "Multi-video scope"] = None,
        request_id: Annotated[str, "Optional request correlation ID"] = "",
        user_identifier_context: Annotated[
            Optional[UserIdentifierContext],
            "Per-request caller identity dict; required when ACL_ENABLED=true.",
        ] = None,
    ) -> Dict[str, Any]:
        """Executes a natural language query against the video knowledge graph.

        Args:
            user_query: The question to answer based on visual and textual content.
            video_id: Optional ID to restrict the search to a specific video.
            video_ids: Optional list of IDs to restrict the search to a subset
                of videos.
            request_id: Optional unique identifier for tracking the request.
            user_identifier_context: Per-request caller identity dict whose
                shape matches the ``acl_callback`` supplied at construction.
                Required when ACL_ENABLED=true; ignored otherwise.

        Returns:
            Dict[str, Any]: The structured response containing the answer,
                evidence, and metadata.

        Raises:
            ConfigurationException: When ACL_ENABLED=true and
                ``user_identifier_context`` is None.
        """
        self._require_user_ctx_when_acl_enabled(user_identifier_context)
        async with user_identifier_scope(user_identifier_context):
            return await self._orchestrator.query(
                user_query=user_query,
                video_id=video_id,
                video_ids=video_ids,
                request_id=request_id,
            )

    def query_stream(
        self,
        user_query: Annotated[str, "Natural language question about video content"],
        video_id: Annotated[Optional[str], "Single video scope"] = None,
        video_ids: Annotated[Optional[List[str]], "Multi-video scope"] = None,
        request_id: Annotated[str, "Optional request correlation ID"] = "",
        user_identifier_context: Annotated[
            Optional[UserIdentifierContext],
            "Per-request caller identity dict; required when ACL_ENABLED=true.",
        ] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Executes a query and returns an asynchronous generator for streaming updates.

        Args:
            user_query: The question to answer based on video content.
            video_id: Optional ID to restrict search.
            video_ids: Optional list of IDs to restrict search.
            request_id: Optional correlation ID.
            user_identifier_context: Per-request caller identity dict whose
                shape matches the ``acl_callback`` supplied at construction.
                Required when ACL_ENABLED=true; ignored otherwise.

        Returns:
            AsyncGenerator[Dict[str, Any], None]: A generator yielding status
                updates and the final query results.

        Raises:
            ConfigurationException: When ACL_ENABLED=true and
                ``user_identifier_context`` is None. Raised eagerly before
                the generator yields its first event.
        """
        # Eagerly fail-fast before constructing the generator, so callers see
        # the misconfiguration immediately rather than on first iteration.
        self._require_user_ctx_when_acl_enabled(user_identifier_context)

        async def _stream():
            async with user_identifier_scope(user_identifier_context):
                async for event in self._orchestrator.query_stream(
                    user_query=user_query,
                    video_id=video_id,
                    video_ids=video_ids,
                    request_id=request_id,
                ):
                    yield event

        return _stream()

    def _require_user_ctx_when_acl_enabled(
        self, user_identifier_context: Optional[UserIdentifierContext]
    ) -> None:
        if self._acl_enabled and user_identifier_context is None:
            raise ConfigurationException(
                "ACL_ENABLED=true but no user_identifier_context provided to "
                "VideoQueryPipeline.query()"
            )

    async def close(self) -> None:
        await self._orchestrator.close()

    async def check_health(self) -> Dict[str, Any]:
        """Verify connectivity of all underlying providers.

        Returns:
            Dict mapping provider name to its health status. Each entry
            contains at least ``{"status": "ok"|"error"|"not_configured"}``.
            Works with any provider implementation (Neo4j, custom, etc.)
            since it delegates to the provider's own ``check_health()``.
        """
        results: Dict[str, Any] = {}

        # Graph database provider
        if self._neo4j_provider is not None:
            results["graph_db"] = await self._neo4j_provider.check_health()
        else:
            results["graph_db"] = {"status": "not_configured"}

        # LLM provider (image_llm_provider exposes check_health via BaseLLMProvider)
        if self._image_llm_provider is not None and hasattr(self._image_llm_provider, "check_health"):
            results["llm"] = await self._image_llm_provider.check_health()
        else:
            results["llm"] = {"status": "not_configured"}

        # Storage provider (blob / file storage)
        if self._storage_provider is not None and hasattr(self._storage_provider, "check_health"):
            results["storage"] = await self._storage_provider.check_health()
        else:
            results["storage"] = {"status": "not_configured"}

        return results
