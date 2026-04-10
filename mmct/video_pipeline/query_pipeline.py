"""Unified entrypoint for graph and state query pipelines."""

from enum import Enum
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional

from config.provider_config import get_query_pipeline_providers
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
        embedding_provider: Annotated[Any, "Text embedding provider"] = None,
        image_embedding_provider: Annotated[Any, "Optional image embedding provider"] = None,
        storage_provider: Annotated[Any, "Optional blob storage provider"] = None,
        image_llm_provider: Annotated[Any, "Optional vision-capable LLM provider"] = None,
        use_critic: Annotated[bool, "Enable answer critique/revision pass"] = True,
        max_turns: Annotated[int, "Maximum graph swarm turns"] = 20,
        video_catalog: Annotated[Optional[str], "Optional planner catalog context"] = None,
        use_provider_defaults: Annotated[
            bool,
            "Hydrate missing dependencies from config.provider_config",
        ] = False,
    ) -> None:
        """Initializes the VideoQueryPipeline with selected mode and providers.

        Args:
            mode: The pipeline execution mode (GRAPH_AGENT or GRAPH_STATE).
            model_client: Client for generating chat completions. Required if 
                `use_provider_defaults` is False.
            neo4j_provider: Provider for interacting with the Neo4j video graph.
                Required if `use_provider_defaults` is False.
            embedding_provider: Provider for generating text embeddings.
                Required if `use_provider_defaults` is False.
            image_embedding_provider: Provider for image-based semantic search.
            storage_provider: Provider for retrieving stored assets (keyframes, etc.).
            image_llm_provider: Provider for vision-language tasks (e.g., GPT-4V).
            use_critic: Whether to run a critic/revision cycle on the final answer.
            max_turns: Maximum conversation turns for agent-based reasoning.
            video_catalog: Pre-populated catalog of videos for the planner.
            use_provider_defaults: If True, automatically fetches providers from
                centralized configuration if not explicitly provided.

        Raises:
            ConfigurationException: If required dependencies are missing and
                cannot be hydrated from defaults.
        """
        self.mode = QueryPipelineMode(mode)

        if use_provider_defaults:
            defaults = get_query_pipeline_providers()
            model_client = model_client or defaults.model_client
            neo4j_provider = neo4j_provider or defaults.neo4j_provider
            embedding_provider = embedding_provider or defaults.embedding_provider
            image_embedding_provider = (
                image_embedding_provider or defaults.image_embedding_provider
            )
            storage_provider = storage_provider or defaults.storage_provider
            image_llm_provider = image_llm_provider or defaults.image_llm_provider

        missing = [
            name
            for name, value in (
                ("model_client", model_client),
                ("neo4j_provider", neo4j_provider),
                ("embedding_provider", embedding_provider),
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
        self._orchestrator = orchestrator_cls(
            model_client=model_client,
            neo4j_provider=neo4j_provider,
            embedding_provider=embedding_provider,
            image_embedding_provider=image_embedding_provider,
            storage_provider=storage_provider,
            image_llm_provider=image_llm_provider,
            use_critic=use_critic,
            max_turns=max_turns,
            video_catalog=video_catalog,
        )

    async def query(
        self,
        user_query: Annotated[str, "Natural language question about video content"],
        video_id: Annotated[Optional[str], "Single video scope"] = None,
        video_ids: Annotated[Optional[List[str]], "Multi-video scope"] = None,
        request_id: Annotated[str, "Optional request correlation ID"] = "",
    ) -> Dict[str, Any]:
        """Executes a natural language query against the video knowledge graph.

        Args:
            user_query: The question to answer based on visual and textual content.
            video_id: Optional ID to restrict the search to a specific video.
            video_ids: Optional list of IDs to restrict the search to a subset
                of videos.
            request_id: Optional unique identifier for tracking the request.

        Returns:
            Dict[str, Any]: The structured response containing the answer,
                evidence, and metadata.
        """
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Executes a query and returns an asynchronous generator for streaming updates.

        Args:
            user_query: The question to answer based on video content.
            video_id: Optional ID to restrict search.
            video_ids: Optional list of IDs to restrict search.
            request_id: Optional correlation ID.

        Returns:
            AsyncGenerator[Dict[str, Any], None]: A generator yielding status
                updates and the final query results.
        """
        return self._orchestrator.query_stream(
            user_query=user_query,
            video_id=video_id,
            video_ids=video_ids,
            request_id=request_id,
        )

    async def close(self) -> None:
        await self._orchestrator.close()
