"""Unified entrypoint for graph and state query pipelines."""

from enum import Enum
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional

from config.provider_config import get_query_pipeline_providers
from mmct.utils.error_handler import ConfigurationException
from mmct.video_pipeline.graph.orchestrator import GraphOrchestrator
from mmct.video_pipeline.state.orchestrator import StateOrchestrator


class QueryPipelineMode(str, Enum):
    """Available query pipeline implementations."""

    GRAPH = "graph"
    STATE = "state"


class VideoQueryPipeline:
    """Enum-selected wrapper over graph/state query orchestrators."""

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
            GraphOrchestrator if self.mode == QueryPipelineMode.GRAPH else StateOrchestrator
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
        return self._orchestrator.query_stream(
            user_query=user_query,
            video_id=video_id,
            video_ids=video_ids,
            request_id=request_id,
        )

    async def close(self) -> None:
        await self._orchestrator.close()
