"""V4 Query Module for MMCTAgent.

This module provides the V4 query pipeline with:
- Neo4j graph backend with HNSW vector search
- Multi-granularity retrieval (ChapterGroup, Chapter, Event, Object, Keyframe)
- Cross-video search capabilities
- AutoGen Swarm-based multi-agent orchestration
- Unified graph traversal (up/down hierarchy)

Main exports:
- Neo4jQueryProvider: Query provider for Neo4j graph database
- V4Orchestrator: Multi-agent pipeline orchestrator using AutoGen Swarm
- V4 Agents: V4PlannerAgent, V4VideoAgent, V4ImageAgent, V4CriticAgent
- V4 Tools: GraphSearchTool, GraphTraversalTool, KeyframeRetrievalTool, VideoDiscoveryTool
- V4QueryResponse: Response schema with citations

Example usage:
    from mmct.v4 import V4Orchestrator, Neo4jQueryProvider, process_query_v4
    
    # Initialize Neo4j provider
    neo4j_provider = Neo4jQueryProvider(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
    )
    
    # Process a query
    result = await process_query_v4(
        query="How does the presenter install the component?",
        model_client=model_client,
        neo4j_provider=neo4j_provider,
        embedding_provider=embedding_provider,
        video_id="Dk1toyI7AJs",
    )
    
    print(result["answer"])  # Markdown with [1], [2] citations
    print(result["sources"])  # List of citation sources
"""

# Neo4j provider
from mmct.v4.query.neo4j_provider import (
    Neo4jQueryProvider,
    SearchResult,
    DEFAULT_EF_SEARCH,
)

# Tools
from mmct.v4.tools import (
    GraphSearchTool,
    GraphTraversalTool,
    KeyframeRetrievalTool,
    VideoDiscoveryTool,
)

# Agents
from mmct.v4.agents import (
    V4PlannerAgent,
    V4VideoAgent,
    V4ImageAgent,
    V4CriticAgent,
    V4_PLANNER_SYSTEM_PROMPT,
    V4_PLANNER_SYSTEM_PROMPT_WITH_CRITIC,
    V4_PLANNER_SYSTEM_PROMPT_WITHOUT_CRITIC,
    V4_VIDEO_AGENT_SYSTEM_PROMPT,
    V4_IMAGE_AGENT_SYSTEM_PROMPT,
    V4_CRITIC_SYSTEM_PROMPT,
)

# Schemas
from mmct.v4.schemas import (
    V4QueryResponse,
    CitationSource,
)

# Orchestrator
from mmct.v4.orchestrator import (
    V4Orchestrator,
    V4OrchestratorConfig,
    process_query_v4,
)

__all__ = [
    # Neo4j provider
    "Neo4jQueryProvider",
    "SearchResult",
    "DEFAULT_EF_SEARCH",
    # Tools
    "GraphSearchTool",
    "GraphTraversalTool",
    "KeyframeRetrievalTool",
    "VideoDiscoveryTool",
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
    # Schemas
    "V4QueryResponse",
    "CitationSource",
    # Orchestrator
    "V4Orchestrator",
    "V4OrchestratorConfig",
    "process_query_v4",
]
