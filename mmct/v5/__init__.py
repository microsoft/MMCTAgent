"""V5 Query Pipeline — State Machine-Driven Orchestration.

Replaces V4's AutoGen Swarm with a programmatic state machine.
LLMs are called only for intelligence tasks (planning, synthesis, critique).
All routing, tool selection, and parallelism are code-controlled.

No imports from mmct.v1, mmct.v2, or mmct.v4.
"""

from mmct.v5.orchestrator import V5Orchestrator
from mmct.v5.query.neo4j_provider import Neo4jQueryProvider, SearchResult
from mmct.v5.schemas import V5QueryResponse, CitationSource
from mmct.v5.state_machine import QueryState, QueryContext

__all__ = [
    "V5Orchestrator",
    "Neo4jQueryProvider",
    "SearchResult",
    "V5QueryResponse",
    "CitationSource",
    "QueryState",
    "QueryContext",
]
