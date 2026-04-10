"""Compatibility shim for the state pipeline Neo4j query provider.

The v4 (graph) and v5 (state) query pipelines share a single, unified Neo4j
implementation in `mmct.providers.custom_providers.neo4j_query_provider`.

This module exists only to preserve import paths:
`mmct.video_pipeline.graph_state.query.Neo4jQueryProvider`.
"""

from mmct.providers.custom_providers.neo4j_query_provider import Neo4jQueryProvider
from mmct.providers.base.graph_query_provider import SearchResult

__all__ = ["Neo4jQueryProvider", "SearchResult"]
