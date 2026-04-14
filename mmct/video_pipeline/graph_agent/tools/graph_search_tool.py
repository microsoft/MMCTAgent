"""Graph Search Tool - Multi-granularity vector search for Neo4j.

Wraps Neo4jQueryProvider to provide multi-granularity search
as a callable tool for AutoGen agents.

Uses the node type registry for dynamic target validation and result formatting.
"""

from typing import Annotated, Optional, List, Tuple
import asyncio
import json
from loguru import logger

from mmct.video_pipeline.core.graph import node_registry
from mmct.video_pipeline.utils import OutputFormatterMixin

_log = logger.bind(component="Tool:search_graph")


class GraphSearchTool(OutputFormatterMixin):
    """Tool for multi-granularity vector search on Neo4j graph.
    
    Provides search_graph function that can search across multiple
    granularity levels in parallel. Valid targets are dynamically
    determined from the node type registry.
    
    Transcript vs Chapter:
    - Transcript: Raw verbal content (speech only). Use for quote search, spoken content lookup.
    - Chapter: Multimodal summary (visual + verbal). Use when visual context is needed.
    Avoid searching both for same query to prevent redundant results.
    """
    
    def __init__(self, neo4j_provider):
        """Initialize the tool.
        
        Args:
            neo4j_provider: Neo4jQueryProvider instance.
        """
        self.neo4j_provider = neo4j_provider
        self._embedding_provider = None

    def _get_embedding_provider(self):
        """Lazy-load the text embedding provider."""
        if self._embedding_provider is None:
            from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
            self._embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        return self._embedding_provider
    
    async def search_graph(
        self,
        query: Annotated[str, "Natural language search query"],
        targets: Annotated[List[str], "Granularity levels to search (use node_registry.names() for valid options)"],
        video_ids: Optional[Annotated[List[str], "Video IDs to filter by. Omit for cross-video search."]] = None,
        time_start: Optional[Annotated[float, "Start time in seconds for temporal filtering"]] = None,
        time_end: Optional[Annotated[float, "End time in seconds for temporal filtering"]] = None,
        limit: Annotated[int, "Maximum results per granularity level"] = 5,
        sort_by_time: Annotated[bool, "If True, sort results chronologically. Use for temporal queries."] = False,
    ) -> str:
        """Search the knowledge graph at specified granularity levels.
        
        Executes parallel vector searches at each specified granularity level
        and returns combined results. Supports optional temporal pre-filtering.
        
        Note: Transcript contains verbal-only content, Chapter contains multimodal summary.
        Choose wisely based on query type to avoid redundant results.
        
        Args:
            query: Natural language search query.
            targets: List of granularity levels to search.
            video_ids: Optional list of video IDs to filter results.
            time_start: Optional start time in seconds for temporal filtering.
            time_end: Optional end time in seconds for temporal filtering.
            limit: Maximum results per granularity level.
            sort_by_time: If True, sort results by time instead of relevance score.
            
        Returns:
            JSON string with search results grouped by granularity.
        """
        # Build time_range tuple if either time param is provided
        time_range: Optional[Tuple[float, float]] = None
        if time_start is not None or time_end is not None:
            time_range = (time_start or 0.0, time_end or float('inf'))
        
        time_str = f" time_range={time_range}" if time_range else ""
        sort_str = " sort=time" if sort_by_time else ""
        _log.info(f"query='{query}' targets={targets} video_ids={video_ids}{time_str}{sort_str}")
        try:
            # Validate targets using registry
            valid_targets = set(node_registry.names())
            targets_set = set(targets)
            invalid = targets_set - valid_targets
            if invalid:
                return json.dumps({
                    "error": f"Invalid targets: {invalid}. Valid: {valid_targets}"
                })
            
            # Get query embedding
            query_embedding = await self._get_embedding_provider().embedding(query)
            
            # Execute parallel searches (hybrid: vector + keyword)
            results = await self.neo4j_provider.search_multiple_granularities(
                query_embedding=query_embedding,
                targets=list(targets_set),
                video_ids=video_ids,
                time_range=time_range,
                limit_per_type=limit,
                sort_by_time=sort_by_time,
                query_text=query,
            )
            
            # Format results
            formatted = self._format_results(results)
            total_results = sum(len(v) for v in formatted.values())
            _log.info(f"Found {total_results} results across {len(formatted)} levels")
            return self.format_output(formatted)
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_results(self, results: dict) -> dict:
        """Format search results for readability using node type registry.
        
        Args:
            results: Raw results from Neo4jQueryProvider (Dict[str, List[SearchResult]]).
            
        Returns:
            Formatted results dictionary.
        """
        formatted = {}
        
        for granularity, items in results.items():
            if not items:
                continue
            
            # Get node type from registry for formatting
            node_type = node_registry.get(granularity)
                
            formatted[granularity] = []
            for item in items:
                # Handle both SearchResult objects and dicts
                if hasattr(item, 'to_dict'):
                    item_dict = item.to_dict()
                else:
                    item_dict = item
                
                entry = {
                    "id": item_dict.get("node_id") or item_dict.get("id"),
                    "video_id": item_dict.get("video_id"),
                    "score": round(item_dict.get("score", 0), 4),
                }
                
                # Get properties if available
                props = item_dict.get("properties", item_dict)
                
                # Use registry for type-specific formatting
                if node_type:
                    entry.update(node_type.format_search_result(props))
                
                formatted[granularity].append(entry)
        
        return formatted
