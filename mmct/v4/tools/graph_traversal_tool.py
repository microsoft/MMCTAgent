"""Graph Traversal Tool - Unified relationship traversal (up/down hierarchy).

Wraps Neo4jQueryProvider to traverse relationships between
nodes in the knowledge graph in any direction.

Uses the node type registry for dynamic target validation and result formatting.
"""

from typing import Annotated, List, Optional, Tuple
from datetime import datetime
import json
from loguru import logger

from mmct.graph import node_registry
from mmct.v4.utils import OutputFormatterMixin


# ANSI colors for tool output
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _tool_log(tool_name: str, message: str) -> None:
    """Print tool log with timestamp and colors."""
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{_GRAY}[{now}]{_RESET} {_CYAN}[Tool:{tool_name}]{_RESET} {message}", flush=True)


class GraphTraversalTool(OutputFormatterMixin):
    """Unified tool for traversing relationships in the Neo4j knowledge graph.
    
    Provides a single `traverse_graph` function that can navigate both
    up and down the hierarchy with optional filters.
    
    Valid targets are dynamically determined from the node type registry.
    
    Transcript vs Chapter:
    - Transcript: Raw verbal content (speech only). Use for quote search, spoken content lookup.
    - Chapter: Multimodal summary (visual + verbal). Use when visual context is needed.
    """
    
    def __init__(self, neo4j_provider):
        """Initialize the tool.
        
        Args:
            neo4j_provider: Neo4jQueryProvider instance.
        """
        self.neo4j_provider = neo4j_provider
    
    async def traverse_graph(
        self,
        node_ids: Annotated[List[str], "Starting node IDs to traverse from (e.g., ['chapter_Dk1toyI7AJs_001'])"],
        target: Annotated[str, "Target node type (use node_registry.names() for valid options)"],
        video_id: Annotated[Optional[str], "Filter results by video_id"] = None,
        time_start: Annotated[Optional[float], "Filter by start time (seconds)"] = None,
        time_end: Annotated[Optional[float], "Filter by end time (seconds)"] = None,
        limit: Annotated[int, "Maximum results to return"] = 20,
    ) -> str:
        """Traverse graph relationships up or down the hierarchy.
        
        Automatically detects source node type from node_ids and determines
        traversal direction based on source → target relationship.
        
        Examples:
        - Chapter → Event (down): Get events in specific chapters
        - Event → Chapter (up): Get parent chapter for events
        - Chapter → ChapterGroup (up): Get topic group for chapters
        - Event → Object (down): Get objects/entities in events
        
        Args:
            node_ids: List of source node IDs (type auto-detected from prefix).
            target: Target node type to traverse to.
            video_id: Optional filter by video_id.
            time_start: Optional filter - start of time range (seconds).
            time_end: Optional filter - end of time range (seconds).
            limit: Maximum number of results.
            
        Returns:
            JSON string with traversal results.
        """
        nodes_preview = node_ids[:3]
        _tool_log("traverse_graph", f"nodes={node_ids} → {target} video_id={video_id}")
        try:
            # Validate target using registry
            valid_targets = set(node_registry.names())
            if target not in valid_targets:
                return json.dumps({
                    "error": f"Invalid target: {target}. Valid targets: {valid_targets}"
                })
            
            # Build time_range tuple if provided
            time_range = None
            if time_start is not None and time_end is not None:
                time_range = (time_start, time_end)
            elif time_start is not None:
                time_range = (time_start, float('inf'))
            elif time_end is not None:
                time_range = (0.0, time_end)
            
            # Call provider's traverse method
            results = await self.neo4j_provider.traverse_relationships(
                source_ids=node_ids,
                target_type=target,
                video_id=video_id,
                time_range=time_range,
                limit=limit,
            )
            
            # Format results based on target type
            formatted = self._format_results(results, target)
            _tool_log("traverse_graph", f"{_YELLOW}Found {len(results)} {target} nodes{_RESET}")
            return self.format_output(formatted)
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_results(self, results: list, target_type: str) -> dict:
        """Format traversal results using node type registry.
        
        Args:
            results: List of SearchResult objects from provider.
            target_type: The target node type.
            
        Returns:
            Formatted results dictionary.
        """
        formatted = {
            "target_type": target_type,
            "results": [],
            "total": len(results),
        }
        
        # Get node type from registry for formatting
        node_type = node_registry.get(target_type)
        
        for item in results:
            # Handle both SearchResult objects and dicts
            if hasattr(item, 'to_dict'):
                item_dict = item.to_dict()
            else:
                item_dict = item
            
            # Get properties
            props = item_dict.get("properties", item_dict)
            
            entry = {
                "node_id": item_dict.get("node_id") or props.get("node_id"),
                "video_id": props.get("video_id"),
            }
            
            # Use registry for type-specific formatting
            if node_type:
                entry.update(node_type.format_search_result(props))
            
            formatted["results"].append(entry)
        
        return formatted
