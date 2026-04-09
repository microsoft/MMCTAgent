"""Video Overview Tool - Fetch all nodes for a video without vector search.

Use this tool when the query requires understanding of the whole video,
not just semantically similar segments. This is faster and more complete
for overview-type queries.
"""

from typing import Annotated, Optional, List
from datetime import datetime
import json
from loguru import logger

from mmct.graph import node_registry
from mmct.video_pipeline.utils import OutputFormatterMixin


# ANSI colors for tool output
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _tool_log(tool_name: str, message: str) -> None:
    """Print tool log with timestamp and colors."""
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{_GRAY}[{now}]{_RESET} {_CYAN}[Tool:{tool_name}]{_RESET} {message}", flush=True)


class VideoOverviewTool(OutputFormatterMixin):
    """Tool for fetching all nodes of a type for a specific video.
    
    Use this instead of search_graph when:
    - Query asks for video overview/summary ("What is this video about?")
    - Query needs the complete timeline or structure
    - Query asks to list/enumerate all topics, chapters, events
    - Query needs holistic understanding (not just similar segments)
    
    This is faster than vector search and returns complete information.
    """
    
    def __init__(self, neo4j_provider):
        """Initialize the tool.
        
        Args:
            neo4j_provider: Neo4jQueryProvider instance.
        """
        self.neo4j_provider = neo4j_provider
    
    async def get_video_overview(
        self,
        video_id: Annotated[str, "Video ID to get overview for"],
        level: Annotated[str, "Granularity level: 'ChapterGroup' for high-level topics, 'Chapter' for segment details, 'Transcript' for all speech"] = "ChapterGroup",
        limit: Annotated[int, "Maximum nodes to return (default 50)"] = 50,
    ) -> str:
        """Get all nodes of a type for a video - no vector search.
        
        Use this for overview queries that need the whole video structure,
        not just semantically similar parts.
        
        Recommended usage:
        - "What is this video about?" → level="ChapterGroup"
        - "List all topics covered" → level="ChapterGroup"  
        - "Give me a timeline" → level="Chapter"
        - "What are all the steps?" → level="Chapter"
        - "Full transcript" → level="Transcript"
        
        Args:
            video_id: The video identifier.
            level: Granularity level to fetch.
            limit: Maximum results.
            
        Returns:
            JSON string with all nodes of the specified type.
        """
        _tool_log("get_video_overview", f"video_id={video_id} level={level} limit={limit}")
        
        try:
            # Validate level
            valid_levels = {"ChapterGroup", "Chapter", "Transcript", "Event", "Object"}
            if level not in valid_levels:
                return json.dumps({
                    "error": f"Invalid level: {level}. Valid: {valid_levels}"
                })
            
            # Fetch all nodes
            results = await self.neo4j_provider.get_all_nodes_for_video(
                video_id=video_id,
                node_type=level,
                limit=limit,
            )
            
            # Format results
            formatted = self._format_results(results, level)
            _tool_log("get_video_overview", f"{_YELLOW}Found {len(results)} {level} nodes{_RESET}")
            return self.format_output(formatted)
            
        except Exception as e:
            logger.error(f"Video overview failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_results(self, results: list, level: str) -> dict:
        """Format results for readability.
        
        Args:
            results: List of SearchResult objects.
            level: Node type level.
            
        Returns:
            Formatted results dictionary.
        """
        node_type = node_registry.get(level)
        
        formatted = {
            "level": level,
            "total": len(results),
            "nodes": [],
        }
        
        for item in results:
            # Handle both SearchResult objects and dicts
            if hasattr(item, 'properties'):
                props = item.properties
                node_id = item.node_id
            else:
                props = item
                node_id = item.get("node_id")
            
            entry = {
                "node_id": node_id,
                "video_id": props.get("video_id"),
            }
            
            # Add type-specific formatted fields
            if node_type:
                entry.update(node_type.format_search_result(props))
            
            # Add temporal info for ordering context
            if level == "ChapterGroup":
                entry["order"] = props.get("order")
            elif level in ("Chapter", "Transcript"):
                entry["chunk_index"] = props.get("chunk_index")
                entry["start_time"] = props.get("start_time")
                entry["end_time"] = props.get("end_time")
            elif level == "Event":
                entry["timestamp"] = props.get("timestamp")
            
            formatted["nodes"].append(entry)
        
        return formatted
