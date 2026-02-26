"""Video Discovery Tool - Cross-video search and discovery.

Wraps Neo4jQueryProvider to find relevant videos across
the entire knowledge graph.
"""

from typing import Annotated, Optional, List
from datetime import datetime
import json
from loguru import logger

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


class VideoDiscoveryTool(OutputFormatterMixin):
    """Tool for cross-video discovery via ChapterGroup search.
    
    Searches ChapterGroup summaries across all videos to find
    relevant videos for a given query.
    """
    
    def __init__(self, neo4j_provider, embedding_provider):
        """Initialize the tool.
        
        Args:
            neo4j_provider: Neo4jQueryProvider instance.
            embedding_provider: Text embedding provider.
        """
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider
    
    async def find_relevant_videos(
        self,
        query: Annotated[str, "Query to find relevant videos for"],
        limit: Annotated[int, "Maximum videos to return"] = 5,
    ) -> str:
        """Find videos relevant to a query by searching ChapterGroup summaries.
        
        Searches across ALL videos (no video_id filter) to discover
        which videos contain content related to the query.
        
        Args:
            query: Natural language query.
            limit: Maximum number of videos to return.
            
        Returns:
            JSON string with relevant video_ids and their relevance scores.
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        _tool_log("find_relevant_videos", f"query='{query}' limit={limit}")
        try:
            query_embedding = await self.embedding_provider.embedding(query)
            
            results = await self.neo4j_provider.find_relevant_videos(
                query_embedding=query_embedding,
                limit=limit,
            )
            
            formatted = self._format_results(results)
            _tool_log("find_relevant_videos", f"{_YELLOW}Found {len(results)} relevant videos{_RESET}")
            return self.format_output(formatted)
            
        except Exception as e:
            logger.error(f"Video discovery failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_results(self, results: List[dict]) -> dict:
        """Format video discovery results.
        
        Args:
            results: List of video results with scores.
            
        Returns:
            Formatted results dictionary.
        """
        formatted = {
            "relevant_videos": [],
            "total_found": len(results),
        }
        
        for video in results:
            entry = {
                "video_id": video.get("video_id"),
                "relevance_score": round(video.get("max_score", 0), 4),
                "matching_topics": video.get("matching_topics", []),
            }
            formatted["relevant_videos"].append(entry)
        
        return formatted
