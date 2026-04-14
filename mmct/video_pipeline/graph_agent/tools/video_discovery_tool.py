"""Video Discovery Tool - Cross-video search and discovery.

Wraps Neo4jQueryProvider to find relevant videos across
the entire knowledge graph.
"""

from typing import Annotated, Optional, List
import json
from loguru import logger

from mmct.video_pipeline.utils import OutputFormatterMixin

_log = logger.bind(component="Tool:find_relevant_videos")


class VideoDiscoveryTool(OutputFormatterMixin):
    """Tool for cross-video discovery via ChapterGroup search.
    
    Searches ChapterGroup summaries across all videos to find
    relevant videos for a given query.
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
    
    async def find_relevant_videos(
        self,
        query: Annotated[str, "Query to find relevant videos for"],
        video_ids: Annotated[Optional[List[str]], "Optional list of video IDs to constrain discovery to"] = None,
        limit: Annotated[int, "Maximum videos to return"] = 3,
    ) -> str:
        """Find videos relevant to a query by searching ChapterGroup summaries.
        
        When video_ids is provided, only returns results from those videos.
        When omitted, searches across ALL videos.
        
        Args:
            query: Natural language query.
            video_ids: Optional list of video IDs to filter results.
            limit: Maximum number of videos to return.
            
        Returns:
            JSON string with relevant video_ids and their relevance scores.
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        _log.info(f"query='{query}' video_ids={video_ids} limit={limit}")
        try:
            query_embedding = await self._get_embedding_provider().embedding(query)
            
            results = await self.neo4j_provider.find_relevant_videos(
                query_embedding=query_embedding,
                video_ids=video_ids,
                limit=limit,
            )
            
            formatted = self._format_results(results)
            _log.info(f"Found {len(results)} relevant videos")
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
