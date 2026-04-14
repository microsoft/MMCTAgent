"""Keyframe Retrieval Tool - Vector search for keyframes.

Wraps Neo4jQueryProvider to provide keyframe retrieval via
image embedding vector search.

Note: For graph traversal (Chapter → Keyframe), use GraphTraversalTool.traverse_graph()
"""

from typing import Annotated, Optional, List, Tuple
import json
from loguru import logger

from mmct.video_pipeline.utils import OutputFormatterMixin

_log = logger.bind(component="Tool:search_keyframes")


class KeyframeRetrievalTool(OutputFormatterMixin):
    """Tool for retrieving keyframes via image embedding vector search.
    
    For graph-based keyframe retrieval (Chapter → Keyframe), use
    GraphTraversalTool.traverse_graph(node_ids=[chapter_ids], target="Keyframe")
    """
    
    def __init__(self, neo4j_provider):
        """Initialize the tool.
        
        Args:
            neo4j_provider: Neo4jQueryProvider instance.
        """
        self.neo4j_provider = neo4j_provider
        self._image_embedding_provider = None

    def _get_image_embedding_provider(self):
        """Lazy-load the image embedding provider."""
        if self._image_embedding_provider is None:
            from mmct.providers.custom_providers import FastEmbedQdrantCLIPEmbeddingProvider
            self._image_embedding_provider = FastEmbedQdrantCLIPEmbeddingProvider()
        return self._image_embedding_provider
    
    async def search_keyframes(
        self,
        query: Annotated[str, "Text description of what to find in keyframes"],
        video_ids: Optional[Annotated[List[str], "Video IDs to filter by"]] = None,
        time_start: Optional[Annotated[float, "Start time in seconds for temporal filtering"]] = None,
        time_end: Optional[Annotated[float, "End time in seconds for temporal filtering"]] = None,
        limit: Annotated[int, "Maximum keyframes to return"] = 5,
    ) -> str:
        """Search keyframes by image embedding vector search.
        
        Converts text query to image embedding and searches keyframe index.
        Use this for visual content discovery when you don't have specific chapters.
        Supports optional temporal pre-filtering.
        
        For keyframes from specific chapters, use:
        traverse_graph(node_ids=[chapter_ids], target="Keyframe")
        
        Args:
            query: Text description of visual content to find.
            video_ids: Optional list of video IDs to filter.
            time_start: Optional start time in seconds for temporal filtering.
            time_end: Optional end time in seconds for temporal filtering.
            limit: Maximum keyframes to return.
            
        Returns:
            JSON string with matching keyframes including blob_url, timestamp, video_id.
        """
        # Build time_range tuple if either time param is provided
        time_range: Optional[Tuple[float, float]] = None
        if time_start is not None or time_end is not None:
            time_range = (time_start or 0.0, time_end or float('inf'))
        
        time_str = f" time_range={time_range}" if time_range else ""
        _log.info(f"query='{query}' video_ids={video_ids}{time_str} limit={limit}")
        try:
            # Get image embedding from text query
            query_embedding = await self._get_image_embedding_provider().text_embedding(query)
            
            results = await self.neo4j_provider.search_keyframes(
                query_embedding=query_embedding,
                video_ids=video_ids,
                time_range=time_range,
                limit=limit,
            )
            
            formatted = self._format_results(results)
            _log.info(f"Found {len(results)} keyframes")
            return self.format_output(formatted)
            
        except Exception as e:
            logger.error(f"Keyframe search failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_results(self, results: list) -> dict:
        """Format keyframe vector search results.
        
        Args:
            results: List of SearchResult from vector search.
            
        Returns:
            Formatted results dictionary.
        """
        formatted = {
            "keyframes": [],
            "total": len(results),
        }
        
        for kf in results:
            # Handle both SearchResult objects and dicts
            if hasattr(kf, 'to_dict'):
                kf_dict = kf.to_dict()
            else:
                kf_dict = kf
            
            # Get properties if this is a SearchResult
            props = kf_dict.get("properties", kf_dict)
            
            entry = {
                "keyframe_id": kf_dict.get("node_id") or props.get("node_id"),
                "blob_url": props.get("blob_url"),
                "timestamp": props.get("timestamp"),
                "video_id": props.get("video_id"),
                "chapter_id": props.get("chapter_id"),
                "score": round(kf_dict.get("score", 0), 4),
            }
            formatted["keyframes"].append(entry)
        
        return formatted
