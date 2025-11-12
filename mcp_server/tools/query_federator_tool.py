"""
Query Federator MCP Tool

This module provides an MCP tool wrapper for the query federator agent.
It enables intelligent query routing through the MCP server interface.
"""

from mcp_server.server import mcp
from typing import Optional
from mmct.video_pipeline.core.tools.query_federator import query_federator
from loguru import logger
import sys


@mcp.tool(
    name="query_federator_tool",
    description="""The Query Federator Tool intelligently routes video queries for optimal performance and accuracy.

This tool implements a smart two-stage approach:
1. **Classification**: Analyzes the query to determine if it's SIMPLE or COMPLEX
2. **Routing**: Directs SIMPLE queries to fast direct tools, COMPLEX queries to the full planner-critic team

## Query Types

### SIMPLE Queries (Fast Path)
Routed directly to get_video_analysis() or get_context() tools:
- Vague/general queries: "tell me about the video", "what's in this video"
- Keyword searches: "car", "person in blue", "basketball"
- Basic summaries: "what is this video about"
- Object counting: "how many people", "count the cars"
- Entity searches: Single objects or simple descriptions

**Benefits**: 2-5 second response, 500-2000 tokens, direct answers

### COMPLEX Queries (Deep Analysis Path)
Routed to the full planner-critic team with all tools:
- Temporal analysis: "what happened after the person left"
- Multi-step reasoning: "why did X happen and how does it relate to Y"
- Frame-level visual analysis: Detailed visual inspection
- Cross-referencing: Correlating multiple video segments
- Cause-and-effect: Understanding relationships
- Comparative analysis: Comparing elements across time

**Benefits**: 10-30 second response, comprehensive multi-tool analysis, critic validation

## Input Parameters

- query (string, required) → The natural language question to answer
- index_name (string, required) → Vector index name for the video knowledge base
- video_id (string, optional) → Specific video ID to filter results
- url (string, optional) → Video URL (alternative to video_id)
- use_critic_agent (boolean, default=True) → Enable critic for complex queries
- cache (boolean, default=True) → Enable response caching for faster repeated queries

## Output

Returns a structured response containing:
- result: The answer to the query (string)
- tokens: Token usage information (dict with total_input and total_output)
- routing: Which path was taken - "simple_handler" or "planner_team"
- classification: Details about how the query was classified

## When to Use This Tool vs video_agent_tool

**Use query_federator_tool when**:
- You have a mix of simple and complex queries
- You want to optimize for performance and cost
- Query complexity varies (keywords, summaries, AND complex questions)
- You need faster responses for simple queries

**Use video_agent_tool when**:
- All queries require deep analysis
- You always want the full planner-critic workflow
- Consistent comprehensive behavior is needed
- Query complexity is uniformly high

## Examples

```python
# Simple query - fast response
await query_federator_tool(
    query="car",
    index_name="education-video-index-v2"
)
# Classification: SIMPLE, Routing: simple_handler, ~3 seconds

# Complex query - deep analysis
await query_federator_tool(
    query="what happened after the person in red left and how did others react",
    index_name="education-video-index-v2",
    use_critic_agent=True
)
# Classification: COMPLEX, Routing: planner_team, ~15 seconds
```

The query federator automatically optimizes each query for the best balance of speed and accuracy.
"""
)
async def query_federator_tool(
    query: str,
    index_name: str,
    video_id: Optional[str] = None,
    url: Optional[str] = None,
    use_critic_agent: Optional[bool] = True,
    cache: Optional[bool] = True,
):
    """
    MCP tool wrapper for the query federator agent.
    
    Args:
        query: The natural language question to answer
        index_name: Vector index name for video retrieval
        video_id: Optional video ID to filter results
        url: Optional video URL
        use_critic_agent: Whether to use critic for complex queries
        cache: Whether to enable response caching
        
    Returns:
        Structured response with answer, tokens, routing, and classification
    """
    # Configure logger to output to console
    logger.remove()  # Remove default handlers
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    logger.info(f"Query Federator Tool called with query: {query}")
    logger.info(f"Index: {index_name}, Video ID: {video_id}, URL: {url}")
    logger.info(f"Use Critic: {use_critic_agent}, Cache: {cache}")
    
    try:
        # Call the query federator
        result = await query_federator(
            query=query,
            index_name=index_name,
            video_id=video_id,
            url=url,
            use_critic_agent=use_critic_agent,
            cache=cache,
        )
        return result
        
    except Exception as e:
        logger.error(f"Error in query federator tool: {str(e)}")
        logger.exception(e)
        
        return {
            "result": f"Error processing query: {str(e)}",
            "tokens": {"total_input": 0, "total_output": 0},
            "routing": "error",
            "classification": {
                "classification": "ERROR",
                "reasoning": str(e),
                "recommended_tool": "none"
            }
        }
