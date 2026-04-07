from mcp_server.server import mcp
from typing import Optional, Annotated
from mmct.video_pipeline import VideoAgent

@mcp.tool(
    name="video_agent_tool",
    description="""The Video Agent Tool (video_agent_tool) enables agents to answer natural language questions over ingested video data using the MMCT Agent Framework (Multi-modal Critical Thinking Agent).

This tool orchestrates a video question-answering flow that combines transcript-based search, timestamp extraction, visual frame analysis, and reasoning with optional critic feedback. It abstracts away the complexity of retrieving relevant documents from the vector database — agents only need to provide a query and target index, while the tool handles document retrieval, reasoning, and multimodal fusion internally.

The tool can leverage:
1. AI Search Tool → Retrieves relevant transcripts and summaries.
2. Video Q&A (MMCT Flow) → Uses multi-step reasoning (text + vision) with optional critic agent validation.
3. Critic Agent (optional) → Provides validation and refinement of preliminary answers for improved accuracy.

## Input Schema

- query (string, required) → Natural language user question.
- index_name (string, required) → Target video knowledge base index.
- video_id (string, optional) → Specific video ID to constrain the search.
- url (string, optional) → video source (if available).
- use_critic_agent (boolean, default=True) → Enable/disable critic agent for validation.

## Output

A final structured answer to the user's video-related question, enriched with transcript evidence, relevant timestamps, and optional frame-level insights. When enabled, the critic agent ensures refined, higher-quality responses. 
This tool internally fetches relevant document from the vector database, You won't be required to fetch documents from vector database index.
"""
)
async def video_agent_tool(
    query: str,
    index_name: str,
    video_id: Optional[str] = None,
    url: Optional[str] = None,
    use_critic_agent: Optional[bool] = True,
):
    video_agent = VideoAgent(
        query=query,
        index_name=index_name,
        video_id=video_id,
        url=url,
        use_critic_agent=use_critic_agent,
        stream=True,  # Always enable streaming for log capture
    )
    
    # Run the agent
    response = await video_agent()
    
    return response