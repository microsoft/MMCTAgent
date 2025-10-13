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
- top_n (integer, default=1) → Number of most relevant answers to return.
- use_computer_vision_tool (boolean, default=True) → Enable/disable frame-level visual analysis.
- use_critic_agent (boolean, default=True) → Enable/disable critic agent for validation.
- stream (boolean, optional, default=False) → Whether to stream intermediate reasoning steps.
- disable_console_log (boolean, optional, default=False) → Suppress console logs during execution.

## Output

A final structured answer to the user’s video-related question, enriched with transcript evidence, relevant timestamps, and optional frame-level insights. When enabled, the critic agent ensures refined, higher-quality responses. 
This tool internally fetches relevant document from the vector database, You won't be required to fetch documents from vector database index.
"""
)
async def video_agent_tool(
    query: str,
    index_name: str,
    video_id: Optional[str] = None,
    url: Optional[str] = None,
    top_n: int = 1,
    use_computer_vision_tool: bool = True,
    use_critic_agent: Optional[bool] = True,
    stream: Optional[bool] = False,
    disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
):
    video_agent = VideoAgent(
        query=query,
        index_name=index_name,
        video_id=video_id,
        url=url,
        top_n=top_n,
        use_computer_vision_tool=use_computer_vision_tool,
        use_critic_agent=use_critic_agent,
        stream=stream,
        disable_console_log=disable_console_log
    )   
    return await video_agent()