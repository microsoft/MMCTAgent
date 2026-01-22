import asyncio
import json
import tempfile, os
from datetime import datetime
from fastapi import HTTPException, UploadFile
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.video_pipeline import VideoAgent
from loguru import logger
from app.config import get_video_agent_provider, get_image_agent_provider
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    TextMessage,
    MultiModalMessage,
    StopMessage,
    HandoffMessage,
    ToolCallSummaryMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    ModelClientStreamingChunkEvent,
    ThoughtEvent,
    BaseMessage,
)
from fastapi.responses import StreamingResponse


def serialize_chunk(chunk):
    # TaskResult at end, keep your logic
    if isinstance(chunk, TaskResult):
        # Extract last message content from TaskResult
        if chunk.messages:
            content = chunk.messages[-1].content
            return json.dumps({"type": "TaskResult", "content": content}) + "\n"
        return json.dumps({"type": "TaskResult", "content": ""}) + "\n"

    # AutoGen BaseMessage types
    if isinstance(chunk, BaseMessage):
        base = {
            "type": getattr(chunk, "type", chunk.__class__.__name__),
            "source": getattr(chunk, "source", None),
        }

        # Text message types
        if isinstance(chunk, TextMessage):
            base["content"] = chunk.content

        elif isinstance(chunk, MultiModalMessage):
            # multimodal content: list of images/strings
            base["content"] = [c for c in chunk.content]

        elif isinstance(chunk, StopMessage):
            base["stop_text"] = chunk.content

        elif isinstance(chunk, HandoffMessage):
            base["target_agent"] = chunk.target
            base["content"] = chunk.content

        elif isinstance(chunk, ToolCallSummaryMessage):
            base["summary"] = chunk.content

        elif isinstance(chunk, ToolCallRequestEvent):
            # list of function calls
            base["tool_calls"] = [
                {"name": fc.name, "arguments": fc.arguments} for fc in chunk.content
            ]

        elif isinstance(chunk, ToolCallExecutionEvent):
            # list of execution results
            base["tool_results"] = [
                {
                    "call_id": r.call_id,
                    "content": getattr(r, "content", None),
                    "is_error": getattr(r, "is_error", None),
                }
                for r in chunk.content
            ]

        elif isinstance(chunk, ModelClientStreamingChunkEvent):
            base["chunk"] = chunk.content
            # also include full_message_id if available
            if getattr(chunk, "full_message_id", None):
                base["full_message_id"] = chunk.full_message_id

        elif isinstance(chunk, ThoughtEvent):
            base["thought"] = chunk.content

        # include any metadata if present
        if hasattr(chunk, "metadata") and chunk.metadata:
            base["metadata"] = chunk.metadata

        return json.dumps(base) + "\n"

    # fallback for dicts
    if isinstance(chunk, dict):
        return json.dumps(chunk) + "\n"

    # fallback for unknowns
    return json.dumps({"type": "unknown", "content": str(chunk)}) + "\n"


async def stream_generator(agent_response):
    """Iterate over the agent's async generator and yield JSON lines."""
    async for chunk in agent_response:
        yield serialize_chunk(chunk)


async def process_image_query(file: UploadFile, body: dict):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # print(body['tools'])
        tool_list = [getattr(ImageQnaTools, t) for t in body["tools"]]
    except AttributeError:
        os.remove(tmp_path)
        raise HTTPException(400, "Invalid tool")

    # Get provider configuration
    provider = get_image_agent_provider()

    stream_enabled = body.get("stream", False)

    agent = ImageAgent(
        query=body["query"],
        image_path=tmp_path,
        tools=tool_list,
        use_critic_agent=body["use_critic_agent"],
        stream=stream_enabled,
        provider=provider,
        use_console=False if stream_enabled else True,  # Disable console for API streaming
    )

    try:
        response = await agent()

        if stream_enabled:
            # response is an async generator
            return StreamingResponse(stream_generator(response), media_type="application/x-ndjson")

        return response
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Image processing failed")
    finally:
        # Note: We can't easily remove the file if streaming,
        # but modern tempfile usage handles cleanup or we rely on OS
        if not stream_enabled:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


async def process_video_query(body: dict):
    """
    Process video query using VideoAgent with configured providers.

    Args:
        body: Request body containing query, video_id, url, and other parameters

    Returns:
        VideoAgent response or StreamingResponse
    """
    # Get provider configuration
    provider = get_video_agent_provider()

    stream_enabled = body.get("stream", False)

    # Create VideoAgent with provider
    agent = VideoAgent(
        query=body["query"],
        video_id=body["video_id"],
        url=body["url"],
        use_critic_agent=body.get("use_critic_agent", True),
        stream=stream_enabled,
        cache=body.get("cache", False),
        provider=provider,
        use_console=False if stream_enabled else True,  # Disable console for API streaming
    )

    try:
        response = await agent()

        if stream_enabled:
            # response is an async generator
            return StreamingResponse(stream_generator(response), media_type="application/x-ndjson")

        return response
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(500, f"Video processing failed: {str(e)}")


async def process_query_v2_stream(body: dict):
    """
    Stream V2 query agent logs as Server-Sent Events (SSE).
    Yields formatted SSE events for each agent message.
    """
    from mmct.v2.orchestrator import process_query_v2
    
    video_provider = get_video_agent_provider()
    image_provider = get_image_agent_provider()
    
    query = body.get("query")
    video_id = body.get("video_id")
    url = body.get("url")
    image_path = body.get("image_path")
    
    def format_sse(event_type: str, data: dict) -> str:
        """Format data as SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    
    try:
        # Yield initial connection event
        yield format_sse("connected", {
            "message": "Stream connected",
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Orchestrator returns dict stream when stream=True, use_console=False
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create unique filename for this query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(logs_dir, f"query_{timestamp}.json")
        
        # Collect all events for saving
        events = []
        
        dict_stream = await process_query_v2(
            query=query,
            video_provider=video_provider,
            image_provider=image_provider,
            video_id=video_id,
            url=url,
            image_path=image_path,
            use_critic=body.get("use_critic_agent", True),
            stream=True,
            cache=body.get("cache", False),
            use_console=True
        )
        
        # Stream is already dict-formatted from orchestrator
        async for event_data in dict_stream:
            events.append(event_data)
            yield format_sse(event_data["type"], event_data)
        
        # Save all events to JSON file
        with open(log_file, "w") as f:
            json.dump(events, f, indent=2, default=str)
        
        logger.info(f"Query events saved to {log_file}")
        
        # Yield completion event
        yield format_sse("complete", {
            "message": "Query processing complete",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"V2 streaming failed: {e}")
        yield format_sse("error", {
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })


async def process_query_v2_endpoint(body: dict):
    """
    Process unified query (Video + Image) using V2 multi-agent system.
    """
    from mmct.v2.orchestrator import process_query_v2
    
    # Get provider configurations
    # We need both providers or a unified one. Existing code has separate getters.
    video_provider = get_video_agent_provider()
    image_provider = get_image_agent_provider()

    stream_enabled = body.get("stream", False)
    
    # Extract known fields
    query = body.get("query")
    video_id = body.get("video_id")
    url = body.get("url")
    image_path = body.get("image_path") # Optional image path for image-first queries
    
    try:
        response = await process_query_v2(
            query=query,
            video_provider=video_provider,
            image_provider=image_provider,
            video_id=video_id,
            url=url,
            image_path=image_path,
            use_critic=body.get("use_critic_agent", True),
            stream=stream_enabled,
            cache=body.get("cache", False),
            use_console=True
        )
        
        if stream_enabled:
             return StreamingResponse(stream_generator(response), media_type="application/x-ndjson")
        
        # Response from orchestrator is now a dict with 'content' and 'token_usage'
        content = response.get("content", "")
        token_usage = response.get("token_usage", {})
        
        # Try to parse content if it's JSON
        try:
            clean_content = content.replace("```json", "").replace("```", "").replace("TERMINATE", "").strip()
            parsed_content = json.loads(clean_content)
            # Include token usage in parsed response
            if isinstance(parsed_content, dict):
                parsed_content["token_usage"] = token_usage
                return parsed_content
            return {"response": parsed_content, "token_usage": token_usage}
        except:
            return {"response": content, "token_usage": token_usage}

    except Exception as e:
        logger.error(f"V2 processing failed: {e}")
        raise HTTPException(500, f"V2 processing failed: {str(e)}")
