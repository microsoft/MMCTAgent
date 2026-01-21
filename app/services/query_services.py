import asyncio
import json
import tempfile, os
from fastapi import HTTPException, UploadFile
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.video_pipeline import VideoAgent
from loguru import logger
from config import get_video_agent_provider, get_image_agent_provider
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
