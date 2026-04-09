"""Routers for POST /video-query and POST /video-query/stream."""

from fastapi import APIRouter, Form
from sse_starlette.sse import EventSourceResponse

from api.schemas.video_query import VideoQueryResponse
from api.services.video_query_service import run_video_query, stream_video_query
from mmct.video_pipeline.query_pipeline import QueryPipelineMode

router = APIRouter()


@router.post(
    "",
    response_model=VideoQueryResponse,
    summary="Query a video",
    description=(
        "Ask a natural language question about a previously ingested video. "
        "Provide the **video_id** returned by the ingestion endpoint. "
        "Choose **graph** mode for agentic graph traversal or **state** mode for "
        "deterministic LangGraph execution."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/x-www-form-urlencoded": {
                    "examples": {
                        "graph_mode": {
                            "summary": "Graph-mode query",
                            "value": {
                                "query": "What are the main topics covered in this video?",
                                "mode": "graph",
                                "video_id": "a3f2c1d8e9b74052abc123def456789012345678901234567890abcdef012345",
                            },
                        },
                        "state_mode": {
                            "summary": "State-mode query",
                            "value": {
                                "query": "What happens at the 2-minute mark?",
                                "mode": "state",
                                "video_id": "a3f2c1d8e9b74052abc123def456789012345678901234567890abcdef012345",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def video_query(
    query: str = Form(
        ...,
        description="Natural language question about the video content",
        examples=["What are the key takeaways from this video?"],
    ),
    video_id: str = Form(
        ...,
        description="Video ID returned by the ingestion endpoint",
    ),
    mode: QueryPipelineMode = Form(
        QueryPipelineMode.STATE,
        description="Pipeline mode: **graph** (agentic knowledge-graph traversal) or **state** (LangGraph state machine)",
    ),
):
    return await run_video_query(query=query, mode=mode, video_id=video_id)


@router.post(
    "/stream",
    summary="Query a video with SSE streaming",
    description=(
        "Same as **POST /video-query** but returns a **Server-Sent Events** stream. "
        "Each event is a JSON-encoded dict. Intermediate events have `type: message`; "
        "the final event has `type: final` with the complete answer in its `data` field. "
        "Error events have `type: error`."
    ),
    response_description="Server-Sent Events stream of pipeline events",
)
async def video_query_stream(
    query: str = Form(
        ...,
        description="Natural language question about the video content",
        examples=["Summarise the presenter's main argument."],
    ),
    video_id: str = Form(
        ...,
        description="Video ID returned by the ingestion endpoint",
    ),
    mode: QueryPipelineMode = Form(
        QueryPipelineMode.STATE,
        description="Pipeline mode: **graph** or **state**",
    ),
):
    async def event_generator():
        async for json_str in stream_video_query(query=query, mode=mode, video_id=video_id):
            yield {"data": json_str}

    return EventSourceResponse(event_generator())
