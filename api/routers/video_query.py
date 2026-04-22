"""Routers for POST /video-query and POST /video-query/stream."""

from typing import Optional

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
        "Optionally provide the **video_id** returned by the ingestion endpoint to scope the query to a single video. "
        "Choose **graph_agent** mode for agentic graph traversal or **graph_state** mode for "
        "deterministic state machine execution."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/x-www-form-urlencoded": {
                    "examples": {
                        "graph_agent_mode": {
                            "summary": "Graph-agent mode query",
                            "value": {
                                "query": "What are the main topics covered in this video?",
                                "mode": "graph_agent",
                                "video_id": "a3f2c1d8e9b74052abc123def456789012345678901234567890abcdef012345",
                            },
                        },
                        "graph_state_mode": {
                            "summary": "Graph-state mode query",
                            "value": {
                                "query": "What happens at the 2-minute mark?",
                                "mode": "graph_state",
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
    video_id: Optional[str] = Form(
        None,
        description="Video ID returned by the ingestion endpoint. Omit to query across all ingested videos.",
    ),
    mode: QueryPipelineMode = Form(
        QueryPipelineMode.GRAPH_STATE,
        description="Pipeline mode: **graph_agent** (agentic knowledge-graph traversal) or **graph_state** (deterministic state machine)",
    ),
    database: Optional[str] = Form(
        None,
        description="Graph database name override. When provided, queries target this database instead of the server default.",
    ),
):
    return await run_video_query(query=query, mode=mode, video_id=video_id, database=database)


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
    video_id: Optional[str] = Form(
        None,
        description="Video ID returned by the ingestion endpoint. Omit to query across all ingested videos.",
    ),
    mode: QueryPipelineMode = Form(
        QueryPipelineMode.GRAPH_STATE,
        description="Pipeline mode: **graph_agent** or **graph_state**",
    ),
    database: Optional[str] = Form(
        None,
        description="Graph database name override. When provided, queries target this database instead of the server default.",
    ),
):
    async def event_generator():
        async for json_str in stream_video_query(query=query, mode=mode, video_id=video_id, database=database):
            yield {"data": json_str}

    return EventSourceResponse(event_generator())
