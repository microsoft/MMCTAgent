"""Routers for POST /video-query and POST /video-query/stream."""

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Form, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas.video_query import VideoQueryResponse
from api.services.video_query_service import run_video_query, stream_video_query
from mmct.video_pipeline.query_pipeline import QueryPipelineMode

router = APIRouter()


def _parse_user_identifier_context(
    raw: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Decode the form field's JSON string into a dict, or return None.

    A 400 is raised if the field is present but doesn't decode to a JSON
    object. Whether the field is required is enforced one layer down by
    VideoQueryPipeline (which raises ConfigurationException when
    ACL_ENABLED=true and the dict is missing).
    """
    if raw is None or raw == "":
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"user_identifier_context must be a JSON object ({exc.msg})",
        )
    if not isinstance(decoded, dict):
        raise HTTPException(
            status_code=400,
            detail="user_identifier_context must be a JSON object, not a "
                   f"{type(decoded).__name__}",
        )
    return decoded


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
    user_identifier_context: Optional[str] = Form(
        None,
        description=(
            "JSON-encoded freeform dict carrying caller identity (e.g. "
            "`{\"email\":\"alice@example.com\",\"graph_token\":\"...\"}`). "
            "Required when the server has ACL_ENABLED=true; ignored otherwise. "
            "Shape is a private contract between the caller and the deployment's "
            "configured access-check callback."
        ),
    ),
):
    user_ctx = _parse_user_identifier_context(user_identifier_context)
    return await run_video_query(
        query=query,
        mode=mode,
        video_id=video_id,
        database=database,
        user_identifier_context=user_ctx,
    )


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
    user_identifier_context: Optional[str] = Form(
        None,
        description=(
            "JSON-encoded freeform dict carrying caller identity. Required "
            "when the server has ACL_ENABLED=true; ignored otherwise."
        ),
    ),
):
    user_ctx = _parse_user_identifier_context(user_identifier_context)

    async def event_generator():
        async for json_str in stream_video_query(
            query=query,
            mode=mode,
            video_id=video_id,
            database=database,
            user_identifier_context=user_ctx,
        ):
            yield {"data": json_str}

    return EventSourceResponse(event_generator())
