from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse

# from app.schemas.query import ImageQueryRequest, VideoQueryRequest
# from app.services.query_services import process_image_query, process_video_query
# from app.schemas.query import UnifiedQueryRequest
# from app.services.query_services import process_query_v2_endpoint, process_query_v2_stream
from app.schemas.v4_query import V4QueryRequest, V4QueryResponse
from app.services.v4_query_service import process_v4_query, process_v4_query_stream
from app.utilities.request_id_middleware import get_request_id

router = APIRouter()

# @router.post(
#     "/query-on-images",
#     summary="Query image with specified tools",
#     description="Upload an image and specify which tools to run. Tools: object_detection, ocr, recog, vit.",
# )
# async def query_images(file: UploadFile = File(...), data: ImageQueryRequest = Depends()):
#     return await process_image_query(file, data.model_dump())

# @router.post("/query-on-videos")
# async def query_videos(data: VideoQueryRequest = Depends()):
#     return await process_video_query(data.model_dump())

# @router.post(
#     "/v2/query",
#     summary="Unified V2 Query (Video + Image)",
#     description="Unified endpoint for video and image queries using the V2 multi-agent system.",
# )
# async def query_unified(
#     file: UploadFile = File(None),
#     data: UnifiedQueryRequest = Depends()
# ):
#     body = data.model_dump()
#
#     if file:
#         import tempfile
#         import os
#         suffix = os.path.splitext(file.filename)[1]
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(await file.read())
#             body["image_path"] = tmp.name
#
#     return await process_query_v2_endpoint(body)


# @router.post(
#     "/v2/query/stream",
#     summary="Streaming V2 Query (Video + Image)",
#     description="Streaming endpoint that sends agent logs as Server-Sent Events (SSE).",
# )
# async def query_unified_stream(
#     file: UploadFile = File(None),
#     data: UnifiedQueryRequest = Depends()
# ):
#     body = data.model_dump()
#
#     if file:
#         import tempfile
#         import os
#         suffix = os.path.splitext(file.filename)[1]
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(await file.read())
#             body["image_path"] = tmp.name
#
#     return StreamingResponse(
#         process_query_v2_stream(body),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no",  # Disable nginx buffering
#         }
#     )


# =============================================================================
# V4 Query Endpoints (Neo4j Graph Backend)
# =============================================================================


@router.post(
    "/v4/query",
    summary="V4 Query (Neo4j Graph)",
    description="""
Query videos using the V4 pipeline with Neo4j graph backend.

Features:
- Multi-granularity search (ChapterGroup, Chapter, Event, Object, Keyframe)
- Cross-video discovery
- HNSW vector search for low latency
- Citations with video_id and timestamps

The response includes:
- `answer`: Markdown text with inline citations [1], [2], etc.
- `sources`: List of citation sources with video_id, timestamps, and type
""",
    response_model=V4QueryResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "single_video": {
                            "summary": "Single video query",
                            "description": "Query a specific video by ID",
                            "value": {
                                "query": "How does the farmer collect soil samples?",
                                "video_id": "Dk1toyI7AJs",
                                "use_critic": True,
                            },
                        },
                        "cross_video": {
                            "summary": "Cross-video query",
                            "description": "Search across all indexed videos",
                            "value": {
                                "query": "Which videos show farming techniques?",
                                "use_critic": True,
                            },
                        },
                        "multi_video": {
                            "summary": "Multi-video query",
                            "description": "Search within specific video IDs",
                            "value": {
                                "query": "Compare soil preparation methods",
                                "video_ids": ["Dk1toyI7AJs", "E9sM2b3uV3c"],
                                "use_critic": False,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def query_v4(data: V4QueryRequest):
    """Process a V4 query against the Neo4j knowledge graph."""
    request_id = get_request_id()
    return await process_v4_query(data.model_dump(), request_id=request_id)


@router.post(
    "/v4/query/stream",
    summary="Streaming V4 Query (Neo4j Graph)",
    description="""
Streaming V4 query endpoint using Server-Sent Events (SSE).

Events:
- `connected`: Initial connection confirmation
- `agent_message`: Intermediate agent messages (planner, video, image, critic)
- `complete`: Final result with answer and sources
- `error`: Error message if processing fails
""",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "single_video_stream": {
                            "summary": "Stream single video query",
                            "value": {
                                "query": "How does the farmer collect soil samples?",
                                "video_id": "Dk1toyI7AJs",
                                "use_critic": True,
                            },
                        },
                        "cross_video_stream": {
                            "summary": "Stream cross-video query",
                            "value": {
                                "query": "Which videos show farming techniques?",
                                "use_critic": True,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def query_v4_stream(data: V4QueryRequest):
    """Process a V4 query with streaming output."""
    request_id = get_request_id()
    body = data.model_dump()

    return StreamingResponse(
        process_v4_query_stream(body, request_id=request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
