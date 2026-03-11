"""V5 Query Router — State machine-driven pipeline endpoints."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.v5_query import V5QueryRequest, V5QueryResponse
from app.services.v5_query_service import process_v5_query, process_v5_query_stream
from app.utilities.request_id_middleware import get_request_id

router = APIRouter()


@router.post(
    "/v5/query",
    summary="V5 Query (State Machine Pipeline)",
    description="""
Query videos using the V5 state-machine pipeline.

**Key differences from V4:**
- Deterministic state machine (no LLM-driven routing)
- Parallel sub-query execution enforced by code
- Structured JSON output from all LLM calls
- Fewer LLM calls (2 in happy path vs 6+ in V4)

Same response format as V4 for compatibility.
""",
    response_model=V5QueryResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "single_video": {
                            "summary": "Single video query",
                            "value": {
                                "query": "How does the farmer collect soil samples?",
                                "video_id": "Dk1toyI7AJs",
                                "use_critic": True,
                            },
                        },
                        "cross_video": {
                            "summary": "Cross-video query",
                            "value": {
                                "query": "Which videos show farming techniques?",
                                "use_critic": True,
                            },
                        },
                        "multi_video": {
                            "summary": "Multi-video query",
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
async def query_v5(data: V5QueryRequest):
    """Process a V5 query through the state machine pipeline."""
    request_id = get_request_id()
    return await process_v5_query(data.model_dump(), request_id=request_id)


@router.post(
    "/v5/query/stream",
    summary="Streaming V5 Query (State Machine Pipeline)",
    description="""
Streaming V5 query with Server-Sent Events (SSE).

Events:
- `connected`: Connection confirmation
- `agent_message`: State transition messages
- `complete`: Final result
- `error`: Error message
""",
)
async def query_v5_stream(data: V5QueryRequest):
    """Process a V5 query with streaming output."""
    request_id = get_request_id()
    body = data.model_dump()

    return StreamingResponse(
        process_v5_query_stream(body, request_id=request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
