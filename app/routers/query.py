"""Query router — video and image question-answering endpoints.

Exposes the MMCT Agent query capabilities over HTTP.  All endpoints delegate
to the service layer which handles agent orchestration, streaming, and error
handling.
"""

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.query import ImageQueryRequest, VideoQueryRequest, VideoQueryResponse
from app.services.query_services import process_image_query, process_video_query
from app.utilities.request_id_middleware import get_request_id

router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Image query
# ---------------------------------------------------------------------------

@router.post(
    "/query/image",
    summary="Image Analysis Query",
    description=(
        "Upload an image and ask a natural language question. "
        "Specify which analysis tools to run: `object_detection`, `ocr`, `recog`, `vit`."
    ),
)
async def query_image(
    file: UploadFile = File(..., description="Image file to analyse"),
    data: ImageQueryRequest = Depends(),
):
    """Analyse an image using the selected tools and return an answer."""
    return await process_image_query(file, data.model_dump())


# ---------------------------------------------------------------------------
# Video query
# ---------------------------------------------------------------------------

@router.post(
    "/query/video",
    summary="Video Question Answering",
    description=(
        "Ask a natural language question about one or more ingested videos. "
        "Optionally scope the search to a specific video using `video_id` or `url`. "
        "Returns a structured answer with source references."
    ),
    response_model=VideoQueryResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "single_video": {
                            "summary": "Single video query",
                            "value": {
                                "query": "What topics are covered in this lecture?",
                                "video_id": "Dk1toyI7AJs",
                                "use_critic_agent": True,
                            },
                        },
                        "cross_video": {
                            "summary": "Cross-video search",
                            "value": {
                                "query": "Which videos discuss NASA missions?",
                                "use_critic_agent": True,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def query_video(data: VideoQueryRequest):
    """Run a video question-answering query and return a structured response."""
    return await process_video_query(data.model_dump())


@router.post(
    "/query/video/stream",
    summary="Streaming Video Question Answering",
    description=(
        "Same as `/query/video` but streams agent events as NDJSON. "
        "Each line is a JSON object with a `type` field describing the event kind."
    ),
)
async def query_video_stream(data: VideoQueryRequest):
    """Stream video query agent events as NDJSON."""
    request_id = get_request_id()
    body = data.model_dump()
    body["stream"] = True

    response = await process_video_query(body)

    if isinstance(response, StreamingResponse):
        return response

    # Already resolved (non-streaming fallback)
    import json
    async def _wrap():
        yield json.dumps(response if isinstance(response, dict) else response.model_dump()) + "\n"

    return StreamingResponse(_wrap(), media_type="application/x-ndjson")
