"""Router for POST /image-query."""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from api.schemas.image_query import ImageQueryResponse, ImageToolName
from api.services.image_query_service import run_image_query
from api.utilities.dependencies import get_image_agent_provider_config_dep
from api.utilities.file_handler import delete_file, save_upload_file

router = APIRouter()

_ALL_TOOLS = [t.value for t in ImageToolName]


@router.post(
    "",
    response_model=ImageQueryResponse,
    summary="Query an image",
    description=(
        "Upload an image file and ask a natural language question. "
        "Select one or more analysis **tools** via the repeatable query parameter — "
        "e.g. `?tools=vit&tools=ocr`. "
        "Enable **use_critic_agent** to have a critic agent review and refine the answer. "
        "Available tools:\n"
        "- **vit** — Vision Transformer (general image understanding)\n"
        "- **recog** — Image recognition / scene captioning\n"
        "- **object_detection** — YOLOv8 object detection\n"
        "- **ocr** — TrOCR text extraction"
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "ocr_only": {
                            "summary": "Extract text from a menu image",
                            "value": {
                                "query": "What dishes are listed under House Special?",
                                "use_critic_agent": False,
                            },
                        },
                        "full_analysis_with_critic": {
                            "summary": "Full multi-tool analysis with critic review",
                            "value": {
                                "query": "Describe all objects and any text visible in this image.",
                                "use_critic_agent": True,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def image_query(
    image_file: UploadFile = File(..., description="Image file to analyse (JPEG, PNG, etc.)"),
    query: str = Form(
        ...,
        description="Natural language question about the image",
        examples=["What objects are visible in this image?"],
    ),
    use_critic_agent: bool = Form(
        False,
        description="Enable the critic agent to review and refine the answer (adds one extra LLM pass)",
    ),
    tools: List[ImageToolName] = Query(
        default=_ALL_TOOLS,
        description=(
            "Tools to use for analysis. Repeat this parameter to select multiple tools. "
            "Options: **vit**, **recog**, **object_detection**, **ocr**. "
            "Defaults to all four tools."
        ),
    ),
    provider_config=Depends(get_image_agent_provider_config_dep),
):
    saved_path: Optional[Path] = None
    try:
        saved_path = await save_upload_file(image_file, subdir="image")
        result = await run_image_query(
            image_path=saved_path,
            query=query,
            tool_names=[t.value for t in tools],
            use_critic_agent=use_critic_agent,
            provider_config=provider_config,
        )
        return result
    finally:
        if saved_path:
            await delete_file(saved_path)
