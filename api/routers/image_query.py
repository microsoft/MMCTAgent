"""Router for POST /image-query."""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from api.schemas.image_query import ImageQueryResponse, ImageToolName
from api.services.image_query_service import run_image_query
from api.utilities.dependencies import get_image_agent_provider_config_dep
from api.utilities.file_handler import delete_file, save_upload_file

router = APIRouter()


@router.post(
    "",
    response_model=ImageQueryResponse,
    summary="Query an image",
    description=(
        "Upload an image file and ask a natural language question. "
        "Enable or disable individual analysis **tools** using the boolean checkboxes below. "
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
                            "summary": "OCR only — extract text",
                            "value": {
                                "query": "What dishes are listed under House Special?",
                                "use_critic_agent": False,
                                "use_vit": False,
                                "use_recog": False,
                                "use_object_detection": False,
                                "use_ocr": True,
                            },
                        },
                        "full_analysis_with_critic": {
                            "summary": "Full multi-tool analysis with critic review",
                            "value": {
                                "query": "Describe all objects and any text visible in this image.",
                                "use_critic_agent": True,
                                "use_vit": True,
                                "use_recog": True,
                                "use_object_detection": True,
                                "use_ocr": True,
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
    use_vit: bool = Form(True, description="Enable **vit** — Vision Transformer (general image understanding)"),
    use_recog: bool = Form(True, description="Enable **recog** — Image recognition / scene captioning"),
    use_object_detection: bool = Form(True, description="Enable **object_detection** — YOLOv8 object detection"),
    use_ocr: bool = Form(True, description="Enable **ocr** — TrOCR text extraction"),
    provider_config=Depends(get_image_agent_provider_config_dep),
):
    tool_map = {
        ImageToolName.vit: use_vit,
        ImageToolName.recog: use_recog,
        ImageToolName.object_detection: use_object_detection,
        ImageToolName.ocr: use_ocr,
    }
    selected = [t.value for t, enabled in tool_map.items() if enabled]
    if not selected:
        raise HTTPException(status_code=400, detail="At least one tool must be enabled.")

    saved_path: Optional[Path] = None
    try:
        saved_path = await save_upload_file(image_file, subdir="image")
        result = await run_image_query(
            image_path=saved_path,
            query=query,
            tool_names=selected,
            use_critic_agent=use_critic_agent,
            provider_config=provider_config,
        )
        return result
    finally:
        if saved_path:
            await delete_file(saved_path)
