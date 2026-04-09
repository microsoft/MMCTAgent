"""Router for POST /ingestion."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.schemas.ingestion import IngestionResponse, LanguageCode
from api.services.ingestion_service import run_ingestion
from api.utilities.dependencies import get_ingestion_providers_dep
from api.utilities.file_handler import delete_file, save_upload_file

router = APIRouter()


@router.post(
    "",
    response_model=IngestionResponse,
    summary="Ingest a video",
    description=(
        "Upload an MP4 file and run the full MMCT ingestion pipeline: "
        "transcription, keyframe extraction, chapter grouping, embedding, "
        "and Neo4j knowledge-graph construction. "
        "The uploaded file is saved locally for the duration of the pipeline, then deleted."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "english_video": {
                            "summary": "English-language video with default settings",
                            "value": {
                                "language": "en-US",
                                "verbosity": 0,
                                "frame_stacking_grid_size": 4,
                                "save_local_report": False,
                            },
                        },
                        "french_playlist_video": {
                            "summary": "French video that is part of a playlist",
                            "value": {
                                "language": "fr-FR",
                                "url": "https://example.com/playlist/video-3",
                                "playlist_id": "PL_abc123",
                                "playlist_order": 3,
                                "verbosity": 1,
                            },
                        },
                    }
                }
            }
        }
    },
)
async def ingest_video(
    video_file: UploadFile = File(..., description="MP4 video file to ingest"),
    language: LanguageCode = Form(  # type: ignore[valid-type]
        "en-US",
        description="BCP-47 language code used for transcription (e.g. en-US, fr-FR, hi-IN)",
    ),
    url: Optional[str] = Form(None, description="Optional source URL associated with the video"),
    frame_stacking_grid_size: int = Form(
        4, description="Grid size for horizontal frame stacking (1 disables stacking)"
    ),
    save_local_report: bool = Form(False, description="Persist the pipeline report to disk"),
    verbosity: int = Form(
        0, description="Logging verbosity level: 0 = progress bar only, 1 = info, 2 = debug"
    ),
    playlist_id: Optional[str] = Form(None, description="Optional playlist ID this video belongs to"),
    playlist_order: Optional[int] = Form(None, description="1-based position of this video within its playlist"),
    providers=Depends(get_ingestion_providers_dep),
):
    saved_path: Optional[Path] = None
    try:
        saved_path = await save_upload_file(video_file, subdir="video")
        result = await run_ingestion(
            video_path=saved_path,
            providers=providers,
            language_value=language if isinstance(language, str) else language.value,
            url=url,
            frame_stacking_grid_size=frame_stacking_grid_size,
            save_local_report=save_local_report,
            verbosity=verbosity,
            playlist_id=playlist_id,
            playlist_order=playlist_order,
        )
        return result
    finally:
        if saved_path:
            await delete_file(saved_path)
