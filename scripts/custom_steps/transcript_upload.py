"""Custom step: Transcript upload to blob storage.

Reads the SRT transcript produced by the transcribe step and uploads it
to a configurable blob container.

This is an example of a custom ingestion step that lives outside the
core MMCT library.  Register it by importing this module (or the
``custom_steps`` package) before running the pipeline.

Blob path convention:
    <normalized-video-id>/transcript.srt
"""

import os
import tempfile
from typing import Optional

from loguru import logger

from mmct.video_pipeline import PipelineStep, StepContext, StepResult, register_step
from mmct.providers.base.storage_provider import BaseStorageProvider
from mmct.utils.blob_helpers import normalize_video_id


@register_step("ingestion.transcript_upload")
class TranscriptUploadStep(PipelineStep):
    """Upload the SRT transcript to blob storage.

    Params:
        source_transcribe_step: Step ID for transcript data (default: "transcribe")
        container_name: Blob container (default: "video-transcript-lively")
    """

    step_type = "ingestion.transcript_upload"
    description = "Upload SRT transcript to blob storage."

    async def run(self, context: StepContext) -> StepResult:
        source_step: str = self.get_param(
            "source_transcribe_step", context, default="transcribe"
        )
        container_name: str = self.get_param(
            "container_name", context, default="video-transcript-lively"
        )

        transcript_text: Optional[str] = context.data_store.get(source_step, "transcript")
        transcript_path: Optional[str] = context.data_store.get(source_step, "transcript_path")

        if not transcript_text and not transcript_path:
            context.logger.warning("No transcript available to upload")
            return StepResult(
                step_id=self.step_id,
                outputs={"uploaded": False, "error": "No transcript available"},
                metrics={"uploaded": 0},
            )

        video_id: str = getattr(context, "video_id", "unknown")
        norm_id = normalize_video_id(video_id)

        # Ensure we have a local file to upload
        tmp_created = False
        if transcript_path and os.path.exists(transcript_path):
            upload_path = transcript_path
        elif transcript_text:
            fd, upload_path = tempfile.mkstemp(suffix=".srt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            tmp_created = True
        else:
            context.logger.warning("Transcript path not found and no text available")
            return StepResult(
                step_id=self.step_id,
                outputs={"uploaded": False, "error": "Transcript file missing"},
                metrics={"uploaded": 0},
            )

        storage_provider: Optional[BaseStorageProvider] = getattr(
            context.provider, "storage_provider", None
        )
        if storage_provider is None:
            context.logger.error("No storage provider on context.provider")
            if tmp_created:
                os.unlink(upload_path)
            return StepResult(
                step_id=self.step_id,
                outputs={"uploaded": False, "error": "No storage provider"},
                metrics={"uploaded": 0},
            )

        blob_name = f"{norm_id}/transcript.srt"

        try:
            blob_url = await storage_provider.upload_file(
                file_name=blob_name,
                src_file_path=upload_path,
                folder_name=container_name,
            )
            context.logger.info(f"Uploaded transcript to {blob_url}")
        except Exception as exc:
            logger.error(f"Transcript upload failed: {exc}")
            blob_url = None
        finally:
            if tmp_created:
                os.unlink(upload_path)

        if not blob_url:
            return StepResult(
                step_id=self.step_id,
                outputs={"uploaded": False, "error": "Upload failed"},
                metrics={"uploaded": 0},
            )

        return StepResult(
            step_id=self.step_id,
            outputs={
                "uploaded": True,
                "blob_url": blob_url,
                "blob_name": blob_name,
                "container_name": container_name,
                "normalized_video_id": norm_id,
            },
            metrics={"uploaded": 1},
        )
