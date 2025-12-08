"""Transcription step using provider or external transcript."""

import os
import aiofiles
from .base import PipelineStep, StepContext, StepResult
from .registry import register_step
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.video_pipeline.core.ingestion.utils.helper import (
    load_srt,
)


@register_step("ingestion.transcribe")
class TranscriptionStep(PipelineStep):
    """
    Transcribe video or use external transcript with time offset support.

    Params:
        video_step: Step ID containing video_path (default: "compress")
        apply_time_offset: Whether to apply time offset for split videos (default: True)
    """

    step_type = "ingestion.transcribe"
    description = "Transcribe video or load external transcript"

    async def run(self, context: StepContext) -> StepResult:
        """Execute transcription or load external transcript."""
        # Get video path from previous step
        video_step = self.get_param("video_step", context, default="compress")
        video_path = context.data_store.get(video_step, "video_path")

        if not video_path:
            video_path = context.video_path

        context.logger.info(f"Transcription step for video: {context.video_id}")

        try:
            transcript = None
            transcript_path = None

            if context.transcript_path:
                # Use existing transcript as-is
                context.logger.info(f"Using provided transcript: {context.transcript_path}")
                transcript = await load_srt(context.transcript_path)
                transcript_path = context.transcript_path

            else:
                # Generate transcription using provider
                transcription_provider = context.provider.transcription_provider
                context.logger.info(f"Transcribing with {transcription_provider.__class__.__name__}")

                output_dir = await get_media_folder()

                # Extract language name for translation
                source_language_name = None
                if context.language:
                    source_language_name = context.language.name.split("_")[0].capitalize()

                # Transcribe video
                transcript, local_paths = await transcription_provider.transcribe_video(
                    video_path=video_path,
                    hash_id=context.video_id,
                    output_dir=output_dir,
                    language=context.language.value if context.language else None,
                    translate_to_english=True,
                    source_language_name=source_language_name,
                    response_format="srt",
                )

                context.logger.info("Transcription generated successfully")
                
                # Extract transcript path from local_paths
                if transcript and not transcript_path:
                     # Check if it's in local_paths
                     for path in local_paths:
                         if path.endswith(".srt"):
                             transcript_path = path
                             break

            return StepResult(
                step_id=self.step_id,
                outputs={
                    "transcript": transcript,
                    "transcript_path": transcript_path,
                },
                metrics={
                    "transcript_length": len(transcript) if transcript else 0,
                },
                artifacts=[transcript_path] if transcript_path else [],
            )

        except Exception as e:
            context.logger.exception(f"Transcription step failed: {e}")
            raise
