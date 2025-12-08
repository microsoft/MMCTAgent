"""Pre-validation step combining early check and audio validation."""

import asyncio
from .base import PipelineStep, StepContext, StepResult
from .registry import register_step
from mmct.video_pipeline.utils.helper import get_file_hash
from mmct.video_pipeline.core.ingestion.utils.helper import check_video_already_ingested


@register_step("ingestion.pre_validation")
class PreValidationStep(PipelineStep):
    """
    Combined pre-validation step that:
    1. Checks if video is already ingested
    2. Validates audio stream exists (if needed)

    Params:
        skip_if_exists: Skip processing if video already exists (default: True)
        require_audio_when_no_transcript: Require audio when transcript not provided (default: True)
    """

    step_type = "ingestion.pre_validation"
    description = "Pre-validate video before processing"

    async def run(self, context: StepContext) -> StepResult:
        """Execute pre-validation checks."""
        skip_if_exists = self.get_param("skip_if_exists", context, default=True)
        require_audio = self.get_param("require_audio_when_no_transcript", context, default=True)

        context.logger.info("Running pre-validation checks...")

        # ============================================================
        # CHECK 1: Early Ingestion Check
        # ============================================================
        try:
            context.logger.info("1. Checking if video already ingested...")
            video_hash_id = await get_file_hash(context.video_path)

            is_already_ingested = await check_video_already_ingested(
                hash_id=video_hash_id,
                search_provider=context.provider.vectordb_chapter,
            )

            if is_already_ingested and skip_if_exists:
                context.logger.info(
                    f"Video {video_hash_id} already ingested. Pipeline will be skipped."
                )
                raise RuntimeError(
                    f"Video {video_hash_id} already ingested. Skipping pipeline."
                )

            context.logger.info("✓ Video not found in index, proceeding...")

        except RuntimeError:
            raise
        except Exception as e:
            context.logger.exception(f"Early check failed: {e}")
            raise

        # ============================================================
        # CHECK 2: Audio Validation
        # ============================================================
        has_audio = True
        audio_check_skipped = False

        # Only validate if transcript not provided
        if context.transcript_path:
            context.logger.info("2. Transcript provided, skipping audio validation")
            audio_check_skipped = True
        elif not require_audio:
            context.logger.info("2. Audio validation not required")
            audio_check_skipped = True
        else:
            context.logger.info("2. Validating video has audio stream...")
            try:
                # Check for audio stream using ffprobe
                process = await asyncio.create_subprocess_exec(
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    context.video_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, _ = await process.communicate()
                has_audio = stdout and stdout.strip() == b"audio"

                if not has_audio:
                    error_msg = (
                        "ERROR: Video does not have an audio stream!\n"
                        "Please provide either:\n"
                        "  1. A video file with audio, OR\n"
                        "  2. A transcript file using the transcript_path parameter"
                    )
                    context.logger.error(error_msg)
                    raise ValueError(error_msg)

                context.logger.info("✓ Video has audio stream")

            except ValueError:
                raise
            except Exception as e:
                context.logger.warning(f"Could not check for audio stream: {e}")
                # Assume has audio if check fails
                has_audio = True

        context.logger.info("✓ All pre-validation checks passed")

        return StepResult(
            step_id=self.step_id,
            outputs={
                "is_already_ingested": is_already_ingested,
                "has_audio": has_audio,
                "audio_check_skipped": audio_check_skipped,
                "should_continue": not is_already_ingested,
            },
            metrics={},
            artifacts=[],
        )
