"""Upload and indexing step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .upload_orchestrator import UploadOrchestrator


@register_step("ingestion.upload")
class UploadStep(PipelineStep):
    """
    Upload video, keyframes, and index all data (chapters, keyframes, objects).

    Params:
        chapters_step: Step ID that generated chapters (default: "chapters")
        keyframes_step: Step ID that extracted keyframes (default: "keyframes")
        embeddings_step: Step ID that generated embeddings (default: "embeddings")
    """

    step_type = "ingestion.upload"
    description = "Upload files and index data"

    async def run(self, context: StepContext) -> StepResult:
        """Execute upload and indexing."""
        context.logger.debug(f"Starting upload and indexing for video: {context.video_id}")

        try:
            # Get blob manager
            blob_manager = context.provider.storage_provider

            # Get keyframe blob URL
            keyframe_blob_url = await blob_manager.get_file_url(file_name=f"{context.video_id}")

            # Create upload orchestrator
            orchestrator = UploadOrchestrator(
                search_provider={
                    "chapter": context.provider.vectordb_chapter,
                    "object_collection": context.provider.vectordb_object_registry,
                    "keyframe": context.provider.vectordb_keyframes,
                },
                blob_manager=blob_manager,
            )

            # Upload all in parallel
            await orchestrator.upload_all(
                video_id=context.video_id,
                url=context.url,
                keyframe_blob_url=keyframe_blob_url,
            )

            context.logger.debug("Upload and indexing completed successfully")

            return StepResult(
                step_id=self.step_id,
                outputs={"upload_completed": True},
                metrics={},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Upload step failed: {e}")
            raise
