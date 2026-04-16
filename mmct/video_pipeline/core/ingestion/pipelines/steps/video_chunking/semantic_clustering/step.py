"""Semantic clustering step."""

from typing import Optional

from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .semantic_chunker import SemanticChunker


@register_step("ingestion.semantic_clustering")
class SemanticClusteringStep(PipelineStep):
    """
    Groups transcript segments into coherent semantic clusters.

    Params:
        transcript_step: Step ID containing transcript (default: "transcribe")
        max_chunk_duration: Maximum duration of a chunk in seconds (optional)
    """

    step_type = "ingestion.semantic_clustering"
    description = "Group transcript into semantic clusters"

    async def run(self, context: StepContext) -> StepResult:
        """Execute semantic clustering."""
        # Get transcript from previous step or context
        transcript_step = self.get_param("transcript_step", context, default="transcribe")
        transcript = context.data_store.get(transcript_step, "transcript")

        if not transcript:
            raise ValueError(f"Transcript not found from step: {transcript_step}")

        # Get max chunk duration
        max_chunk_duration = self.get_param("max_chunk_duration", context, default=None)

        context.logger.info(
            f"Running semantic clustering for video: {context.video_id} "
            f"with max_chunk_duration={max_chunk_duration}"
        )

        try:
            # Initialize chunker with local embedding provider
            embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
            chunker = SemanticChunker(
                transcript=transcript,
                embedding_provider=embedding_provider,
                max_chunk_duration=max_chunk_duration,
            )

            # Run clustering
            chunked_segments = await chunker.run()

            context.logger.info(f"Generated {len(chunked_segments)} semantic clusters")

            # Serialize segments for data store
            # Assuming Segment objects are dataclasses or Pydantic models.
            # StepResult outputs need to be serializable if possible, but internal data_store can hold objects.
            # However, depending on how data_store is implemented, pure objects might be fine.
            # Let's verify what `chunked_segments` contains.
            # It returns `self.chunked_segments` which is a List of objects usually.

            return StepResult(
                step_id=self.step_id,
                outputs={
                    "clusters": chunked_segments,
                    "clustering_completed": True,
                },
                metrics={"num_clusters": len(chunked_segments)},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Semantic clustering failed: {e}")
            raise
