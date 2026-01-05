"""Embedding generation step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .embedding_orchestrator import EmbeddingOrchestrator


@register_step("ingestion.embeddings")
class EmbeddingGenerationStep(PipelineStep):
    """
    Generate embeddings for chapters and keyframes in parallel.

    Params:
        chapters_step: Step ID that generated chapters (default: "chapters")
        keyframes_step: Step ID that extracted keyframes (default: "keyframes")
    """

    step_type = "ingestion.embeddings"
    description = "Generate embeddings for chapters and keyframes"

    async def run(self, context: StepContext) -> StepResult:
        """Execute embedding generation."""
        context.logger.debug(f"Starting embedding generation for video: {context.video_id}")

        try:
            # Create orchestrator
            orchestrator = EmbeddingOrchestrator(
                embedding_provider=context.provider.embedding_provider,
                image_embedding_provider=context.provider.image_embedding_provider,
            )

            # Generate all embeddings in parallel
            await orchestrator.generate_all_embeddings(context.video_id)

            context.logger.debug("All embeddings generated successfully")

            return StepResult(
                step_id=self.step_id,
                outputs={"embeddings_generated": True},
                metrics={},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Embedding generation failed: {e}")
            raise
