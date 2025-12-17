"""
Video Chunking Step: Splits video based on transcript clusters or scene detection.
"""

from pathlib import Path
from typing import Dict, Any, List
import json

from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .transcript_chunker import TranscriptChunker
from .scene_chunker import SceneChunker
from .semantic_clustering.semantic_chunker import SemanticChunker


@register_step("ingestion.video_chunking")
class VideoChunkingStep(PipelineStep):
    """
    Splits video into chunks based on the configured strategy.

    Params:
        chunking_strategy: "transcript" (default) or "scene"
        transcript_step: Step ID providing transcript (required for "transcript" strategy)
    """

    description = "Splits video into chunks based on transcript clusters or scene detection."

    async def run(self, context: StepContext) -> StepResult:
        # Check for compressed video from previous step
        compress_step = self.get_param("compress_step", context, default="compress")
        compressed_video_path = context.data_store.get(compress_step, "video_path")

        raw_video_path = compressed_video_path if compressed_video_path else context.video_path
        video_path = Path(raw_video_path).expanduser().resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        strategy = self.params.get("chunking_strategy", "transcript")
        context.logger.info(f"Running video chunking with strategy: {strategy}")

        chunks = []

        if strategy == "transcript":
            # 1. Get Transcript
            transcript_step = self.get_param("transcript_step", context, default="transcribe")
            transcript = context.data_store.get(transcript_step, "transcript")

            if not transcript:
                raise ValueError(f"Transcript not found from step: {transcript_step}")

            # 2. Run Semantic Clustering (Inline)
            context.logger.info("Running inline semantic clustering...")
            clusterer = SemanticChunker(
                transcript=transcript,
                embedding_provider=context.provider.embedding_provider,
            )
            clusters = await clusterer.run()

            if not clusters:
                context.logger.warning(
                    "No clusters generated. Proceeding with empty clusters list."
                )

            # 3. Split Video based on Clusters
            chunker = TranscriptChunker(video_path=video_path, clusters=clusters)
            chunks = await chunker.run()

        elif strategy == "scene":
            # Get Transcript (Optional/Required for alignment)
            transcript_step = self.get_param("transcript_step", context, default="transcribe")
            transcript = context.data_store.get(transcript_step, "transcript")

            # 2. Run Scene Chunker with Transcript
            chunker = SceneChunker(video_path=video_path, params=self.params, transcript=transcript)
            chunks = await chunker.run(context)

        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        context.logger.info(f"Generated {len(chunks)} video chunks")

        return StepResult(
            step_id=self.step_id,
            outputs={"video_chunks": chunks},
            metrics={"num_chunks": len(chunks)},
            artifacts=[],
        )
