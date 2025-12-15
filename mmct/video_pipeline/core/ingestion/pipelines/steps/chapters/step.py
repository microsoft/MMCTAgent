"""Chapter generation step wrapper."""

import json
import re
from datetime import timedelta
from typing import Dict, Any, List

from loguru import logger
from pathlib import Path
from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .simple.chapter_ingestion_pipeline import (
    ChapterIngestionPipeline,
)
from .scene_based.chapter_scene_ingestion_pipeline import ChapterSceneIngestionPipeline


def _adjust_transcript_timestamps(transcript_text: str, offset_seconds: float) -> str:
    """
    Adjust timestamps in a transcript string by adding an offset.
    Format expected: HH:MM:SS,mmm --> HH:MM:SS,mmm
    """

    def replace_match(match):
        start_str, end_str = match.groups()

        def parse_to_seconds(t_str):
            h, m, s_ms = t_str.split(":")
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        def format_seconds(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        new_start = format_seconds(parse_to_seconds(start_str) + offset_seconds)
        new_end = format_seconds(parse_to_seconds(end_str) + offset_seconds)
        return f"{new_start} --> {new_end}"

    # Regex to find timestamps
    ts_pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    return re.sub(ts_pattern, replace_match, transcript_text)


@register_step("ingestion.chapters")
class ChapterGenerationStep(PipelineStep):
    """
    Generate semantic chapters from transcript and keyframes.

    Params:
        transcript_step: Step ID containing transcript (default: "transcribe")
        keyframes_step: Step ID containing keyframes (default: "keyframes")
    """

    step_type = "ingestion.chapters"
    description = "Generate semantic chapters from transcript"

    async def run(self, context: StepContext) -> StepResult:
        """Execute chapter generation."""
        transcript_step = self.get_param("transcript_step", context, default="transcribe")
        keyframes_step = self.get_param("keyframes_step", context, default="ingestion.keyframes")

        video_chunking_step = self.get_param(
            "video_chunking_step", context, default="video_chunking"
        )
        video_chunks = context.data_store.get(video_chunking_step, "video_chunks")

        max_concurrent_requests = self.get_param("max_concurrent_requests", context, default=3)
        frame_stacking_grid_size = context.user_params.get("frame_stacking_grid_size", 4)
        chapter_gen_technique = self.get_param("chapter_gen_technique", context, default="scene")

        keyframe_blob_url = await context.provider.storage_provider.get_file_url(
            file_name=f"{context.video_id}"
        )

        all_chapter_responses = []
        all_chapter_transcripts = []
        merged_chapters_path = None

        try:
            # With unified keyframes, we only need video_chunks to exist
            # Keyframes are extracted from full video and filtered by timestamp in ChapterGenerator
            if video_chunks:
                context.logger.info(
                    f"Generating chapters for {len(video_chunks)} chunks using '{chapter_gen_technique}' technique."
                )

                all_chunk_segments = []
                for chunk_data in video_chunks:
                    chunk_transcript = (
                        chunk_data.get("transcript") or chunk_data.get("sentence") or ""
                    )

                    chunk_segment = {
                        "start_time": chunk_data["start_time"],
                        "end_time": chunk_data["end_time"],
                        "sentence": chunk_transcript,
                    }
                    all_chunk_segments.append(chunk_segment)

                context.logger.info(
                    f"Prepared {len(all_chunk_segments)} segments for batch processing."
                )

                if chapter_gen_technique == "scene":
                    pipeline = ChapterSceneIngestionPipeline(
                        hash_id=context.video_id,
                        transcript="",  # Global transcript not used/needed if chunks provide their own text
                        keyframe_blob_url=keyframe_blob_url,
                        llm_provider=context.provider.llm_provider,
                        embedding_provider=context.provider.embedding_provider,
                        keyframe_index_name=context.provider.vectordb_keyframes.index_name,
                        step_params={
                            **context.user_params,
                            **self.params,
                        },
                        video_duration=context.video_duration,  # Global duration
                        max_concurrent_requests=max_concurrent_requests,
                    )

                    all_chapter_responses, all_chapter_transcripts, merged_chapters_path = (
                        await pipeline.run(url=context.url, chunks=all_chunk_segments)
                    )

                elif chapter_gen_technique == "simple":
                    pipeline = ChapterIngestionPipeline(
                        hash_id=context.video_id,
                        keyframe_blob_url=keyframe_blob_url,
                        llm_provider=context.provider.llm_provider,
                        embedding_provider=context.provider.embedding_provider,
                        frame_stacking_grid_size=frame_stacking_grid_size,
                        video_duration=context.video_duration,  # Global duration
                        max_concurrent_requests=max_concurrent_requests,
                    )

                    all_chapter_responses, all_chapter_transcripts, merged_chapters_path = (
                        await pipeline.run(url=context.url, chunks=all_chunk_segments)
                    )

                else:
                    raise NotImplementedError(
                        "Only 'scene' and 'simple' techniques supported for chunked processing."
                    )
                if not all_chapter_responses:
                    all_chapter_responses = []
                if not all_chapter_transcripts:
                    all_chapter_transcripts = []

                context.logger.info(f"Generated {len(all_chapter_responses)} chapters in batch.")

            else:
                # Fallback to Original Logic (Single Video)
                context.logger.info("Running single-video chapter generation.")
                max_chapter_duration = self.get_param("max_chapter_duration", context, default=None)

                # ... [Copy of existing logic for single video retrieval] ...
                transcript = context.data_store.get(transcript_step, "transcript")
                clusters_data = context.data_store.get("semantic_clustering", "clusters")  # default
                if isinstance(clusters_data, dict):
                    chunks = clusters_data.get("clusters")
                else:
                    chunks = clusters_data

                # Load frames logic (copied from original)
                keyframes_data = context.data_store.get_all(keyframes_step)
                frames = keyframes_data.get("frames", [])
                if not frames:
                    kjp = keyframes_data.get("keyframe_json_path")
                    if kjp and Path(kjp).exists():
                        with open(kjp) as f:
                            frames = json.load(f).get("keyframes", [])

                if chapter_gen_technique == "scene":
                    pipeline = ChapterSceneIngestionPipeline(
                        hash_id=context.video_id,
                        transcript=transcript,
                        keyframe_blob_url=keyframe_blob_url,
                        llm_provider=context.provider.llm_provider,
                        embedding_provider=context.provider.embedding_provider,
                        keyframe_index_name=context.provider.vectordb_keyframes.index_name,
                        step_params={**context.user_params, **self.params, "frames_list": frames},
                        video_duration=context.video_duration,
                        max_concurrent_requests=max_concurrent_requests,
                        max_chapter_duration=max_chapter_duration,
                    )
                    responses, transcripts, json_path = await pipeline.run(
                        url=context.url, chunks=chunks
                    )
                    all_chapter_responses = responses
                    all_chapter_transcripts = transcripts
                    merged_chapters_path = json_path
                else:
                    # Simple fallback
                    chapter_pipeline = ChapterIngestionPipeline(
                        hash_id=context.video_id,
                        keyframe_blob_url=keyframe_blob_url,
                        frame_stacking_grid_size=frame_stacking_grid_size,
                        video_duration=context.video_duration,
                        llm_provider=context.provider.llm_provider,
                        embedding_provider=context.provider.embedding_provider,
                        max_concurrent_requests=max_concurrent_requests,
                        max_chapter_duration=max_chapter_duration,
                    )
                    all_chapter_responses, all_chapter_transcripts, _ = await chapter_pipeline.run(
                        url=context.url, chunks=chunks
                    )

            return StepResult(
                step_id=self.step_id,
                outputs={
                    "chapter_responses": all_chapter_responses,
                    "chapter_transcripts": all_chapter_transcripts,
                    "chapters_generated": True,
                },
                metrics={"num_chapters": len(all_chapter_responses)},
                artifacts=[str(merged_chapters_path)] if merged_chapters_path else [],
            )

        except Exception as e:
            context.logger.exception(f"Chapter generation failed: {e}")
            raise
