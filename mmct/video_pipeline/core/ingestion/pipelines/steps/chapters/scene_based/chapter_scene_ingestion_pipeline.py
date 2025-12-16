"""
Scene Narrative Pipeline Module

This module orchestrates the experimental chapter generation workflow:
1. Scene LLM Analysis (Frames + Transcript)
2. Context Enrichment
3. Object Enrichment (Dedup)
4. Timeline Summary
"""

import os
import json
from typing import List, Optional, Tuple, Any, Dict
from loguru import logger

from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.base import BaseLLMProvider, BaseEmbeddingProvider
from mmct.video_pipeline.core.ingestion.models import ChapterMetadata, ChapterMetadataCollection

# Experimental components
from mmct.video_pipeline.core.ingestion.pipelines.steps.chapters.simple.video_summary import (
    VideoSummary,
)
from .scene_llm import SceneLLMChapterGenerationStep
from .context_enricher import (
    ChapterRecord,
    ChapterContextEnrichmentStep,
)
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse
from .object_enricher import ChapterObjectBundle
from .timeline_summary import ChapterTimelineSummaryStep

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class ChapterSceneIngestionPipeline:
    """
    Orchestrates the experimental 'Scene Narrative' workflow.
    """

    def __init__(
        self,
        hash_id: str,
        keyframe_blob_url: str,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseEmbeddingProvider,
        frame_stacking_grid_size: int = 4,
        step_params: Dict[str, Any] = {},
        video_duration: Optional[float] = None,
        max_concurrent_requests: int = 3,
        max_chapter_duration: Optional[int] = None,
    ) -> None:

        self.hash_id = hash_id
        self.keyframe_blob_url = keyframe_blob_url
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.video_duration = video_duration
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.max_concurrent_requests = max_concurrent_requests
        self.step_params = step_params

        # Initialize steps
        self.scene_llm_step = SceneLLMChapterGenerationStep()
        self.context_enricher_step = ChapterContextEnrichmentStep()
        self.timeline_summary_step = ChapterTimelineSummaryStep()
        self.video_summary_processor = VideoSummary(llm_provider=self.llm_provider)

        self.chapter_responses = []
        self.chapter_transcripts = []
        self.chapter_timestamps = []

    async def run(
        self,
        url: Optional[str] = None,
        chunks: List[Dict[str, Any]] = [],
    ) -> Tuple[Optional[List], Optional[List], str]:

        chunked_segments = chunks
        if not chunked_segments:
            logger.error("No semantic chunks provided to ChapterSceneIngestionPipeline")
            return None, None, ""

        # Prepare chunks format for SceneLLM
        formatted_chunks = []
        for i, segment in enumerate(chunked_segments):
            start = segment.get("start_time") if isinstance(segment, dict) else segment.start_time
            end = segment.get("end_time") if isinstance(segment, dict) else segment.end_time
            sentence = segment.get("sentence") if isinstance(segment, dict) else segment.sentence

            formatted_chunks.append(
                {
                    "index": i,
                    "start": start,
                    "end": end,
                    "transcript": {"text": sentence},
                    "sentence": sentence,
                }
            )

        # 2. Scene LLM
        logger.info("[SceneNarrative] Step 2: Scene LLM Analysis...")

        # Prepare params for SceneLLM
        # Use defaults but allow step_params to override or provide extra (like object enrichment)
        step_params = {
            "start_time": 0.0,
            "max_parallel_requests": self.max_concurrent_requests,
            **self.step_params,  # Merge provided params (containing object_enrichment)
        }

        raw_chapters = await self.scene_llm_step.run_direct(
            chunks=formatted_chunks,
            llm_provider=self.llm_provider,
            video_id=self.hash_id,
            params=step_params,
        )

        # 3. Context Enrichment
        logger.info("[SceneNarrative] Step 3: Context Enrichment & Object Deduplication...")

        # Prepare records for enrichment
        enrichment_records = []
        for item in raw_chapters:
            enrichment_records.append(
                ChapterRecord(
                    chunk_index=item["chunk_index"],
                    start=item["start"],
                    end=item["end"],
                    transcript=item["transcript"],
                    transcript_segments=item.get("transcript_segments", []),
                    frame_paths=item.get("frame_paths", []),
                    raw_chapter=ChapterCreationResponse.model_validate(item["chapter"]),
                )
            )

        enriched_chapters_payload, object_payload = await self.context_enricher_step.run_direct(
            records=enrichment_records,
            llm_provider=self.llm_provider,
            params=step_params,
        )

        # 4. Save to JSON
        logger.info("[SceneNarrative] Step 4: Saving Chapters...")

        final_chapter_metadata = []
        self.chapter_responses = []
        self.chapter_transcripts = []
        self.chapter_timestamps = (
            []
        )  # Not fully populated in scene logic yet, but available in items

        for item in enriched_chapters_payload:
            chap_model = ChapterCreationResponse.model_validate(item["chapter"])
            self.chapter_responses.append(chap_model)
            self.chapter_transcripts.append(item["transcript"])
            self.chapter_timestamps.append([item["start"], item["end"]])

            # Serialize object_collection if present
            obj_json = "[]"
            if chap_model.object_collection:
                try:
                    obj_json = json.dumps([o.model_dump() for o in chap_model.object_collection])
                except:
                    pass

            final_chapter_metadata.append(
                ChapterMetadata(
                    topic_of_video="None",
                    action_taken=chap_model.action_taken or "None",
                    detailed_summary=chap_model.detailed_summary,
                    category="None",
                    sub_category="None",
                    text_from_scene=chap_model.text_from_scene or "None",
                    object_collection=obj_json,
                    chapter_transcript=item["transcript"],
                    blob_frames_folder_path=self.keyframe_blob_url or "None",
                    start_time=item["start"],
                    end_time=item["end"],
                    embeddings=None,
                )
            )

        # Save Chapters JSON
        media_folder = await get_media_folder()
        chapters_dir = os.path.join(media_folder, "chapters")
        os.makedirs(chapters_dir, exist_ok=True)
        json_file_path = os.path.join(chapters_dir, f"chapters_{self.hash_id}.json")

        collection = ChapterMetadataCollection(
            hash_video_id=self.hash_id,
            video_duration=str(self.video_duration) if self.video_duration else "None",
            url=url or "None",
            chapters=final_chapter_metadata,
        )

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(collection.model_dump(), f, indent=2, ensure_ascii=False)

        # Save Object Collection if available
        if object_payload:
            object_collections_dir = os.path.join(media_folder, "object_collections")
            os.makedirs(object_collections_dir, exist_ok=True)
            object_collection_path = os.path.join(
                object_collections_dir, f"object_collection_{self.hash_id}.json"
            )

            # Prepare data matching ObjectCollectionMetadata schema
            # object_payload from context_enricher has "object_collection" as List[Dict]
            # We need to convert it to JSON string string for ObjectCollectionMetadata

            raw_objects = object_payload.get("object_collection", [])
            objects_json_str = json.dumps(raw_objects) if raw_objects else "[]"

            aggregated_video_summary = await self.video_summary_processor.create_video_summary(
                chapter_responses=self.chapter_responses
            )

            metadata_dict = {
                "video_id": self.hash_id,
                "url": url or "",
                "object_collection": objects_json_str,
                "object_count": len(raw_objects),
                "video_summary": aggregated_video_summary or "",
                "video_duration": self.video_duration if self.video_duration else 0.0,
                # "created_at": ... (defaults to now in model)
                # context_enricher stats can be ignored or logged
            }

            with open(object_collection_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved object collection to {object_collection_path}")

        logger.info(
            f"Saved {len(final_chapter_metadata)} experimental chapters to {json_file_path}"
        )

        return self.chapter_responses, self.chapter_transcripts, json_file_path
