"""Chapter enrichment step."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.core.ingestion.models import (
    ChapterMetadata,
    ChapterMetadataCollection,
    ChapterCreationResponse,
)
from mmct.video_pipeline.utils.helper import get_media_folder

from .context_enricher import (
    ChapterRecord,
    ChapterContextEnricher,
)
from .segmented_context_enricher import SegmentedChapterContextEnricher


@register_step("ingestion.chapter_enrichment")
class ChapterEnrichmentStep(PipelineStep):
    """
    Enrich chapters with context and objects, then save to disk.
    """

    step_type = "ingestion.chapter_enrichment"
    description = "Enrich raw chapters and save results."

    async def run(self, context: StepContext) -> StepResult:
        chapters_step_id = self.get_param("chapters_step", context, default="ingestion.chapters")

        # Retrieve outputs from the previous chapters step
        raw_chapters = context.data_store.get(chapters_step_id, "raw_chapters")
        video_summary = context.data_store.get(chapters_step_id, "video_summary") or ""

        if not raw_chapters:
            logger.warning("No raw chapters found for enrichment.")
            return StepResult(step_id=self.step_id, outputs={}, metrics={})

        collect_object_collection_param = self.get_param(
            "collect_object_collection", context, default=True
        )

        # Handle both boolean and dict (new config structure)
        if isinstance(collect_object_collection_param, dict):
            collect_object_collection = (
                str(collect_object_collection_param.get("enabled", "true")).lower() == "true"
            )
        else:
            collect_object_collection = bool(collect_object_collection_param)

        logger.info(
            f"Enriching {len(raw_chapters)} chapters. (collect_object_collection={collect_object_collection})"
        )

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

        step_params = self.params
        enriched_chapters_payload = []
        object_payload = {}

        if collect_object_collection:
            # Check if segmented enrichment is enabled
            segmented_config = self.params.get("segmented_enrichment")
            use_segmented = (
                segmented_config and str(segmented_config.get("enabled", "false")).lower() == "true"
            )

            if use_segmented:
                logger.debug(
                    f"Using Segmented Enrichment (segments={segmented_config.get('segment_count')})"
                )
                enricher = SegmentedChapterContextEnricher()
                enriched_chapters_payload, object_payload = await enricher.run_direct(
                    records=enrichment_records,
                    llm_provider=context.provider.llm_provider,
                    params=step_params,
                )
            else:
                enricher = ChapterContextEnricher()
                enriched_chapters_payload, object_payload = await enricher.run_direct(
                    records=enrichment_records,
                    llm_provider=context.provider.llm_provider,
                    params=step_params,
                )
        else:
            logger.debug(
                "Skipping object enrichment (collect_object_collection=False). Using raw chapters."
            )
            # For compatibility, we format raw_chapters into the structure expected by saving logic
            # run_direct usually returns a list of items where each item has 'chapter' (dict), 'transcript', 'start', 'end'
            for item in raw_chapters:
                enriched_chapters_payload.append(
                    {
                        "chapter": item["chapter"],
                        "transcript": item.get(
                            "transcript", ""
                        ),  # or item['chapter']['transcript'] if exists?
                        # Note: raw_chapters items have 'transcript' as string, check steps.py
                        "start": item["start"],
                        "end": item["end"],
                    }
                )

        # --- Saving Logic ---
        final_chapter_metadata = []
        chapter_responses = []

        # Reconstruct blob url
        keyframe_blob_url = await context.provider.storage_provider.get_file_url(
            file_name=f"{context.video_id}"
        )

        for item in enriched_chapters_payload:
            chap_model = ChapterCreationResponse.model_validate(item["chapter"])
            chapter_responses.append(chap_model)

            # Serialize object_collection if present
            obj_json = "[]"
            if not collect_object_collection:
                obj_json = "object collection for this video is not available"
            elif chap_model.object_collection:
                try:
                    obj_json = json.dumps([o.model_dump() for o in chap_model.object_collection])
                except Exception:
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
                    blob_frames_folder_path=keyframe_blob_url or "None",
                    start_time=item["start"],
                    end_time=item["end"],
                    embeddings=None,
                )
            )

        # Save Chapters JSON
        media_folder = await get_media_folder()
        chapters_dir = os.path.join(media_folder, "chapters")
        os.makedirs(chapters_dir, exist_ok=True)
        json_file_path = os.path.join(chapters_dir, f"chapters_{context.video_id}.json")

        collection = ChapterMetadataCollection(
            hash_video_id=context.video_id,
            video_duration=str(context.video_duration) if context.video_duration else "None",
            url=context.url or "None",
            chapters=final_chapter_metadata,
        )

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(collection.model_dump(), f, indent=2, ensure_ascii=False)

        # Save Object Collection
        object_collection_path = None
        # Persist object collection file if objects or video summary are present.
        if object_payload or video_summary:
            object_collections_dir = os.path.join(media_folder, "object_collections")
            os.makedirs(object_collections_dir, exist_ok=True)
            object_collection_path = os.path.join(
                object_collections_dir, f"object_collection_{context.video_id}.json"
            )

            raw_objects = object_payload.get("object_collection", [])

            if not collect_object_collection:
                objects_json_str = "object collection for this video is not available"
            else:
                objects_json_str = json.dumps(raw_objects) if raw_objects else "[]"

            metadata_dict = {
                "video_id": context.video_id,
                "url": context.url or "",
                "object_collection": objects_json_str,
                "object_count": len(raw_objects),
                "video_summary": video_summary,  # Use the summary from previous step
                "video_duration": context.video_duration if context.video_duration else 0.0,
            }

            with open(object_collection_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        context.logger.debug(f"Saved object collection to {object_collection_path}")

        logger.info(f"Saved chapters to {json_file_path}")

        return StepResult(
            step_id=self.step_id,
            outputs={
                "raw_chapters": enriched_chapters_payload,
                "video_summary": video_summary,
                "chapters_file": json_file_path,
                "object_collection_file": object_collection_path,
            },
            artifacts=[json_file_path],
        )
