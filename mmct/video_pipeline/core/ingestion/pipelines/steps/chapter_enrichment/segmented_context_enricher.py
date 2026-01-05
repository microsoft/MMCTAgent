"""Segmented chapter enrichment with parallel segments and boundary smoothing."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Adjusted imports for local environment
from mmct.providers.base.llm_provider import BaseLLMProvider
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse

from .context_enricher import (
    ChapterContextEnrichmentStep,
    ChapterRecord,
    ObjectEnrichmentConfig,
)
from .object_enricher import ChapterObjectBundle, ObjectRosterManager, ObjectRosterResults


@dataclass
class SegmentBatch:
    """Represents a contiguous slice of chapters for segmented enrichment."""

    segment_index: int
    records: List[ChapterRecord]


@dataclass
class EnrichedChapterData:
    """Holds the final enriched payload for a single chapter."""

    record: ChapterRecord
    chapter: ChapterCreationResponse


@dataclass
class SegmentResult:
    """Captures the enrichment results for a single segment."""

    segment_index: int
    chapters: List[EnrichedChapterData]


class SegmentedChapterContextEnrichmentStep(ChapterContextEnrichmentStep):
    """Runs enrichment in parallel segments, then smooths boundaries."""

    description = (
        "Segments the chapter timeline into buckets, enriches each bucket sequentially in parallel, "
        "then replays boundary chapters with upstream context before merging summaries and objects."
    )

    def __init__(self, step_id="ingestion.chapters.segmented-context-enrich"):
        self.step_id = step_id

    async def run_direct(
        self,
        records: List[ChapterRecord],
        llm_provider: BaseLLMProvider,
        params: Dict[str, Any] = {},
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:

        # Parse nested config if available, otherwise use flattened params
        seg_params = params.get("segmented_enrichment", {})

        # Merge top-level params with segmented params (segmented takes precedence if keys conflict, though structure differs)
        # Actually, we should look in seg_params first for segment specific settings

        context_window = int(seg_params.get("context_window", params.get("context_window", 3)))
        segment_count = int(seg_params.get("segment_count", 5))
        boundary_window = int(seg_params.get("boundary_window", max(1, context_window)))

        # Object config logic
        # We need to reuse _build_object_config but it expects a dict.
        # We pass the full params dict which contains "object_enrichment" key.
        object_config = self._build_object_config(params)

        # LLM options might be nested in segmented_enrichment or top-level params
        llm_request_options = dict(params.get("llm_request_options", {}) or {})
        # If specific options provided for segmentation
        if "llm_request_options" in seg_params:
            llm_request_options.update(seg_params["llm_request_options"])

        if not records:
            logger.warning(f"[{self.step_id}] No records provided for enrichment.")
            return [], None

        if segment_count <= 0:
            raise ValueError("segment_count must be positive")

        logger.info(
            "[{}] Segmented enrichment over {} chapters (segments={}, context_window={}, boundary_window={})",
            self.step_id,
            len(records),
            segment_count,
            context_window,
            boundary_window,
        )

        if object_config:
            has_objects = any(self._chapter_has_objects(record.raw_chapter) for record in records)
            if not has_objects:
                logger.info(
                    "[{}] Skipping object roster merge because no chapters contain object collections",
                    self.step_id,
                )
                object_config = None

        segments = self._segment_records(records, segment_count)
        if not segments:
            # Fallback if segmentation fails (e.g. empty list?) - simpler handling
            return [], None

        (
            enriched_data,
            object_results,
        ) = await self._process_segments(
            segments,
            context_window=context_window,
            boundary_window=boundary_window,
            llm_request_options=llm_request_options,
            object_config=object_config,
            llm_provider=llm_provider,
        )

        produced_chapters: List[Dict[str, Any]] = []
        for data in enriched_data:
            record = data.record
            produced_chapters.append(
                {
                    "chunk_index": record.chunk_index,
                    "start": record.start,
                    "end": record.end,
                    "duration": record.duration,
                    "transcript": record.transcript,
                    "transcript_segments": record.transcript_segments,
                    "frame_paths": record.frame_paths,
                    "chapter": data.chapter.model_dump(),
                    "original_chapter": record.raw_chapter.model_dump(),
                }
            )

        object_payload: Optional[Dict[str, Any]] = None
        if object_results:
            object_payload = self._serialize_object_results(object_results)

        return produced_chapters, object_payload

    def _segment_records(
        self,
        records: Sequence[ChapterRecord],
        segment_count: int,
    ) -> List[SegmentBatch]:
        total = len(records)
        if total == 0:
            return []
        chunk_size = max(1, math.ceil(total / segment_count))
        segments: List[SegmentBatch] = []
        for idx in range(segment_count):
            start = idx * chunk_size
            if start >= total:
                break
            end = min(total, start + chunk_size)
            segment_records = list(records[start:end])
            if not segment_records:
                continue
            segments.append(SegmentBatch(segment_index=idx, records=segment_records))
        return segments

    async def _process_segments(
        self,
        segments: Sequence[SegmentBatch],
        *,
        context_window: int,
        boundary_window: int,
        llm_request_options: Dict[str, Any],
        object_config: Optional[ObjectEnrichmentConfig],
        llm_provider: BaseLLMProvider,
    ) -> Tuple[List[EnrichedChapterData], Optional[ObjectRosterResults]]:
        segment_results = await self._run_segment_enrichment(
            segments,
            context_window=context_window,
            llm_request_options=llm_request_options,
            llm_provider=llm_provider,
        )

        final_data = await self._smooth_segment_boundaries(
            segment_results,
            context_window=context_window,
            boundary_window=boundary_window,
            llm_request_options=llm_request_options,
            llm_provider=llm_provider,
        )

        object_results: Optional[ObjectRosterResults] = None
        if object_config:
            object_results = await self._run_object_roster(
                final_data, object_config, llm_provider=llm_provider
            )

        return final_data, object_results

    async def _run_segment_enrichment(
        self,
        segments: Sequence[SegmentBatch],
        *,
        context_window: int,
        llm_request_options: Dict[str, Any],
        llm_provider: BaseLLMProvider,
    ) -> List[SegmentResult]:
        tasks = [
            self._enrich_single_segment(
                batch,
                context_window=context_window,
                llm_request_options=llm_request_options,
                llm_provider=llm_provider,
            )
            for batch in segments
        ]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda item: item.segment_index)

    async def _enrich_single_segment(
        self,
        batch: SegmentBatch,
        *,
        context_window: int,
        llm_request_options: Dict[str, Any],
        llm_provider: BaseLLMProvider,
    ) -> SegmentResult:
        logger.info(
            "[{}] Segment {} enriching {} chapters",
            self.step_id,
            batch.segment_index,
            len(batch.records),
        )
        history: List[ChapterCreationResponse] = []
        enriched: List[EnrichedChapterData] = []
        options = dict(llm_request_options)

        for record in batch.records:
            messages = self._build_messages(
                record,
                previous_context=history[-context_window:] if context_window > 0 else [],
            )
            # Log removed for brevity

            raw = await llm_provider.chat_completion(
                messages,
                response_format=ChapterCreationResponse,
                **options,
            )
            response = self._coerce_response(raw)
            enriched.append(
                EnrichedChapterData(
                    record=record,
                    chapter=response,
                )
            )
            history.append(response)

        return SegmentResult(segment_index=batch.segment_index, chapters=enriched)

    async def _smooth_segment_boundaries(
        self,
        segment_results: Sequence[SegmentResult],
        *,
        context_window: int,
        boundary_window: int,
        llm_request_options: Dict[str, Any],
        llm_provider: BaseLLMProvider,
    ) -> List[EnrichedChapterData]:
        if boundary_window <= 0 or context_window <= 0 or len(segment_results) <= 1:
            return self._flatten_segments(segment_results)

        history: List[ChapterCreationResponse] = []
        options = dict(llm_request_options)

        for result in segment_results:
            for idx, data in enumerate(result.chapters):
                needs_smoothing = result.segment_index > 0 and idx < boundary_window
                if needs_smoothing:
                    messages = self._build_messages(
                        data.record,
                        previous_context=history[-context_window:] if context_window > 0 else [],
                    )
                    logger.info(
                        "[{}] Boundary smoothing chunk {} (segment={})",
                        self.step_id,
                        data.record.chunk_index,
                        result.segment_index,
                    )
                    raw = await llm_provider.chat_completion(
                        messages,
                        response_format=ChapterCreationResponse,
                        **options,
                    )
                    data.chapter = self._coerce_response(raw)

                history.append(data.chapter)

        return self._flatten_segments(segment_results)

    # Overriding _run_object_roster to accept llm_provider
    async def _run_object_roster(
        self,
        enriched_data: Sequence[EnrichedChapterData],
        object_config: ObjectEnrichmentConfig,
        llm_provider: BaseLLMProvider,  # Added
    ) -> ObjectRosterResults:
        manager = ObjectRosterManager(
            step_id=f"{self.step_id}.objects",
            llm_request_options=object_config.llm_request_options,
            max_active_context=object_config.max_active_context,
            min_screen_time_seconds=object_config.min_screen_time_seconds,
            min_chunk_occurrences=object_config.min_chunk_occurrences,
            llm_client=llm_provider,  # Usage
        )

        for data in enriched_data:
            record = data.record
            bundle = ChapterObjectBundle(
                chunk_index=record.chunk_index,
                start=record.start,
                end=record.end,
                transcript=record.transcript,
                chapter_summary=data.chapter.detailed_summary,
                actions=data.chapter.action_taken,
                objects=list(data.chapter.object_collection or []),
            )
            await manager.process_chapter(bundle)

        logger.info(
            "[{}] Object roster enrichment complete after {} chunks",
            f"{self.step_id}.objects",
            len(enriched_data),
        )
        return manager.finalize()

    @staticmethod
    def _flatten_segments(segment_results: Sequence[SegmentResult]) -> List[EnrichedChapterData]:
        """Flattens segmented results into a single list while preserving order."""
        flattened = []
        # Segments should be already sorted by index
        sorted_segments = sorted(segment_results, key=lambda s: s.segment_index)
        for segment in sorted_segments:
            flattened.extend(segment.chapters)
        return flattened
