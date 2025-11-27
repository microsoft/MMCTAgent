"""Segmented chapter enrichment with parallel segments and boundary smoothing."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from ..base import StepContext, StepResult
from ..registry import register_step
from .context_enricher import (
    ChapterContextEnrichmentStep,
    ChapterRecord,
    ObjectEnrichmentConfig,
    llm_provider,
)
from .models import ChapterCreationResponse
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


@register_step("chapters.segmented-context-enrich")
class SegmentedChapterContextEnrichmentStep(ChapterContextEnrichmentStep):
    """Runs enrichment in five parallel segments, then smooths boundaries."""

    description = (
        "Segments the chapter timeline into buckets, enriches each bucket sequentially in parallel, "
        "then replays boundary chapters with upstream context before merging summaries and objects."
    )

    def run(self, context: StepContext) -> StepResult:
        chapters_step = self.params.get("chapters_step")
        if not chapters_step:
            raise ValueError("'chapters_step' parameter is required")

        context_window = int(self.params.get("context_window", 3))
        segment_count = int(self.params.get("segment_count", 5))
        boundary_window = int(self.params.get("boundary_window", max(1, context_window)))
        llm_request_options: Dict[str, Any] = dict(self.params.get("llm_request_options", {}) or {})
        records = self._load_chapters(context, chapters_step)
        if not records:
            raise ValueError(
                f"Step '{self.step_id}' did not find chapters in step '{chapters_step}'."
            )

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

        object_config = self._build_object_config()
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
            raise ValueError("No segments produced for enrichment")

        (
            enriched_data,
            object_results,
        ) = asyncio.run(
            self._process_segments(
                segments,
                context_window=context_window,
                boundary_window=boundary_window,
                llm_request_options=llm_request_options,
                object_config=object_config,
            )
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

        metrics = {
            "chapters_enriched": float(len(produced_chapters)),
            "context_window": float(context_window),
            "segments": float(len(segments)),
        }

        object_payload: Optional[Dict[str, Any]] = None
        if object_results:
            object_payload = self._serialize_object_results(object_results)
            metrics.update(
                {
                    "unique_objects": float(len(object_results.object_collection)),
                    "objects_filtered_out": float(object_results.filtered_out),
                }
            )

        produced_payload = {
            "chapters": produced_chapters,
        }
        if object_payload:
            produced_payload.update(object_payload)

        return StepResult(
            step_id=self.step_id,
            produced=produced_payload,
            metrics=metrics,
        )

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
    ) -> Tuple[List[EnrichedChapterData], Optional[ObjectRosterResults]]:
        segment_results = await self._run_segment_enrichment(
            segments,
            context_window=context_window,
            llm_request_options=llm_request_options,
        )

        final_data = await self._smooth_segment_boundaries(
            segment_results,
            context_window=context_window,
            boundary_window=boundary_window,
            llm_request_options=llm_request_options,
        )

        object_results: Optional[ObjectRosterResults] = None
        if object_config:
            object_results = await self._run_object_roster(final_data, object_config)

        return final_data, object_results

    async def _run_segment_enrichment(
        self,
        segments: Sequence[SegmentBatch],
        *,
        context_window: int,
        llm_request_options: Dict[str, Any],
    ) -> List[SegmentResult]:
        tasks = [
            self._enrich_single_segment(
                batch,
                context_window=context_window,
                llm_request_options=llm_request_options,
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
            logger.info(
                "[{}] Segment {} chunk {} (local history={})",
                self.step_id,
                batch.segment_index,
                record.chunk_index,
                len(history[-context_window:]) if context_window > 0 else 0,
            )
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
                        "[{}] Boundary smoothing chunk {} (segment={}, localslot={}, context={})",
                        self.step_id,
                        data.record.chunk_index,
                        result.segment_index,
                        idx,
                        len(history[-context_window:]) if context_window > 0 else 0,
                    )
                    raw = await llm_provider.chat_completion(
                        messages,
                        response_format=ChapterCreationResponse,
                        **options,
                    )
                    data.chapter = self._coerce_response(raw)

                history.append(data.chapter)

        return self._flatten_segments(segment_results)

    @staticmethod
    def _flatten_segments(segment_results: Sequence[SegmentResult]) -> List[EnrichedChapterData]:
        flattened: List[EnrichedChapterData] = []
        for result in segment_results:
            flattened.extend(result.chapters)
        return flattened

    async def _run_object_roster(
        self,
        enriched_data: Sequence[EnrichedChapterData],
        object_config: ObjectEnrichmentConfig,
    ) -> ObjectRosterResults:
        manager = ObjectRosterManager(
            step_id=f"{self.step_id}.objects",
            llm_request_options=object_config.llm_request_options,
            max_active_context=object_config.max_active_context,
            min_screen_time_seconds=object_config.min_screen_time_seconds,
            min_chunk_occurrences=object_config.min_chunk_occurrences,
            llm_client=llm_provider,
        )

        for data in enriched_data:
            record = data.record
            logger.info(
                "[{}] Object roster ingest chunk {} (duration {:.2f}s)",
                f"{self.step_id}.objects",
                record.chunk_index,
                record.duration,
            )
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
