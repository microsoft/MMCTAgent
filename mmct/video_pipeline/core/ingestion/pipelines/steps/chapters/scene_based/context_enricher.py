"""Context-aware chapter enrichment driven by prior summaries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel

from mmct.providers.base.llm_provider import BaseLLMProvider

from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, ObjectResponse
from .object_enricher import (
    ChapterObjectBundle,
    ObjectRosterManager,
    ObjectRosterResults,
)

# from ..base import PipelineStep, StepContext, StepResult
# from ..registry import register_step -> Removed

PARALLEL_ENRICHMENT_ENABLED = True


@dataclass
class ChapterRecord:
    """Container representing a chapter ready for enrichment."""

    chunk_index: int
    start: float
    end: float
    transcript: str
    transcript_segments: List[Dict[str, Any]]
    frame_paths: List[str]
    raw_chapter: ChapterCreationResponse

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ObjectEnrichmentConfig:
    """Configuration payload for optional object roster tracking."""

    llm_request_options: Dict[str, Any]
    max_active_context: int
    min_screen_time_seconds: float
    min_chunk_occurrences: int


class ChapterContextEnrichmentStep:  # Removed PipelineStep inheritance
    """Enriches per-chunk chapters using limited prior context."""

    description = "Sequentially refines chapter summaries/actions using a sliding contextual window from prior chapters."

    def __init__(self, step_id="ingestion.chapters.context-enrich"):
        self.step_id = step_id

    async def run_direct(
        self,
        records: List[ChapterRecord],
        llm_provider: BaseLLMProvider,
        params: Dict[str, Any] = {},
    ) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]
    ]:  # Returns enriched chapters payloads and object payload

        context_window = int(params.get("context_window", 3))
        llm_request_options: Dict[str, Any] = dict(params.get("llm_request_options", {}) or {})

        logger.info(
            "[{}] Enriching {} chapters with context window {}",
            self.step_id,
            len(records),
            context_window,
        )

        object_config = self._build_object_config(params)
        if object_config:
            has_objects = any(self._chapter_has_objects(record.raw_chapter) for record in records)
            if not has_objects:
                logger.info(
                    "[{}] Skipping object roster merge because no chapters contain object collections",
                    self.step_id,
                )
                object_config = None

        logger.info(
            "[{}] Parallel enrichment {}",
            self.step_id,
            "enabled" if PARALLEL_ENRICHMENT_ENABLED else "disabled",
        )

        (
            chapters,
            object_results,
        ) = await self._enrich_batches(
            records,
            llm_provider=llm_provider,
            context_window=context_window,
            llm_request_options=llm_request_options,
            object_config=object_config,
            parallel=PARALLEL_ENRICHMENT_ENABLED,
        )

        produced_chapters: List[Dict[str, Any]] = []
        for record, enriched in zip(records, chapters):
            # Update record.raw_chapter to reflect enriched version for consistency if needed later?
            # Actually we just return the payload structure
            produced_chapters.append(
                {
                    "chunk_index": record.chunk_index,
                    "start": record.start,
                    "end": record.end,
                    "duration": record.duration,
                    "transcript": record.transcript,
                    "transcript_segments": record.transcript_segments,
                    "frame_paths": record.frame_paths,
                    "chapter": enriched.model_dump(),
                    "original_chapter": record.raw_chapter.model_dump(),
                }
            )

        object_payload: Optional[Dict[str, Any]] = None
        if object_results:
            object_payload = self._serialize_object_results(object_results)

        return produced_chapters, object_payload

    def _build_object_config(self, params: Dict[str, Any]) -> Optional[ObjectEnrichmentConfig]:
        obj_params = params.get("object_enrichment")
        if not obj_params:
            return None
        if not bool(obj_params.get("enabled", True)):
            return None
        return ObjectEnrichmentConfig(
            llm_request_options=dict(obj_params.get("llm_request_options", {}) or {}),
            max_active_context=int(obj_params.get("max_active_context", 12)),
            min_screen_time_seconds=float(obj_params.get("min_screen_time_seconds", 8.0)),
            min_chunk_occurrences=int(obj_params.get("min_chunk_occurrences", 2)),
        )

    @staticmethod
    def _chapter_has_objects(chapter: ChapterCreationResponse) -> bool:
        if not chapter.object_collection:
            return False
        return any(bool(obj) for obj in chapter.object_collection)

    async def _enrich_batches(
        self,
        records: List[ChapterRecord],
        *,
        llm_provider: BaseLLMProvider,
        context_window: int,
        llm_request_options: Dict[str, Any],
        object_config: Optional[ObjectEnrichmentConfig],
        parallel: bool,
    ) -> Tuple[List[ChapterCreationResponse], Optional[ObjectRosterResults]]:
        async def chapter_task() -> List[ChapterCreationResponse]:
            history: List[ChapterCreationResponse] = []
            enriched: List[ChapterCreationResponse] = []
            options = dict(llm_request_options)

            for record in records:
                messages = self._build_messages(
                    record,
                    previous_context=history[-context_window:] if context_window > 0 else [],
                )
                logger.info(
                    "[{}] Enriching chunk {} with {} prior chapters",
                    self.step_id,
                    record.chunk_index,
                    len(history[-context_window:]) if context_window > 0 else 0,
                )
                raw = await llm_provider.chat_completion(
                    messages,
                    response_format=ChapterCreationResponse,
                    **options,
                )
                response = self._coerce_response(raw)
                enriched.append(response)
                history.append(response)

            return enriched

        async def object_task() -> Optional[ObjectRosterResults]:
            if not object_config:
                return None

            manager = ObjectRosterManager(
                step_id=f"{self.step_id}.objects",
                llm_request_options=object_config.llm_request_options,
                max_active_context=object_config.max_active_context,
                min_screen_time_seconds=object_config.min_screen_time_seconds,
                min_chunk_occurrences=object_config.min_chunk_occurrences,
                llm_client=llm_provider,
            )

            for record in records:
                logger.info(
                    "[{}] Object roster ingest for chunk {} (duration {:.2f}s)",
                    f"{self.step_id}.objects",
                    record.chunk_index,
                    record.duration,
                )
                bundle = ChapterObjectBundle(
                    chunk_index=record.chunk_index,
                    start=record.start,
                    end=record.end,
                    transcript=record.transcript,
                    chapter_summary=record.raw_chapter.detailed_summary,
                    actions=record.raw_chapter.action_taken,
                    objects=list(record.raw_chapter.object_collection or []),
                )
                await manager.process_chapter(bundle)

            logger.info(
                "[{}] Object roster enrichment complete after {} chunks",
                f"{self.step_id}.objects",
                len(records),
            )

            return manager.finalize()

        if object_config and parallel:
            (chapters_data, object_results) = await asyncio.gather(
                chapter_task(),
                object_task(),
            )
        else:
            chapters_data = await chapter_task()
            if object_config:
                logger.info(
                    "[{}] Object roster running sequentially after chapter enrichment",
                    f"{self.step_id}.objects",
                )
                object_results = await object_task()
            else:
                object_results = None

        return chapters_data, object_results

    def _serialize_object_results(self, results: ObjectRosterResults) -> Dict[str, Any]:
        return {
            "object_collection": [obj.model_dump() for obj in results.object_collection],
            "object_operations": results.operations,
            "object_stats": results.stats,
        }

    def _build_messages(
        self,
        record: ChapterRecord,
        *,
        previous_context: List[ChapterCreationResponse],
    ) -> List[Dict[str, Any]]:
        context_lines: List[str] = []
        for idx, chapter in enumerate(previous_context, start=1):
            context_lines.append(
                f"Context {idx}: summary='{chapter.detailed_summary}' | actions='{chapter.action_taken or 'None'}'"
            )
        context_block = "\n".join(context_lines) or "No prior chapters available."

        user_prompt = (
            f"Chunk Index: {record.chunk_index}\n"
            f"Start: {self._format_seconds(record.start)} ({record.start:.2f}s)\n"
            f"End: {self._format_seconds(record.end)} ({record.end:.2f}s)\n"
            f"Duration: {record.duration:.2f}s\n\n"
            "Existing Chapter Summary:\n"
            f"{record.raw_chapter.detailed_summary}\n\n"
            "Existing Actions:\n"
            f"{record.raw_chapter.action_taken or 'None provided.'}\n\n"
            "Transcript Snippet:\n"
            f"{record.transcript or 'No transcript text.'}\n\n"
            "Relevant Prior Chapters (oldest first within the window):\n"
            f"{context_block}\n\n"
            "Instructions:\n"
            "- Produce a ChapterCreationResponse payload that reflects any new insights from the transcript and limited context.\n"
            "- Maintain factual grounding; do not invent events that contradict the current transcript.\n"
            "- If context does not add value, keep the summary focused but explain continuity if possible.\n"
            "- For procedural videos, enumerate steps, tools, measurements, and results. For stories, include key characters, motivations, and plot progress."
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a SeniorNarrativeAnalystGPT. Refine chapter summaries so they reflect narrative continuity "
                    "across sequential video sections while staying faithful to the supplied transcript."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    def _coerce_response(self, payload: Any) -> ChapterCreationResponse:
        content: Any = payload
        if isinstance(payload, dict) and "content" in payload:
            content = payload["content"]

        if isinstance(content, ChapterCreationResponse):
            return content

        if isinstance(content, BaseModel):
            return ChapterCreationResponse.model_validate(content.model_dump())

        if isinstance(content, dict):
            return ChapterCreationResponse.model_validate(content)

        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as err:
                raise ValueError("LLM provider returned non-JSON string content") from err
            return ChapterCreationResponse.model_validate(parsed)

        raise TypeError(f"Unsupported enrichment response type: {type(payload)!r}")

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0.0, value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
