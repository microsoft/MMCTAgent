"""Context-aware chapter enrichment driven by prior summaries."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from providers.base.llm_provider import LLMProvider
from providers.factory import provider_factory

from .models import ChapterCreationResponse

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step

llm_provider = provider_factory.create_llm_provider()


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


class ContextEnrichmentResponse(BaseModel):
    """Structured response containing the enriched chapter plus a global summary delta."""

    chapter: ChapterCreationResponse = Field(
        ...,
        description="Fully enriched chapter payload",
    )
    global_summary_update: str = Field(
        ...,
        description="Narrative delta summarizing newly discovered information to append to the global story",
    )


@register_step("chapters.context-enrich")
class ChapterContextEnrichmentStep(PipelineStep):
    """Enriches per-chunk chapters using limited prior context."""

    description = "Sequentially refines chapter summaries/actions using a sliding contextual window from prior chapters."

    def run(self, context: StepContext) -> StepResult:
        chapters_step = self.params.get("chapters_step")
        if not chapters_step:
            raise ValueError("'chapters_step' parameter is required")

        context_window = int(self.params.get("context_window", 3))
        llm_request_options: Dict[str, Any] = dict(self.params.get("llm_request_options", {}) or {})

        records = self._load_chapters(context, chapters_step)
        if not records:
            raise ValueError(
                f"Step '{self.step_id}' did not find chapters in step '{chapters_step}'."
            )

        logger.info(
            "[{}] Enriching {} chapters with context window {}",
            self.step_id,
            len(records),
            context_window,
        )

        chapters, global_summary, summary_sections = asyncio.run(
            self._enrich_sequentially(
                records,
                context_window=context_window,
                llm_request_options=llm_request_options,
            )
        )

        produced_chapters: List[Dict[str, Any]] = []
        for record, enriched in zip(records, chapters):
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

        metrics = {
            "chapters_enriched": float(len(produced_chapters)),
            "context_window": float(context_window),
        }

        produced_payload = {
            "chapters": produced_chapters,
            "global_summary": global_summary,
            "global_summary_sections": summary_sections,
        }

        return StepResult(
            step_id=self.step_id,
            produced=produced_payload,
            metrics=metrics,
        )

    def _load_chapters(self, context: StepContext, chapters_step: str) -> List[ChapterRecord]:
        payload = context.data_store.get(chapters_step, {})
        raw_chapters: List[Dict[str, Any]] = payload.get("chapters", [])
        records: List[ChapterRecord] = []
        for entry in raw_chapters:
            chapter_data = entry.get("chapter")
            if not chapter_data:
                continue
            chapter = ChapterCreationResponse.model_validate(chapter_data)
            records.append(
                ChapterRecord(
                    chunk_index=int(entry.get("chunk_index", len(records))),
                    start=float(entry.get("start", 0.0)),
                    end=float(entry.get("end", 0.0)),
                    transcript=entry.get("transcript", ""),
                    transcript_segments=entry.get("transcript_segments", []),
                    frame_paths=[str(path) for path in entry.get("frame_paths", [])],
                    raw_chapter=chapter,
                )
            )
        records.sort(key=lambda rec: rec.chunk_index)
        return records

    async def _enrich_sequentially(
        self,
        records: List[ChapterRecord],
        *,
        context_window: int,
        llm_request_options: Dict[str, Any],
    ) -> tuple[List[ChapterCreationResponse], str, List[str]]:
        history: List[ChapterCreationResponse] = []
        enriched: List[ChapterCreationResponse] = []
        options = dict(llm_request_options)
        summary_sections: List[str] = []

        for record in records:
            messages = self._build_messages(
                record,
                previous_context=history[-context_window:] if context_window > 0 else [],
                global_summary_so_far="\n\n".join(summary_sections) or "No summary captured yet.",
            )
            logger.info(
                "[{}] Enriching chunk {} with {} prior chapters",
                self.step_id,
                record.chunk_index,
                len(history[-context_window:]) if context_window > 0 else 0,
            )
            raw = await llm_provider.chat_completion(
                messages,
                response_format=ContextEnrichmentResponse,
                **options,
            )
            response = self._coerce_response(raw)
            enriched.append(response.chapter)
            history.append(response.chapter)
            if response.global_summary_update:
                summary_sections.append(response.global_summary_update.strip())

        global_summary = "\n\n".join(summary_sections).strip()
        return enriched, global_summary, summary_sections

    def _build_messages(
        self,
        record: ChapterRecord,
        *,
        previous_context: List[ChapterCreationResponse],
        global_summary_so_far: str,
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
            "Video Summary So Far:\n"
            f"{global_summary_so_far}\n\n"
            "Instructions:\n"
            "- Produce a ContextEnrichmentResponse containing (a) a fully updated ChapterCreationResponse and (b) a succinct global_summary_update paragraph that captures any new storyline, procedural steps, materials, or outcomes introduced in this chunk.\n"
            "- Maintain factual grounding; do not invent events that contradict the current transcript.\n"
            "- If context does not add value, keep the summary focused but explain continuity if possible.\n"
            "- For procedural videos, enumerate steps, tools, measurements, and results. For stories, include key characters, motivations, and plot progress.\n"
            "- Append only new information to the global summary update; avoid repeating prior context."
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

    def _coerce_response(self, payload: Any) -> ContextEnrichmentResponse:
        content: Any = payload
        if isinstance(payload, dict) and "content" in payload:
            content = payload["content"]

        if isinstance(content, ContextEnrichmentResponse):
            return content

        if isinstance(content, BaseModel):
            return ContextEnrichmentResponse.model_validate(content.model_dump())

        if isinstance(content, dict):
            return ContextEnrichmentResponse.model_validate(content)

        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as err:
                raise ValueError("LLM provider returned non-JSON string content") from err
            return ContextEnrichmentResponse.model_validate(parsed)

        raise TypeError(f"Unsupported enrichment response type: {type(payload)!r}")

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0.0, value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
