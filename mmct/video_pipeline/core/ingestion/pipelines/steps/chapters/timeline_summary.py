"""Timestamped global summary generator built on enriched chapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from mmct.providers.base.llm_provider import BaseLLMProvider

from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse
from ..base import PipelineStep, StepContext, StepResult

DEFAULT_WINDOW_MINUTES = 10.0
DEFAULT_OVERLAP_MINUTES = 2.0
DEFAULT_TOKEN_BUDGET = 4000
MIN_SECTION_TOKEN_BUDGET = 64


@dataclass
class SummaryConfig:
    """Normalized configuration for chapter window summarization."""

    chapters_step: str
    window_seconds: float
    overlap_seconds: float
    target_token_budget: int
    llm_request_options: Dict[str, Any]


@dataclass
class ChapterSlice:
    """Lightweight view of a chapter needed for summary windows."""

    chunk_index: int
    start: float
    end: float
    summary: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ChapterWindow:
    """Represents a contiguous, possibly overlapping, slice of the video."""

    index: int
    start: float
    end: float
    chapters: List[ChapterSlice]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class WindowSummaryResponse(BaseModel):
    """Structured response for a single window summary."""

    summary_text: str = Field(
        ...,
        description="Narrative summary highlighting the window's key developments",
    )


class ChapterTimelineSummarizer:  # Removed PipelineStep inheritance for direct usage in wrapper
    """Creates an overlapping, timestamped global summary from detailed chapters."""

    description = "Aggregates chapter summaries into ~10 minute windows and crafts a 4k-token global narrative."

    def __init__(self, step_id="ingestion.chapters.timeline-summary", log_prefix=""):
        self.step_id = step_id
        self.log_prefix = log_prefix

    async def run_direct(
        self,
        chapters: List[Dict[str, Any]],
        llm_provider: BaseLLMProvider,
        params: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Modified run method for direct invocation.
        """
        config = self._build_config(params)
        slices = self._load_chapters_direct({"chapters": chapters})

        if not slices:
            logger.warning(f"{self.log_prefix} No chapters found for summary.")
            return {}

        windows = self._build_windows(slices, config)
        if not windows:
            logger.warning(
                f"{self.log_prefix} Unable to create windows for global summary generation."
            )
            return {}

        per_window_budget = max(
            MIN_SECTION_TOKEN_BUDGET,
            config.target_token_budget // max(1, len(windows)),
        )

        logger.info(
            "[{}] Summarizing {} windows (~{} tokens per window)",
            self.step_id,
            len(windows),
            per_window_budget,
        )

        window_summaries = await self._summarize_batch(
            windows,
            llm_provider=llm_provider,
            token_budget=per_window_budget,
            llm_request_options=config.llm_request_options,
        )

        sections: List[str] = []
        serialized_windows: List[Dict[str, Any]] = []
        for window, text in zip(windows, window_summaries):
            label = f"{self._format_seconds(window.start)} - {self._format_seconds(window.end)}"
            block = f"{label}: {text.strip()}"
            sections.append(block)
            serialized_windows.append(
                {
                    "window_index": window.index,
                    "start": window.start,
                    "end": window.end,
                    "duration": window.duration,
                    "chapter_count": len(window.chapters),
                    "summary": text.strip(),
                    "labeled_summary": block,
                }
            )

        original_sections = list(sections)
        sections = self._enforce_token_budget(sections, config.target_token_budget)
        trimmed = sections != original_sections

        global_summary = "\n\n".join(sections).strip()
        approx_tokens = self._estimate_tokens(global_summary)
        if trimmed:
            logger.info(
                "[{}] Global summary assembled (~{} tokens est, trimmed to budget)",
                self.step_id,
                approx_tokens,
            )
        else:
            logger.info(
                "[{}] Global summary assembled (~{} tokens est)",
                self.step_id,
                approx_tokens,
            )

        produced_payload = {
            "global_summary": global_summary,
            "global_summary_sections": sections,
            "windows": serialized_windows,
        }

        return produced_payload

    def _build_config(self, params: Dict[str, Any]) -> SummaryConfig:
        window_minutes = float(params.get("window_minutes", DEFAULT_WINDOW_MINUTES))
        if window_minutes <= 0:
            raise ValueError("'window_minutes' must be positive")
        overlap_minutes = float(params.get("window_overlap_minutes", DEFAULT_OVERLAP_MINUTES))
        window_seconds = window_minutes * 60.0
        overlap_seconds = max(0.0, min(window_seconds - 1.0, overlap_minutes * 60.0))

        token_budget = int(params.get("target_token_budget", DEFAULT_TOKEN_BUDGET))
        if token_budget <= 0:
            raise ValueError("'target_token_budget' must be positive")

        llm_request_options: Dict[str, Any] = dict(params.get("llm_request_options", {}) or {})

        return SummaryConfig(
            chapters_step="direct",  # Not used in direct mode
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            target_token_budget=token_budget,
            llm_request_options=llm_request_options,
        )

    def _load_chapters_direct(self, payload: Dict[str, Any]) -> List[ChapterSlice]:
        raw_chapters = payload.get("chapters") or []
        slices: List[ChapterSlice] = []
        for entry in raw_chapters:
            chapter_data = entry.get("chapter")
            if not chapter_data:
                continue
            chapter = ChapterCreationResponse.model_validate(chapter_data)
            summary = chapter.detailed_summary.strip()
            if not summary:
                continue
            chunk_index = int(entry.get("chunk_index", len(slices)))
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
            slices.append(
                ChapterSlice(
                    chunk_index=chunk_index,
                    start=start,
                    end=end,
                    summary=summary,
                )
            )

        slices.sort(key=lambda item: (item.start, item.chunk_index))
        return slices

    def _build_windows(
        self, slices: List[ChapterSlice], config: SummaryConfig
    ) -> List[ChapterWindow]:
        if not slices:
            return []

        timeline_start = min(item.start for item in slices)
        timeline_end = max(item.end for item in slices)
        stride = max(1.0, config.window_seconds - config.overlap_seconds)

        windows: List[ChapterWindow] = []
        cursor = max(0.0, timeline_start)
        index = 0
        while cursor < timeline_end:
            window_end = min(timeline_end, cursor + config.window_seconds)
            members = [
                item
                for item in slices
                if self._chapters_overlap(item.start, item.end, cursor, window_end)
            ]
            if members:
                windows.append(
                    ChapterWindow(
                        index=index,
                        start=cursor,
                        end=window_end,
                        chapters=members,
                    )
                )
                index += 1
            cursor += stride

        if not windows:
            # fallback: treat whole video as single window
            windows.append(
                ChapterWindow(
                    index=0,
                    start=timeline_start,
                    end=timeline_end,
                    chapters=slices,
                )
            )
        return windows

    async def _summarize_batch(
        self,
        windows: List[ChapterWindow],
        *,
        llm_provider: BaseLLMProvider,
        token_budget: int,
        llm_request_options: Dict[str, Any],
    ) -> List[str]:
        summaries: List[str] = []
        options = dict(llm_request_options)
        for window in windows:
            messages = self._build_messages(window, token_budget)
            logger.info(
                "[{}] Window {} | {:.0f}s span | {} chapters",
                self.step_id,
                window.index,
                window.duration,
                len(window.chapters),
            )
            raw = await llm_provider.chat_completion(
                messages,
                response_format=WindowSummaryResponse,
                **options,
            )
            response = self._coerce_response(raw)
            summaries.append(response.summary_text.strip())
        return summaries

    def _build_messages(self, window: ChapterWindow, token_budget: int) -> List[Dict[str, Any]]:
        chapter_lines = []
        for chapter in window.chapters:
            chapter_lines.append(
                f"Chunk {chapter.chunk_index} [{self._format_seconds(chapter.start)} - {self._format_seconds(chapter.end)}]: {chapter.summary}"
            )
        chapter_block = "\n".join(chapter_lines)
        instructions = (
            "Summarize the developments in this window, weaving together cause/effect, materials, and outcomes. "
            "Keep the prose flowing (no bullet lists) and stay factual."
        )
        user_prompt = (
            f"Window Start: {self._format_seconds(window.start)} ({window.start:.2f}s)\n"
            f"Window End: {self._format_seconds(window.end)} ({window.end:.2f}s)\n"
            f"Window Duration: {window.duration:.2f}s\n"
            f"Chapters Covered: {len(window.chapters)}\n\n"
            f"Chapter Summaries:\n{chapter_block}\n\n"
            "Instructions:\n"
            f"- {instructions}\n"
            f"- Limit the response to roughly {token_budget} tokens (about {token_budget * 4} characters).\n"
            "- Highlight transitions between scenes so the narrative reads like a timeline entry."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a timeline narrator ensuring continuity across overlapping video windows."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    def _coerce_response(self, payload: Any) -> WindowSummaryResponse:
        content: Any = payload
        if isinstance(payload, dict) and "content" in payload:
            content = payload["content"]
        if isinstance(content, WindowSummaryResponse):
            return content
        if isinstance(content, BaseModel):
            return WindowSummaryResponse.model_validate(content.model_dump())
        if isinstance(content, dict):
            return WindowSummaryResponse.model_validate(content)
        if isinstance(content, str):
            return WindowSummaryResponse.model_validate({"summary_text": content})
        raise TypeError(f"Unsupported summary response type: {type(payload)!r}")

    @staticmethod
    def _chapters_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
        return max(start_a, start_b) < min(end_a, end_b)

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0.0, value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _enforce_token_budget(self, sections: List[str], budget: int) -> List[str]:
        if budget <= 0 or not sections:
            return sections
        approx = self._estimate_tokens("\n\n".join(sections))
        if approx <= budget:
            return sections
        per_section_budget = max(16, budget // len(sections))
        trimmed_sections = [self._trim_to_tokens(block, per_section_budget) for block in sections]
        trimmed_total = self._estimate_tokens("\n\n".join(trimmed_sections))
        if trimmed_total <= budget:
            return trimmed_sections
        pruned = list(trimmed_sections)
        while pruned and self._estimate_tokens("\n\n".join(pruned)) > budget:
            pruned.pop()
        return pruned or trimmed_sections

    def _trim_to_tokens(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        words = text.split()
        if len(words) <= token_budget:
            return text
        trimmed = " ".join(words[:token_budget]).strip()
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "..."
        return trimmed

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        words = text.split()
        return max(1, int(len(words) * 1.3))
