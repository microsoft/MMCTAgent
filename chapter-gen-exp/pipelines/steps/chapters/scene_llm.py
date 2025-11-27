"""LLM-backed chapter generation driven by scene chunks and frames."""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger
from pydantic import BaseModel

from providers.base.llm_provider import LLMProvider
from providers.factory import provider_factory

from .models import ChapterCreationResponse

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step

llm_provider = provider_factory.create_llm_provider()
emb_provider = provider_factory.create_embedding_provider()

@dataclass
class FrameSample:
    """Lightweight holder describing a single frame destined for the LLM payload."""

    path: Path
    timestamp: Optional[float]


@dataclass
class ChunkWorkItem:
    """Aggregated transcription + frame context for a single chunk."""

    index: int
    start: float
    end: float
    transcript_text: str
    transcript_segments: List[Dict[str, Any]]
    frames: List[FrameSample]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

@register_step("chapters.scene-llm")
class SceneLLMChapterGenerationStep(PipelineStep):
    """Generates structured chapters by fusing chunk transcripts with frame evidence."""

    description = (
        "LLM-backed chunk explainer that batches up to five frame/transcript pairs at a time."
    )

    def run(self, context: StepContext) -> StepResult:
        chunks_step = self.params.get("chunks_step")
        frames_step = self.params.get("frames_step")
        if not chunks_step or not frames_step:
            raise ValueError("'chunks_step' and 'frames_step' parameters are required")

        max_frames_per_chapter = int(self.params.get("max_frames_per_chapter", 12))
        batch_size = int(self.params.get("batch_size", 5))
        max_parallel_requests = int(self.params.get("max_parallel_requests", batch_size))
        llm_request_options: Dict[str, Any] = dict(self.params.get("llm_request_options", {}) or {})
        collect_object_collection = bool(self.params.get("collect_object_collection", True))

        work_items = self._prepare_work_items(
            context,
            chunks_step=chunks_step,
            frames_step=frames_step,
            max_frames=max_frames_per_chapter,
        )
        if not work_items:
            raise ValueError(
                f"Step '{self.step_id}' could not locate aligned chunks and frames to process."
            )

        logger.info(
            "[{}] Generating {} chapters in batches of {} (max parallel {})",
            self.step_id,
            len(work_items),
            batch_size,
            max_parallel_requests,
        )

        chapter_results = asyncio.run(
            self._generate_batches(
                work_items,
                llm_provider=llm_provider,
                batch_size=batch_size,
                max_parallel_requests=max_parallel_requests,
                llm_request_options=llm_request_options,
                collect_object_collection=collect_object_collection,
            )
        )

        produced_chapters: List[Dict[str, Any]] = []
        for item, response in chapter_results:
            produced_chapters.append(
                {
                    "chunk_index": item.index,
                    "start": item.start,
                    "end": item.end,
                    "duration": item.duration,
                    "transcript": item.transcript_text,
                    "transcript_segments": item.transcript_segments,
                    "frame_paths": [str(frame.path) for frame in item.frames],
                    "chapter": response.model_dump(),
                }
            )

        metrics = {
            "chapters_emitted": float(len(produced_chapters)),
            "chunks_processed": float(len(work_items)),
            "avg_frames_per_chunk": float(
                sum(len(item.frames) for item in work_items) / max(1, len(work_items))
            ),
        }

        return StepResult(
            step_id=self.step_id,
            produced={"chapters": produced_chapters},
            metrics=metrics,
        )

    def _prepare_work_items(
        self,
        context: StepContext,
        *,
        chunks_step: str,
        frames_step: str,
        max_frames: int,
    ) -> List[ChunkWorkItem]:
        chunk_payload = context.data_store.get(chunks_step, {})
        frame_payload = context.data_store.get(frames_step, {})
        raw_chunks: List[Dict[str, Any]] = chunk_payload.get("chunks", [])
        raw_frames: List[Dict[str, Any]] = frame_payload.get("frames", [])
        if not raw_chunks:
            return []

        frames_by_chunk: Dict[int, List[FrameSample]] = {}
        for frame in raw_frames:
            chunk_index = int(frame.get("chunk_index", -1))
            if chunk_index < 0:
                continue
            timestamp = frame.get("timestamp")
            path = frame.get("path")
            if not path:
                continue
            sample = FrameSample(path=Path(path), timestamp=float(timestamp) if timestamp is not None else None)
            frames_by_chunk.setdefault(chunk_index, []).append(sample)

        items: List[ChunkWorkItem] = []
        for chunk in raw_chunks:
            chunk_index = int(chunk.get("index", len(items)))
            transcript_info = chunk.get("transcript", {}) or {}
            transcript_text = transcript_info.get("text")
            if not transcript_text:
                segments = transcript_info.get("segments") or chunk.get("segments") or []
                transcript_text = " ".join(seg.get("text", "") for seg in segments).strip()
            frames = sorted(frames_by_chunk.get(chunk_index, []), key=lambda item: item.timestamp or 0.0)
            if max_frames > 0:
                frames = frames[:max_frames]
            items.append(
                ChunkWorkItem(
                    index=chunk_index,
                    start=float(chunk.get("start", 0.0)),
                    end=float(chunk.get("end", 0.0)),
                    transcript_text=transcript_text or "",
                    transcript_segments=transcript_info.get("segments") or [],
                    frames=frames,
                )
            )
        items.sort(key=lambda item: item.index)
        return items

    async def _generate_batches(
        self,
        work_items: List[ChunkWorkItem],
        *,
        llm_provider: LLMProvider,
        batch_size: int,
        max_parallel_requests: int,
        llm_request_options: Dict[str, Any],
        collect_object_collection: bool,
    ) -> List[tuple[ChunkWorkItem, ChapterCreationResponse]]:
        semaphore = asyncio.Semaphore(max(1, max_parallel_requests))
        results: List[tuple[ChunkWorkItem, ChapterCreationResponse]] = []
        request_options = dict(llm_request_options)

        async def invoke(item: ChunkWorkItem) -> Optional[tuple[ChunkWorkItem, ChapterCreationResponse]]:
            try:
                async with semaphore:
                    messages = self._build_messages(
                        item,
                        collect_object_collection=collect_object_collection,
                    )
                    logger.info(
                        "[{}] Chunk {} -> dispatching {} frames covering {:.2f}s",
                        self.step_id,
                        item.index,
                        len(item.frames),
                        item.duration,
                    )
                    raw_response = await llm_provider.chat_completion(
                        messages,
                        response_format=ChapterCreationResponse,
                        **request_options,
                    )
                    response = self._ensure_chapter_response(raw_response)
                    logger.info(
                        "[{}] Chunk {} -> chapter generation finished",
                        self.step_id,
                        item.index,
                    )
                    return item, response
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Chapter generation failed for chunk %s: %s", item.index, exc)
                return None

        for batch in _chunk_iter(work_items, batch_size):
            batch_results = await asyncio.gather(*(invoke(item) for item in batch))
            for result in batch_results:
                if result is not None:
                    results.append(result)

        results.sort(key=lambda pair: pair[0].index)
        return results

    def _ensure_chapter_response(self, payload: Any) -> ChapterCreationResponse:
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

        raise TypeError(f"Unsupported chapter response type: {type(payload)!r}")

    def _build_messages(
        self,
        item: ChunkWorkItem,
        *,
        collect_object_collection: bool,
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a VideoAnalyzerGPT. Analyze the provided frames and transcript snippet to "
            "produce an exhaustive ChapterCreationResponse describing everything in English."
        )
        if collect_object_collection:
            system_prompt += (
                " Track actions, visible text, and every object with detailed appearance/identity."
            )
        else:
            system_prompt += " Focus on the narrative summary and actions; objects SHOULD be omitted."

        frame_timeline = self._format_frame_timeline(item.frames)
        object_instruction = (
            "- Populate object_collection with every identifiable entity, including people, objects, text, and background items."
            if collect_object_collection
            else "- Set object_collection to an empty list or null if no structured tracking is needed."
        )

        transcript_block = (
            f"Chunk Index: {item.index}\n"
            f"Start: {self._format_seconds(item.start)} ({item.start:.2f}s)\n"
            f"End: {self._format_seconds(item.end)} ({item.end:.2f}s)\n"
            f"Duration: {item.duration:.2f}s\n\n"
            "Transcript (never break sentences; already aligned):\n"
            "<<<TRANSCRIPT>>>\n"
            f"{item.transcript_text}\n"
            "<<<END_TRANSCRIPT>>>\n\n"
            "Frame timeline:\n"
            f"{frame_timeline or 'No frames captured for this chunk.'}\n\n"
            "Output requirements:\n"
            "- Return valid JSON matching ChapterCreationResponse fields.\n"
            "- Translate any non-English content to English.\n"
            "- Write detailed_summary directly about the scene (no phrases like 'In this video' or 'In this segment').\n"
            "- Ensure detailed_summary integrates every observable detail from the frames, even if not stated in the transcript.\n"
            f"{object_instruction}"
        )

        user_blocks: List[Dict[str, Any]] = []
        for encoded in self._encode_frames(item.frames):
            user_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                }
            )
        user_blocks.append({"type": "text", "text": transcript_block})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_blocks},
        ]

    def _encode_frames(self, frames: Iterable[FrameSample]) -> List[str]:
        encoded: List[str] = []
        for frame in frames:
            try:
                data = frame.path.read_bytes()
                encoded.append(base64.b64encode(data).decode("utf-8"))
            except OSError:
                logger.warning("Unable to read frame %s; skipping", frame.path)
        return encoded

    def _format_frame_timeline(self, frames: List[FrameSample]) -> str:
        if not frames:
            return ""
        lines = []
        for idx, frame in enumerate(frames, start=1):
            ts = frame.timestamp
            if ts is None:
                lines.append(f"Frame {idx}: timestamp unknown -> {frame.path.name}")
            else:
                lines.append(
                    f"Frame {idx}: {self._format_seconds(ts)} ({ts:.2f}s) -> {frame.path.name}"
                )
        return "\n".join(lines)

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0.0, value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _chunk_iter(sequence: Sequence[ChunkWorkItem], size: int) -> Iterable[List[ChunkWorkItem]]:
    if size <= 0:
        size = 1
    for idx in range(0, len(sequence), size):
        yield sequence[idx : idx + size]