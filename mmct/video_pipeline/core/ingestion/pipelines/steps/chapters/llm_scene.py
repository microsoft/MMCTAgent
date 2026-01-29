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

from mmct.providers.base.llm_provider import BaseLLMProvider
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse
from mmct.video_pipeline.utils.helper import get_media_folder


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


class SceneLLMChapterGenerator:
    """Generates structured chapters by fusing chunk transcripts with frame evidence."""

    description = (
        "LLM-backed chunk explainer that batches up to five frame/transcript pairs at a time."
    )

    def __init__(self, step_id="ingestion.chapters.scene-llm"):
        self.step_id = step_id

    async def run_direct(
        self,
        chunks: List[Dict[str, Any]],
        llm_provider: BaseLLMProvider,
        frames: List[Dict[str, Any]] = [],
        video_id: Optional[str] = None,
        params: Dict[str, Any] = {},
    ) -> List[Dict[str, Any]]:  # Returns list of chapter dicts

        val = params.get("max_frames_per_chapter")
        max_frames_per_chapter = int(val) if val is not None else -1
        print(f"DEBUG: max_frames_per_chapter={max_frames_per_chapter}")
        batch_size = int(params.get("batch_size", 5))
        max_parallel_requests = int(params.get("max_parallel_requests", batch_size))
        llm_request_options: Dict[str, Any] = dict(params.get("llm_request_options", {}) or {})
        collect_object_collection = bool(params.get("collect_object_collection", True))

        keyframe_metadata = []
        media_dir = None

        # If frames not provided but video_id is, load metadata to be processed per-chunk
        if not frames and video_id:
            keyframe_metadata = await self._load_keyframe_metadata(video_id)
            media_dir = Path(await get_media_folder())
            if not keyframe_metadata:
                logger.warning(f"[{self.step_id}] No frames found for video_id {video_id}")

        work_items = self._prepare_work_items(
            raw_chunks=chunks,
            raw_frames=frames,  # Passed if available
            keyframe_metadata=keyframe_metadata,
            video_id=video_id,
            media_dir=media_dir,
            max_frames=max_frames_per_chapter,
        )
        if not work_items:
            logger.warning(
                f"Step '{self.step_id}' could not locate aligned chunks and frames to process."
            )
            return []

        logger.info(
            "[{}] Generating {} chapters in batches of {} (max parallel {})",
            self.step_id,
            len(work_items),
            batch_size,
            max_parallel_requests,
        )

        chapter_results = await self._generate_batches(
            work_items,
            llm_provider=llm_provider,
            batch_size=batch_size,
            max_parallel_requests=max_parallel_requests,
            llm_request_options=llm_request_options,
            collect_object_collection=collect_object_collection,
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

        return produced_chapters

    async def _load_keyframe_metadata(self, video_id: str) -> List[Dict[str, Any]]:
        """Load keyframe metadata from JSON."""
        try:
            base_dir = await get_media_folder()
            keyframes_dir = Path(base_dir) / "keyframes" / video_id
            json_file_path = keyframes_dir / f"keyframe_metadata_{video_id}.json"

            if not json_file_path.exists():
                logger.warning(f"Keyframe metadata JSON not found: {json_file_path}")
                return []

            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("keyframes", [])
        except Exception as e:
            logger.error(f"Failed to load keyframe metadata: {e}")
            return []

    def _prepare_work_items(
        self,
        *,
        raw_chunks: List[Dict[str, Any]],
        raw_frames: List[Dict[str, Any]] = [],
        keyframe_metadata: List[Dict[str, Any]] = [],
        video_id: Optional[str] = None,
        media_dir: Optional[Path] = None,
        max_frames: int,
    ) -> List[ChunkWorkItem]:

        if not raw_chunks:
            return []

        # Strategy: Use raw_frames (pre-resolved) if present, else use metadata + video_id
        use_metadata = bool(keyframe_metadata and video_id and media_dir)

        # Sort frames if using pre-resolved list
        if not use_metadata and raw_frames:
            # Normalize raw_frames
            normalized_frames = []
            for f in raw_frames:
                p = f.get("path") or f.get("file_path")
                t = f.get("timestamp") or f.get("time_offset")
                if p:
                    normalized_frames.append(
                        FrameSample(Path(p), float(t) if t is not None else None)
                    )
            normalized_frames.sort(key=lambda x: x.timestamp or 0.0)
            raw_frames_samples = normalized_frames
        else:
            raw_frames_samples = []

        items: List[ChunkWorkItem] = []

        for chunk in raw_chunks:
            chunk_start = float(chunk.get("start", 0.0))
            chunk_end = float(chunk.get("end", 0.0))
            chunk_index = int(chunk.get("index", len(items)))

            transcript_info = chunk.get("transcript", {}) or {}
            transcript_text = transcript_info.get("text")
            if not transcript_text:
                transcript_text = chunk.get("sentence", "")

            # Resolve frames for this chunk
            chunk_frames: List[FrameSample] = []

            if use_metadata:
                # Filter metadata by timestamp
                # Optimization: Metadata could be sorted? Assuming it is or we sort it once.
                # Assuming keyframe_metadata is passed sorted or we scan.
                # Scan is okay for reasonable N.

                # We need to construct path here.
                keyframes_dir = media_dir / "keyframes" / video_id

                for kf in keyframe_metadata:
                    ts = kf.get("timestamp_seconds")
                    if ts is None:
                        continue
                    if chunk_start <= ts < chunk_end:
                        fname = kf.get("keyframe_filename")
                        # Optimistic path construction
                        # Trying standard pattern first
                        fpath = keyframes_dir / fname

                        # Verify existence only for matches?
                        # Or trust it exists if metadata says so?
                        # ChapterGenerator checks existence. We should probably too.
                        if fpath.exists():
                            chunk_frames.append(FrameSample(fpath, ts))
                        else:
                            # Try prefixed fallback
                            if "frame_" in fname:
                                suffix = fname.split("frame_")[-1]
                                prefix_path = (
                                    keyframes_dir / f"{video_id}_{suffix}"
                                )  # legacy fallback?
                                if prefix_path.exists():
                                    chunk_frames.append(FrameSample(prefix_path, ts))
                                else:
                                    # Try just prefixed with original name
                                    p_path = keyframes_dir / f"{video_id}_{fname}"
                                    if p_path.exists():
                                        chunk_frames.append(FrameSample(p_path, ts))
            else:
                # Use pre-resolved raw_frames_samples
                chunk_frames = [
                    f
                    for f in raw_frames_samples
                    if f.timestamp is not None and chunk_start <= f.timestamp < chunk_end
                ]

            if max_frames > 0:
                chunk_frames = chunk_frames[:max_frames]

            items.append(
                ChunkWorkItem(
                    index=chunk_index,
                    start=chunk_start,
                    end=chunk_end,
                    transcript_text=transcript_text or "",
                    transcript_segments=chunk.get("transcript_segments") or [],
                    frames=chunk_frames,
                )
            )

        items.sort(key=lambda item: item.index)
        return items

    async def _generate_batches(
        self,
        work_items: List[ChunkWorkItem],
        *,
        llm_provider: BaseLLMProvider,
        batch_size: int,
        max_parallel_requests: int,
        llm_request_options: Dict[str, Any],
        collect_object_collection: bool,
    ) -> List[tuple[ChunkWorkItem, ChapterCreationResponse]]:
        semaphore = asyncio.Semaphore(max(1, max_parallel_requests))
        results: List[tuple[ChunkWorkItem, ChapterCreationResponse]] = []
        request_options = dict(llm_request_options)

        async def invoke(
            item: ChunkWorkItem,
        ) -> Optional[tuple[ChunkWorkItem, ChapterCreationResponse]]:
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
            system_prompt += " Track actions and physical objects with detailed appearance/identity. Capture visible text primarily for the 'text_from_scene' field, NOT as objects."
        else:
            system_prompt += (
                " Focus on the narrative summary and actions; objects SHOULD be omitted."
            )

        frame_timeline = self._format_frame_timeline(item.frames)
        object_instruction = (
            "- Populate object_collection ONLY with physical entities (people, animals, distinct objects). EXCLUDE on-screen text, subtitles, and generic background elements (e.g. 'background', 'wall', 'sky') unless critical to the action."
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
