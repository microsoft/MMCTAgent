"""Scene-detection-driven video chunking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from loguru import logger
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


@dataclass
class SceneChunk:
    index: int
    start: float
    end: float
    transcript: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _build_detector(params: Dict[str, Any], frame_rate: float):
    detector_type = str(params.get("detector", "content")).lower()
    min_scene_len_seconds = float(params.get("min_scene_length", 2.0))
    min_scene_len_frames = max(1, int(frame_rate * min_scene_len_seconds))

    if detector_type == "adaptive":
        threshold = float(params.get("threshold", 3.0))
        return AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len_frames,
        )

    threshold = float(params.get("threshold", 30.0))
    return ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len_frames,
    )


def _detect_scenes(video_path: Path, params: Dict[str, Any]) -> List[SceneChunk]:
    video_manager = VideoManager([str(video_path)])
    downscale = int(params.get("downscale", 2))
    if downscale > 1:
        video_manager.set_downscale_factor(downscale)

    scene_manager = SceneManager()
    detector = _build_detector(params, video_manager.get_base_timecode().framerate)
    scene_manager.add_detector(detector)

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        base_timecode = video_manager.get_base_timecode()
        detected = scene_manager.get_scene_list(base_timecode)
    finally:
        video_manager.release()

    chunks: List[SceneChunk] = []
    for idx, (start_tc, end_tc) in enumerate(detected):
        chunks.append(
            SceneChunk(
                index=idx,
                start=start_tc.get_seconds(),
                end=end_tc.get_seconds(),
            )
        )
    return chunks


def _enforce_max_scene_length(
    chunks: List[SceneChunk],
    max_length: float,
    video_end: Optional[float] = None,
) -> List[SceneChunk]:
    if max_length <= 0.0:
        return chunks

    bounded: List[SceneChunk] = []
    for chunk in chunks:
        start = chunk.start
        end = chunk.end if video_end is None else min(chunk.end, video_end)

        while start < end:
            split_end = min(end, start + max_length)
            if split_end <= start:
                break
            bounded.append(
                SceneChunk(
                    index=len(bounded),
                    start=start,
                    end=split_end,
                )
            )
            start = split_end

    return bounded


def _apply_chunk_overlap(
    chunks: List[SceneChunk],
    overlap_seconds: float,
    video_end: Optional[float] = None,
) -> List[SceneChunk]:
    if overlap_seconds <= 0.0 or not chunks:
        return chunks

    overlapped: List[SceneChunk] = []
    for chunk in chunks:
        start = max(0.0, chunk.start - overlap_seconds)
        end = chunk.end + overlap_seconds
        if video_end is not None:
            end = min(end, video_end)

        if end <= start:
            # If the chunk collapsed due to clamping, fall back to the original span.
            start = chunk.start
            end = chunk.end

        overlapped.append(
            SceneChunk(
                index=chunk.index,
                start=start,
                end=end,
            )
        )

    return overlapped


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _advance_past_prior_segments(
    segments: List[TranscriptSegment],
    start_index: int,
    chunk_start: float,
    tolerance: float,
) -> int:
    idx = start_index
    while idx < len(segments) and segments[idx].end <= chunk_start + tolerance:
        idx += 1
    return idx


def _parse_transcript_simple(srt_text: str) -> List[TranscriptSegment]:
    """Parses SRT transcript into list of TranscriptSegment objects."""
    segments = []
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\n|\Z)", re.DOTALL
    )

    def parse_ts(ts_str):
        h, m, s, ms = map(int, re.split("[:,]", ts_str))
        return h * 3600 + m * 60 + s + ms / 1000.0

    for match in pattern.finditer(srt_text):
        start_str, end_str, text = match.groups()
        segments.append(
            TranscriptSegment(
                start=parse_ts(start_str),
                end=parse_ts(end_str),
                text=text.replace("\n", " ").strip(),
            )
        )
    return segments


def _align_to_transcript(scenes: List[SceneChunk], transcript: str) -> List[SceneChunk]:
    """Adjusts scene boundaries to snap to transcript segment spans."""
    if not transcript:
        return scenes

    segments = _parse_transcript_simple(transcript)
    if not segments:
        return scenes

    allowed_scenes: List[SceneChunk] = []

    tolerance = 1e-3
    segment_idx = 0
    last_end = 0.0

    for i, scene in enumerate(scenes):
        chunk_start = max(scene.start, last_end)
        chunk_end = max(scene.end, chunk_start)

        segment_idx = _advance_past_prior_segments(segments, segment_idx, chunk_start, tolerance)

        chunk_segments: List[TranscriptSegment] = []
        scan_idx = segment_idx
        is_last_chunk = i == len(scenes) - 1

        if is_last_chunk:
            # Last chunk grabs everything remaining
            if scan_idx < len(segments):
                chunk_segments = segments[scan_idx:]
                scan_idx = len(segments)
        else:
            # Collect segments falling within this chunk
            while scan_idx < len(segments):
                segment = segments[scan_idx]

                # Stop if segment starts after chunk end (with tolerance)
                # AND we already have some segments.
                # If we don't have segments yet, we might grab one that slightly overlaps?
                # The logic from snippet:
                if segment.start >= chunk_end - tolerance and chunk_segments:
                    break
                if segment.start >= chunk_end - tolerance and not chunk_segments:
                    # If it's too far, maybe don't grab it? Snippet breaks here.
                    break

                chunk_segments.append(segment)
                scan_idx += 1

                # If this segment ends after or at chunk end, stop collecting
                if segment.end >= chunk_end - tolerance:
                    break

        if chunk_segments:
            first_seg = chunk_segments[0]
            last_seg = chunk_segments[-1]

            candidate_start = (
                min(chunk_start, first_seg.start) if first_seg.start <= chunk_start else chunk_start
            )
            adjusted_start = max(last_end, candidate_start)

            adjusted_end = chunk_end
            if adjusted_end < last_seg.end - tolerance:
                adjusted_end = last_seg.end

            segment_idx = scan_idx
        else:
            # No matching segments
            adjusted_start = chunk_start
            adjusted_end = chunk_end
            segment_idx = scan_idx

        text_blob = " ".join(seg.text.strip() for seg in chunk_segments).strip()

        aligned_scenes.append(
            SceneChunk(
                index=scene.index, start=adjusted_start, end=adjusted_end, transcript=text_blob
            )
        )
        last_end = adjusted_end

    return aligned_scenes


class SceneChunker:
    """
    Scene-aware chunking that emits per-scene spans with optional transcript alignment.
    Does NOT split physical video files.
    """

    def __init__(self, video_path: Path, params: Dict[str, Any], transcript: Optional[str] = None):
        self.video_path = video_path
        self.params = params
        self.transcript = transcript

    async def run(self, context) -> List[Dict[str, Any]]:
        # 1. Detect Scenes
        try:
            loop = asyncio.get_running_loop()
            chunks: List[SceneChunk] = await loop.run_in_executor(
                None, _detect_scenes, self.video_path, self.params
            )
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            chunks = []

        if not chunks:
            # fallback to single chunk covering entire video duration
            duration = context.video_duration or 0.0
            chunks = [SceneChunk(index=0, start=0.0, end=duration)]

        # 2. Enforce Max Length
        max_scenes = int(self.params.get("max_scenes", 1000))
        min_scene_len_seconds = float(self.params.get("min_scene_length", 2.0))
        max_scene_length = float(self.params.get("max_scene_length", min_scene_len_seconds + 5.0))
        overlap_seconds = float(self.params.get("overlap_seconds", 0.0))

        if max_scene_length > 0.0:
            chunks = _enforce_max_scene_length(
                chunks,
                max_scene_length,
                context.video_duration,
            )

        # 3. Apply Overlap
        chunks = _apply_chunk_overlap(
            chunks,
            overlap_seconds,
            context.video_duration,
        )

        # 4. Align with Transcript (Critical for Step.py)
        if self.transcript:
            chunks = _align_to_transcript(chunks, self.transcript)
            logger.info("Aligned scenes with transcript - boundaries extended.")

        # Re-index
        for idx, chunk in enumerate(chunks):
            chunk.index = idx

        chunks = chunks[:max_scenes]

        # Serialize for output
        chunks_metadata = []
        for chunk in chunks:
            chunks_metadata.append(
                {
                    "chunk_id": chunk.index,
                    "start_time": chunk.start,
                    "end_time": chunk.end,
                    "transcript": chunk.transcript,
                }
            )

        logger.info(f"Created {len(chunks_metadata)} chunk metadata entries (no video splitting)")
        return chunks_metadata
