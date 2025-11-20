"""Align chunk boundaries to transcript sentence spans."""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from ...utils.chunks import resolve_chunks
from ...utils.transcript_loader import TranscriptSegment, load_transcript


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


@register_step("video.chunk.align-transcript")
class TranscriptAlignedChunkStep(PipelineStep):
    """Adjusts chunk windows so they never split transcript sentences."""

    description = "Snaps chunk boundaries to transcript segment spans and attaches text snippets."

    def run(self, context: StepContext) -> StepResult:
        chunks_step = self.params.get("chunks_step")
        if not chunks_step:
            raise ValueError("Parameter 'chunks_step' is required for transcript alignment.")

        tolerance = float(self.params.get("tolerance", 1e-3))
        transcript = load_transcript(context.transcript_path)
        transcript_duration = transcript.duration
        fallback_duration = context.video_duration_seconds or transcript_duration
        base_chunks = resolve_chunks(context.data_store, chunks_step, fallback_duration)

        segments = transcript.segments
        segment_idx = 0
        last_end = 0.0
        aligned_chunks: List[Dict[str, Any]] = []
        attached_segments = 0

        for chunk_pos, chunk in enumerate(base_chunks):
            raw_start = float(chunk.get("start", 0.0))
            raw_end = float(chunk.get("end", fallback_duration))
            chunk_start = max(raw_start, last_end)
            chunk_end = max(raw_end, chunk_start)

            segment_idx = _advance_past_prior_segments(segments, segment_idx, chunk_start, tolerance)

            chunk_segments: List[TranscriptSegment] = []
            scan_idx = segment_idx
            is_last_chunk = chunk_pos == len(base_chunks) - 1

            if is_last_chunk:
                if scan_idx < len(segments):
                    chunk_segments = segments[scan_idx:]
                    scan_idx = len(segments)
            else:
                while scan_idx < len(segments):
                    segment = segments[scan_idx]
                    if segment.start >= chunk_end - tolerance and chunk_segments:
                        break
                    if segment.start >= chunk_end - tolerance and not chunk_segments:
                        break
                    chunk_segments.append(segment)
                    scan_idx += 1
                    if segment.end >= chunk_end - tolerance:
                        break

            if chunk_segments:
                first_seg = chunk_segments[0]
                last_seg = chunk_segments[-1]
                candidate_start = min(chunk_start, first_seg.start) if first_seg.start <= chunk_start else chunk_start
                adjusted_start = max(last_end, candidate_start)
                adjusted_end = chunk_end
                if adjusted_end < last_seg.end - tolerance:
                    adjusted_end = last_seg.end
                attached_segments += len(chunk_segments)
                segment_idx = scan_idx
            else:
                adjusted_start = chunk_start
                adjusted_end = chunk_end
                segment_idx = scan_idx

            text_blob = " ".join(seg.text.strip() for seg in chunk_segments).strip()

            aligned_chunks.append(
                {
                    "index": int(chunk.get("index", chunk_pos)),
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "duration": max(0.0, adjusted_end - adjusted_start),
                    "transcript": {
                        "text": text_blob,
                        "segment_count": len(chunk_segments),
                        "segments": [
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                            }
                            for seg in chunk_segments
                        ],
                    },
                    "source_start": raw_start,
                    "source_end": raw_end,
                }
            )
            last_end = adjusted_end

        coverage = attached_segments / len(segments) if segments else 0.0
        metrics = {
            "chunks_aligned": float(len(aligned_chunks)),
            "segments_attached": float(attached_segments),
            "segment_coverage_ratio": float(coverage),
        }

        return StepResult(
            step_id=self.step_id,
            produced={"chunks": aligned_chunks},
            metrics=metrics,
        )
