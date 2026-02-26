"""Dense keyframe extraction with action boundary detection."""

import os
import cv2
import asyncio
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .boundary_detector import detect_action_boundaries, BoundaryThresholds
from mmct.video_pipeline.core.ingestion.models import ExtractionPlan


@dataclass
class DenseKeyframeConfig:
    """Configuration for dense keyframe extraction."""

    motion_threshold: float = 0.8  # Motion score threshold for keyframe selection
    sample_fps: int = 2  # Frames per second to analyze
    max_frame_width: int = 800  # Max width for motion analysis
    num_workers: int = 4  # Parallel workers for extraction
    min_keyframes_per_chunk: int = 4  # Minimum keyframes to extract per chunk
    diversity_threshold: float = 0.15  # Threshold for frame diversity


def _get_video_properties(video_path: str) -> Dict[str, Any]:
    """Get basic video properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    props = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    props["duration_seconds"] = props["frame_count"] / props["fps"] if props["fps"] > 0 else 0
    cap.release()
    return props


def _calculate_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Calculate optical flow motion score between frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=12,
        iterations=2,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def _calculate_frame_diversity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate visual diversity between two frames using histogram comparison."""
    # Convert to HSV for better color comparison
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # Normalize
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Compare using correlation (1 = identical, 0 = no correlation)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # Return diversity (1 - correlation)
    return 1.0 - max(0.0, correlation)


def _scale_factor(width: int, height: int, max_width: int) -> Tuple[float, int, int]:
    """Calculate scale factor to fit within max_width."""
    longest = max(width, height)
    if longest > max_width:
        scale = max_width / float(longest)
    else:
        scale = 1.0
    return scale, int(width * scale), int(height * scale)


def _extract_keyframes_from_video(
    video_path: str,
    start_time: float,
    end_time: float,
    target_frames: int,
    config: DenseKeyframeConfig,
    output_dir: str,
    chunk_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract keyframes from a video segment using motion-based selection.
    
    This function:
    1. Samples frames at configured FPS
    2. Computes motion scores using optical flow
    3. Selects frames with highest motion (action moments)
    4. Ensures temporal coverage and visual diversity
    5. Saves frames to disk and returns metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame range for this segment
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)

    if start_frame >= end_frame:
        cap.release()
        return []

    # Scale factor for motion analysis
    scale, scaled_w, scaled_h = _scale_factor(width, height, config.max_frame_width)

    # Sampling interval
    sample_interval = max(1, int(fps / config.sample_fps))

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # First pass: collect all candidate frames with motion scores
    candidates = []
    prev_gray = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Only process at sample interval
        if (frame_idx - start_frame) % sample_interval == 0:
            timestamp = frame_idx / fps

            # Downscale for motion analysis
            if scale < 1.0:
                small = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            else:
                small = frame_bgr

            curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Calculate motion score
            if prev_gray is not None:
                motion_score = _calculate_motion_score(prev_gray, curr_gray)
            else:
                motion_score = 0.0

            candidates.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "motion_score": motion_score,
                "frame_bgr": frame_bgr.copy(),  # Keep original resolution
            })

            prev_gray = curr_gray

        frame_idx += 1

    cap.release()

    if not candidates:
        return []

    # Second pass: select best keyframes
    selected = _select_diverse_keyframes(candidates, target_frames, config)

    # Save selected keyframes to disk
    os.makedirs(output_dir, exist_ok=True)
    keyframes = []

    for i, kf in enumerate(selected):
        filename = f"{chunk_id}_kf_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, kf["frame_bgr"])

        keyframes.append({
            "frame_index": i,
            "original_frame_idx": kf["frame_idx"],
            "timestamp": kf["timestamp"],
            "motion_score": kf["motion_score"],
            "filepath": filepath,
            "filename": filename,
            "image": kf["frame_bgr"],  # Keep for boundary detection
        })

    return keyframes


def _select_diverse_keyframes(
    candidates: List[Dict[str, Any]],
    target_count: int,
    config: DenseKeyframeConfig,
) -> List[Dict[str, Any]]:
    """
    Select diverse keyframes using a combination of motion scores and temporal coverage.
    
    Strategy:
    1. Always include first and last frames for context
    2. Sort remaining by motion score
    3. Greedily select high-motion frames that are visually diverse from already selected
    """
    if len(candidates) <= target_count:
        return candidates

    # Ensure minimum
    target_count = max(target_count, config.min_keyframes_per_chunk)

    selected = []
    selected_indices = set()

    # Always include first frame
    selected.append(candidates[0])
    selected_indices.add(0)

    # Always include last frame
    if len(candidates) > 1:
        selected.append(candidates[-1])
        selected_indices.add(len(candidates) - 1)

    # Sort remaining by motion score (descending)
    remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
    remaining_indices.sort(key=lambda i: candidates[i]["motion_score"], reverse=True)

    # Greedily select diverse frames
    for idx in remaining_indices:
        if len(selected) >= target_count:
            break

        candidate = candidates[idx]

        # Check diversity against already selected frames
        is_diverse = True
        for sel in selected:
            diversity = _calculate_frame_diversity(candidate["frame_bgr"], sel["frame_bgr"])
            if diversity < config.diversity_threshold:
                is_diverse = False
                break

        if is_diverse:
            selected.append(candidate)
            selected_indices.add(idx)

    # If we still need more frames, add highest motion without diversity check
    if len(selected) < target_count:
        for idx in remaining_indices:
            if idx not in selected_indices:
                selected.append(candidates[idx])
                if len(selected) >= target_count:
                    break

    # Sort by timestamp for sequential output
    selected.sort(key=lambda x: x["timestamp"])

    return selected


@register_step("ingestion.dense_keyframes")
class DenseKeyframeExtractionStep(PipelineStep):
    """
    Enhanced keyframe extraction with action boundary detection.
    
    Improvements over standard keyframe extraction:
    - Motion-based keyframe selection with optical flow
    - Visual diversity enforcement to avoid redundant frames
    - Action boundary detection at scene/action transitions
    - Configurable frames per chapter from extraction plan
    - Parallel processing for multiple chunks
    
    Params:
        source_step: Step ID to get video chunks from (default: "video_chunking")
        frames_per_chapter: Number of frames per chapter (overridden by extraction_plan)
        motion_threshold: Threshold for motion-based selection (default: 0.8)
        sample_fps: Frames per second to analyze (default: 2)
        diversity_threshold: Minimum visual diversity between selected frames (default: 0.15)
        num_workers: Number of parallel workers (default: 4)
        boundary_histogram_threshold: Threshold for histogram-based boundary detection (default: 0.4)
        boundary_motion_threshold: Threshold for motion-based boundary detection (default: 2.0)
    """

    step_type = "ingestion.dense_keyframes"
    description = "Extract keyframes with action boundary detection."

    async def run(self, context: StepContext) -> StepResult:
        """Execute dense keyframe extraction."""
        # Get extraction plan with proper typing
        extraction_plan: Optional[ExtractionPlan] = context.data_store.get(
            "extraction_planning", "extraction_plan"
        )
        frames_per_chapter: int = extraction_plan.frames_per_chapter if extraction_plan else 8

        # Override with param if specified
        frames_per_chapter = self.get_param("frames_per_chapter", context, default=frames_per_chapter)

        # Build config
        config = DenseKeyframeConfig(
            motion_threshold=self.get_param("motion_threshold", context, default=0.8),
            sample_fps=self.get_param("sample_fps", context, default=2),
            max_frame_width=self.get_param("max_frame_width", context, default=800),
            num_workers=self.get_param("num_workers", context, default=4),
            min_keyframes_per_chunk=self.get_param("min_keyframes_per_chunk", context, default=4),
            diversity_threshold=self.get_param("diversity_threshold", context, default=0.15),
        )

        # Boundary detection thresholds
        boundary_thresholds = BoundaryThresholds(
            histogram_threshold=self.get_param("boundary_histogram_threshold", context, default=0.4),
            edge_threshold=self.get_param("boundary_edge_threshold", context, default=0.35),
            motion_threshold=self.get_param("boundary_motion_threshold", context, default=2.0),
            color_threshold=self.get_param("boundary_color_threshold", context, default=0.3),
        )

        # Get video chunks with type hint
        source_step: str = self.get_param("source_step", context, default="video_chunking")
        video_chunks: List[Dict[str, Any]] = context.data_store.get(source_step, "video_chunks") or []

        if not video_chunks:
            context.logger.warning("No video chunks found, returning empty keyframes")
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "keyframes_per_chunk": [],
                    "action_boundaries": [],
                    "frames_per_chunk": frames_per_chapter,
                },
                metrics={
                    "total_keyframes": 0,
                    "action_boundaries_detected": 0,
                },
            )

        # Output directory for keyframes
        output_base: str = os.path.join(context.output_dir, "dense_keyframes")
        os.makedirs(output_base, exist_ok=True)

        # Process chunks in parallel
        all_keyframes: List[Dict[str, Any]] = []
        all_boundaries: List[Dict[str, Any]] = []
        all_artifacts: List[str] = []

        loop = asyncio.get_running_loop()
        workers: int = min(config.num_workers, len(video_chunks))

        async def process_chunk(chunk_idx: int, chunk: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single chunk."""
            chunk_id: str = chunk.get("chunk_id", f"chunk_{chunk_idx}")
            video_path: str = chunk.get("path", context.video_path)
            start_time: float = chunk.get("start_time", 0)
            end_time: float = chunk.get("end_time", context.video_duration)

            chunk_output_dir = os.path.join(output_base, str(chunk_id))

            # Extract keyframes in thread pool
            keyframes = await loop.run_in_executor(
                None,
                _extract_keyframes_from_video,
                video_path,
                start_time,
                end_time,
                frames_per_chapter,
                config,
                chunk_output_dir,
                chunk_id,
            )

            # Detect action boundaries
            boundaries = detect_action_boundaries(keyframes, boundary_thresholds)

            # Clean up image data from keyframes (not needed in output)
            for kf in keyframes:
                if "image" in kf:
                    del kf["image"]
                if "frame_bgr" in kf:
                    del kf["frame_bgr"]

            return {
                "chunk_idx": chunk_idx,
                "chunk_id": chunk_id,
                "keyframes": keyframes,
                "boundaries": boundaries,
                "artifacts": [kf["filepath"] for kf in keyframes],
            }

        # Run all chunks
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(video_chunks)]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        for result in results:
            all_keyframes.append({
                "chunk_index": result["chunk_idx"],
                "chunk_id": result["chunk_id"],
                "keyframes": result["keyframes"],
                "boundaries": result["boundaries"],
            })
            all_boundaries.extend([
                {**b, "chunk_id": result["chunk_id"]}
                for b in result["boundaries"]
            ])
            all_artifacts.extend(result["artifacts"])

        total_kf = sum(len(kf["keyframes"]) for kf in all_keyframes)
        context.logger.info(
            f"Extracted {total_kf} keyframes with {len(all_boundaries)} action boundaries "
            f"from {len(video_chunks)} chunks"
        )

        return StepResult(
            step_id=self.step_id,
            outputs={
                "keyframes_per_chunk": all_keyframes,
                "action_boundaries": all_boundaries,
                "frames_per_chunk": frames_per_chapter,
            },
            metrics={
                "total_keyframes": total_kf,
                "action_boundaries_detected": len(all_boundaries),
                "chunks_processed": len(video_chunks),
                "avg_keyframes_per_chunk": total_kf / len(video_chunks) if video_chunks else 0,
            },
            artifacts=all_artifacts,
        )
