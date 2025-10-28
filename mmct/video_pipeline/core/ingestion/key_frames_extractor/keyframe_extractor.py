import os
import cv2
import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio

from mmct.video_pipeline.utils.helper import get_media_folder, get_file_hash

logger = logging.getLogger(__name__)


# ============================================================
# Data classes / config
# ============================================================

@dataclass(frozen=True)
class FrameMetadata:
    """Metadata for an extracted keyframe frame on disk."""
    frame_number: int
    timestamp_seconds: float
    motion_score: float


@dataclass
class KeyframeExtractionConfig:
    """
    Parameters that control keyframe extraction.

    motion_threshold:
        If motion_score >= this, we save that frame.
    sample_fps:
        Target frames/sec to *analyze* (we downsample from source).
    max_frame_width:
        We'll downscale each frame so its longest edge is <= this.
        (Less pixels => cheaper optical flow.)
    debug_mode:
        If True, we keep extracted frame JPGs on disk even after cleanup.
    num_workers:
        How many parallel segments of the video to process.
        1 = sequential. >1 will split the video by frame ranges.
    """
    motion_threshold: float = 0.8
    sample_fps: int = 1
    max_frame_width: int = 800
    debug_mode: bool = False
    num_workers: int = 4


# ============================================================
# Internal helpers
# ============================================================

def _motion_score_cpu(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute optical flow motion score between prev_gray and curr_gray.
    Uses Farneback optical flow on CPU and returns mean flow magnitude.
    """
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
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _sample_interval(actual_fps: float, target_sample_fps: int) -> int:
    """
    Convert "I want ~target_sample_fps frames/sec" into
    "keep 1 every N frames".
    """
    if target_sample_fps <= 0:
        return 1
    if actual_fps <= 0:
        actual_fps = 30.0  # safe fallback
    interval = int(round(actual_fps / float(target_sample_fps)))
    return max(interval, 1)


def _calc_scale_factor(
    width: int, height: int, max_frame_width: int
) -> Tuple[float, int, int]:
    """
    Compute how much to downscale a frame to respect max_frame_width
    (applied to longest edge). Returns (scale_factor, scaled_w, scaled_h).
    """
    longest = max(width, height)
    if longest > max_frame_width:
        scale = max_frame_width / float(longest)
    else:
        scale = 1.0
    return (
        scale,
        int(width * scale),
        int(height * scale),
    )


def _process_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    config: KeyframeExtractionConfig,
    video_hash_id: str,
    keyframes_dir: str,
) -> List[FrameMetadata]:
    """
    Worker that processes a range of frames [start_frame, end_frame).
    Runs in a threadpool worker.

    Each worker:
    - Opens its own VideoCapture
    - Seeks to start_frame
    - Iterates frames in its range
    - Computes motion scores on downsampled grayscale frames
    - Saves keyframes (JPG) when threshold is crossed (or first frame)
    - Returns FrameMetadata list
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video segment: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # clamp segment within bounds of actual file
    start = max(0, start_frame)
    stop = min(total_frames, end_frame)
    if start >= stop:
        cap.release()
        return []

    # compute downscale + temporal sampling
    scale_factor, scaled_w, scaled_h = _calc_scale_factor(
        width, height, config.max_frame_width
    )
    interval = _sample_interval(fps, config.sample_fps)
    threshold = config.motion_threshold

    # log which backend this worker is using
    logger.info(
        f"[segment {start}-{stop}] optical flow backend: CPU (Farneback)"
    )

    # seek to approximate start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    results: List[FrameMetadata] = []
    prev_gray_small: Optional[np.ndarray] = None

    frame_idx = start - 1
    write_jpeg = cv2.imwrite  # local binding == tiny perf improvement

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx >= stop:
            break

        # temporal downsampling: only process 1 out of `interval` frames
        if (frame_idx - start) % interval != 0:
            continue

        ts_sec = frame_idx / fps

        # spatial downsampling to reduce optical flow cost
        if scale_factor < 1.0:
            small_bgr = cv2.resize(
                frame_bgr,
                (scaled_w, scaled_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small_bgr = frame_bgr

        curr_gray_small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # motion score vs prev frame
        if prev_gray_small is None:
            motion_score = 0.0
        else:
            motion_score = _motion_score_cpu(prev_gray_small, curr_gray_small)

        is_first = prev_gray_small is None
        if is_first or (motion_score >= threshold):
            filename = f"{video_hash_id}_{frame_idx}.jpg"
            abs_path = os.path.join(keyframes_dir, filename)

            # Synchronous write. If disk I/O becomes a bottleneck,
            # you can queue these and write in another thread.
            write_jpeg(abs_path, frame_bgr)

            results.append(
                FrameMetadata(
                    frame_number=frame_idx,
                    timestamp_seconds=float(ts_sec),
                    motion_score=float(motion_score),
                )
            )

        prev_gray_small = curr_gray_small

    cap.release()
    return results


# ============================================================
# Main extractor class
# ============================================================

class KeyframeExtractor:
    """
    Keyframe extractor that:
    - Assumes you've already compressed/downsampled the input video (fast proxy)
      via VideoCompressor.fast_mode for speed.
    - Splits that compressed video into N frame ranges
    - Processes each segment in parallel CPU threads
    - For each segment, runs optical flow between sampled frames
    - Writes selected frames to disk as JPGs
    - Returns metadata about the saved frames
    """

    def __init__(self, config: Optional[KeyframeExtractionConfig] = None) -> None:
        self.config = config or KeyframeExtractionConfig()

    @staticmethod
    def get_video_properties(video_path: str) -> Dict[str, Any]:
        """
        Quick probe of basic video properties.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = (frame_count / fps) if fps > 0 else 0.0
        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
        }

    async def extract_keyframes(
        self,
        video_path: str,
        video_id: Optional[str] = None,
    ) -> List[FrameMetadata]:
        """
        Extract keyframes from a (preferably pre-compressed / proxy) video.

        Steps:
        - Hash video to build deterministic output dir
        - Prepare keyframes folder
        - Split total frame range into segments
        - Process segments in parallel with ThreadPoolExecutor
        - Merge + sort metadata
        """

        # Stable ID for filenames
        video_hash_id = video_id or await get_file_hash(video_path)

        # Where extracted JPGs will live
        media_root = await get_media_folder()
        keyframes_dir = os.path.join(media_root, "keyframes", video_hash_id)
        os.makedirs(keyframes_dir, exist_ok=True)

        # Probe video metadata (frame count etc)
        props = self.get_video_properties(video_path)
        total_frames = props["frame_count"]
        fps = props["fps"] if props["fps"] > 0 else 30.0

        # Choose number of workers
        max_possible = os.cpu_count() or 1
        workers = max(1, min(self.config.num_workers, max_possible))

        # Heuristic: avoid overhead for short clips
        if total_frames < workers * 500:
            workers = 1

        logger.info(
            f"KeyframeExtractor: {os.path.basename(video_path)} | "
            f"frames={total_frames} fps={fps:.2f} workers={workers}"
        )

        # Build frame ranges
        if workers == 1:
            segments = [(0, total_frames)]
        else:
            chunk = math.ceil(total_frames / workers)
            segments = [
                (start, min(start + chunk, total_frames))
                for start in range(0, total_frames, chunk)
            ]

        # Parallel execution:
        # We run CPU-bound work in ThreadPoolExecutor. OpenCV's I/O + decode
        # releases the GIL a lot, so threadpool works reasonably well here.
        loop = asyncio.get_running_loop()
        all_results: List[FrameMetadata] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    _process_segment,
                    video_path,
                    seg_start,
                    seg_end,
                    self.config,
                    video_hash_id,
                    keyframes_dir,
                )
                for (seg_start, seg_end) in segments
            ]

            # gather as they finish
            for fut in asyncio.as_completed(futures):
                seg_meta = await fut
                all_results.extend(seg_meta)

        # Order results by frame_number before returning
        all_results.sort(key=lambda m: m.frame_number)

        logger.info(
            f"KeyframeExtractor: extracted {len(all_results)} keyframes -> {keyframes_dir}"
        )

        return all_results

    def cleanup_frames(
        self,
        keyframes_dir: str,
        frame_metadata_list: List[FrameMetadata],
        video_hash_id: str,
    ) -> None:
        """
        Delete all saved JPGs associated with the given frame_metadata_list,
        unless debug_mode is enabled.
        """
        if self.config.debug_mode:
            return

        for meta in frame_metadata_list:
            filename = f"{video_hash_id}_{meta.frame_number}.jpg"
            abs_path = os.path.join(keyframes_dir, filename)
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
            except Exception as e:
                logger.warning(f"Could not remove {filename}: {e}")


# ============================================================
# Convenience top-level async helper
# ============================================================

async def extract_keyframes_from_video(
    video_path: str,
    motion_threshold: float = 0.8,
    sample_fps: int = 1,
    max_frame_width: int = 800,
    debug_mode: bool = False,
    num_workers: int = 4,
) -> List[FrameMetadata]:
    """
    One-shot convenience wrapper that constructs KeyframeExtractor with
    the given tuning and runs it.
    You should pass in a video that is already "fast-mode compressed"
    (low fps, resized) so decode is cheap.
    """
    config = KeyframeExtractionConfig(
        motion_threshold=motion_threshold,
        sample_fps=sample_fps,
        max_frame_width=max_frame_width,
        debug_mode=debug_mode,
        num_workers=num_workers,
    )

    extractor = KeyframeExtractor(config)
    return await extractor.extract_keyframes(video_path)
