"""
Keyframe extraction using optical flow motion detection.
"""

import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Optional, Any
from ..core import FrameMetadata, KeyframeExtractionConfig, get_file_hash, get_media_folder

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """Optimized keyframe extractor using optical flow motion detection."""

    def __init__(self, config: Optional[KeyframeExtractionConfig] = None):
        """
        Initialize the keyframe extractor.

        Args:
            config: Configuration object for extraction parameters
        """
        self.config = config or KeyframeExtractionConfig()

    def calculate_optical_flow_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        Calculate motion score using optimized optical flow.

        Args:
            prev_gray: Previous frame in grayscale
            curr_gray: Current frame in grayscale

        Returns:
            Motion score (higher = more motion)
        """
        # Balanced optical flow parameters for speed vs accuracy
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=12,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Faster magnitude calculation using cv2.cartToPolar
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)

        return motion_score

    def _save_frame(self, frame: np.ndarray, video_hash_id: str, frame_count: int, keyframes_dir: str) -> str:
        """
        Save a single frame to disk.

        Args:
            frame: Frame image array
            video_hash_id: Video hash ID
            frame_count: Frame number
            keyframes_dir: Directory to save frames

        Returns:
            Path to saved frame
        """
        frame_filename = f"{video_hash_id}_{frame_count}.jpg"
        frame_path = os.path.join(keyframes_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        return frame_path

    async def _process_frame_batch(self, pending_frames: List[Dict]) -> List[FrameMetadata]:
        """
        Process a batch of frame metadata.

        Args:
            pending_frames: List of frame info dictionaries

        Returns:
            List of FrameMetadata objects
        """
        if not pending_frames:
            return []

        return [
            FrameMetadata(
                frame_number=frame_info['frame_number'],
                timestamp_seconds=frame_info['timestamp'],
                motion_score=frame_info['motion_score']
            )
            for frame_info in pending_frames
        ]

    def get_video_properties(self, video_path: str) -> Dict[str, Any]:
        """
        Get video properties.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration_seconds': 0
        }

        # Calculate duration
        if props['fps'] > 0:
            props['duration_seconds'] = props['frame_count'] / props['fps']

        cap.release()
        return props

    async def extract_keyframes(self, video_path: str, video_id: Optional[str] = None) -> List[FrameMetadata]:
        """
        Extract keyframes from video using optical flow motion detection.

        Args:
            video_path: Path to the video file
            video_id: Unique video identifier (hash-based, optional)

        Returns:
            List of FrameMetadata objects for extracted keyframes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_per_second = fps if fps > 0 else 30.0

        # Performance optimization: reduce frame size for optical flow computation
        scale_factor = min(1.0, self.config.max_frame_width / max(original_width, original_height))
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Setup directories for frame storage
        if video_id is None:
            video_hash_id = await get_file_hash(video_path)
        else:
            video_hash_id = video_id

        # Setup media directory structure
        media_folder = await get_media_folder()
        keyframes_dir = os.path.join(media_folder, "keyframes", video_hash_id)
        os.makedirs(keyframes_dir, exist_ok=True)

        extracted_frames = []
        pending_frames = []
        prev_gray = None
        frame_count = 0
        batch_size = 10

        # Calculate frame sampling interval
        sample_interval = int(frames_per_second // self.config.sample_fps)
        sample_interval = max(1, sample_interval)  # Ensure at least 1

        logger.info(f"Processing video: {os.path.basename(video_path)} ({original_width}x{original_height})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Sample frames according to configured FPS
            if frame_count % sample_interval != 0:
                continue

            # Calculate timestamp
            timestamp_seconds = float(frame_count) / float(frames_per_second)

            # Resize frame for faster optical flow computation
            small_frame = cv2.resize(frame, (new_width, new_height)) if scale_factor < 1.0 else frame
            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Calculate motion score (0.0 for first frame)
            motion_score = 0.0
            should_save = prev_gray is None  # Always save first frame

            if prev_gray is not None:
                motion_score = self.calculate_optical_flow_score(prev_gray, curr_gray)
                should_save = motion_score > self.config.motion_threshold

            # Save frame if needed
            if should_save:
                self._save_frame(frame, video_hash_id, frame_count, keyframes_dir)

                # Add to batch for processing
                pending_frames.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp_seconds,
                    'motion_score': motion_score
                })

                # Process batch when it reaches the batch size
                if len(pending_frames) >= batch_size:
                    batch_frames = await self._process_frame_batch(pending_frames)
                    extracted_frames.extend(batch_frames)
                    pending_frames.clear()

            prev_gray = curr_gray

        cap.release()

        # Process any remaining frames in the batch
        if pending_frames:
            batch_frames = await self._process_frame_batch(pending_frames)
            extracted_frames.extend(batch_frames)
            pending_frames.clear()

        logger.info(f"Extracted {len(extracted_frames)} keyframes to {keyframes_dir}")

        return extracted_frames

    def cleanup_frames(self, keyframes_dir: str, frame_metadata_list: List[FrameMetadata], video_hash_id: str) -> None:
        """
        Clean up extracted frame files.

        Args:
            keyframes_dir: Directory containing the keyframes
            frame_metadata_list: List of frame metadata objects
            video_hash_id: Video hash ID for filename generation
        """
        if self.config.debug_mode:
            return

        for frame_metadata in frame_metadata_list:
            try:
                frame_filename = f"{video_hash_id}_{frame_metadata.frame_number}.jpg"
                frame_path = os.path.join(keyframes_dir, frame_filename)
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except Exception as e:
                logger.warning(f"Could not remove frame {frame_filename}: {e}")


async def extract_keyframes_from_video(
    video_path: str,
    motion_threshold: float = 0.8,
    sample_fps: int = 1,
    debug_mode: bool = False
) -> List[FrameMetadata]:
    """
    Convenience function to extract keyframes from a video.

    Args:
        video_path: Path to the video file
        motion_threshold: Motion threshold for keyframe extraction
        sample_fps: Frames per second to sample from video
        debug_mode: Whether to enable debug mode

    Returns:
        List of FrameMetadata objects
    """
    config = KeyframeExtractionConfig(
        motion_threshold=motion_threshold,
        sample_fps=sample_fps,
        debug_mode=debug_mode
    )

    extractor = KeyframeExtractor(config)
    return await extractor.extract_keyframes(video_path)
