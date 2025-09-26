import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from mmct.video_pipeline.utils.helper import get_media_folder, get_file_hash

logger = logging.getLogger(__name__)

@dataclass
class FrameMetadata:
    """Metadata for an extracted frame."""
    frame_number: int
    timestamp_seconds: float
    motion_score: float

@dataclass
class KeyframeExtractionConfig:
    """Configuration for keyframe extraction."""
    motion_threshold: float = 0.8
    sample_fps: int = 1
    max_frame_width: int = 800
    debug_mode: bool = False

class KeyframeExtractor:
    """Optimized keyframe extractor using optical flow motion detection."""
    
    def __init__(self, config: KeyframeExtractionConfig = None):
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
    
    async def _save_frame_batch(self, pending_frames: List[Dict]) -> List[FrameMetadata]:
        """Save a batch of frames and return metadata."""
        if not pending_frames:
            return []

        frame_metadata_list = []

        for frame_info in pending_frames:
            frame_metadata = FrameMetadata(
                frame_number=frame_info['frame_number'],
                timestamp_seconds=frame_info['timestamp'],
                motion_score=frame_info['motion_score']
            )
            frame_metadata_list.append(frame_metadata)

        return frame_metadata_list
    
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
    
    async def extract_keyframes(self, video_path: str, video_id: str = None) -> List[FrameMetadata]:
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
        saved_count = 0
        frame_count = 0
        batch_size = 10
        
        # Calculate frame sampling interval
        sample_interval = frames_per_second // self.config.sample_fps
        if sample_interval <= 0:
            sample_interval = 1
        
        logger.info(f"Processing video: {os.path.basename(video_path)} ({original_width}x{original_height})")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames according to configured FPS
            if frame_count % sample_interval != 0:
                continue
            
            # Calculate timestamp (ensure float precision)
            timestamp_seconds = float(frame_count) / float(frames_per_second)
            
            # Resize frame for faster optical flow computation
            if scale_factor < 1.0:
                small_frame = cv2.resize(frame, (new_width, new_height))
            else:
                small_frame = frame
            
            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            
            if prev_gray is not None:
                # Calculate motion score on smaller frame
                motion_score = self.calculate_optical_flow_score(prev_gray, curr_gray)
                
                # Save frame if motion exceeds threshold
                if motion_score > self.config.motion_threshold:
                    frame_filename = f"{video_hash_id}_{frame_count}.jpg"
                    frame_path = os.path.join(keyframes_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)

                    # Add to batch for processing
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp': timestamp_seconds,
                        'motion_score': motion_score
                    }
                    pending_frames.append(frame_info)
                    saved_count += 1

                    # Process batch when it reaches the batch size
                    if len(pending_frames) >= batch_size:
                        batch_frames = await self._save_frame_batch(pending_frames)
                        extracted_frames.extend(batch_frames)
                        pending_frames.clear()
            else:
                # Always save first frame
                frame_filename = f"{video_hash_id}_{frame_count}.jpg"
                frame_path = os.path.join(keyframes_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                # Add to batch for processing
                frame_info = {
                    'frame_number': frame_count,
                    'timestamp': timestamp_seconds,
                    'motion_score': motion_score
                }
                pending_frames.append(frame_info)
                saved_count += 1
            
            prev_gray = curr_gray
        
        cap.release()
        
        # Process any remaining frames in the batch
        if pending_frames:
            batch_frames = await self._save_frame_batch(pending_frames)
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


async def extract_keyframes_from_video(video_path: str,
                                      motion_threshold: float = 0.8,
                                      sample_fps: int = 1,
                                      debug_mode: bool = False) -> List[FrameMetadata]:
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


if __name__ == "__main__":
    import argparse
    import asyncio

    video_path = "/home/v-amanpatkar/work/chapter_generation_opt/video/sample.mp4"
    motion_threshold = 2.5


    async def main():

        try:
            frames = await extract_keyframes_from_video(
                video_path=video_path,
                motion_threshold=motion_threshold,
            )

            print(f"\nExtracted {len(frames)} keyframes:")
            for frame in frames:
                print(f"  Frame {frame.frame_number}: {frame.timestamp_seconds:.2f}s (motion: {frame.motion_score:.3f})")

        except Exception as e:
            print(f"Error: {e}")
            return 1

        return 0

    exit(asyncio.run(main()))