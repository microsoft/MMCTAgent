"""
Transcript Chunker: Splits video based on semantic clusters.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import subprocess


from mmct.video_pipeline.utils.helper import get_media_folder


class TranscriptChunker:
    """
    Splits video based on semantic clusters provided by SemanticChunker.
    """

    def __init__(self, video_path: Path, clusters: List[Any]):
        self.video_path = video_path
        self.clusters = clusters

    async def run(self) -> List[Dict[str, Any]]:
        """
        Executes the chunking process.
        Returns a list of dicts with keys: chunk_id, start_time, end_time, transcript.

        Note: No physical video splitting is performed since keyframes are extracted
        from the full video and filtered by timestamp range during chapter generation.
        """
        chunks_metadata = []

        for i, cluster in enumerate(self.clusters):
            # Assuming cluster is a TranscriptSegment object (pydantic)
            start_time = cluster.start_time
            end_time = cluster.end_time
            transcript = cluster.sentence

            duration = end_time - start_time
            if duration <= 0:
                logger.warning(
                    f"Skipping empty/negative duration chunk {i}: {start_time} -> {end_time}"
                )
                continue

            chunks_metadata.append(
                {
                    "chunk_id": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "transcript": transcript,
                }
            )

        logger.info(f"Created {len(chunks_metadata)} chunk metadata entries (no video splitting)")
        return chunks_metadata
