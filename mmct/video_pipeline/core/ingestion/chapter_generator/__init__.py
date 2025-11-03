"""
Chapter Generator Module

This module provides chapter generation and ingestion functionality.
"""

from mmct.video_pipeline.core.ingestion.chapter_generator.chapter_generator import ChapterGenerator
from mmct.video_pipeline.core.ingestion.chapter_generator.chapter_ingestion_pipeline import ChapterIngestionPipeline

__all__ = [
    "ChapterGenerator",
    "ChapterIngestionPipeline",
]
