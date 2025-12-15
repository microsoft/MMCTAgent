"""
Chapter Generator Module

This module provides chapter generation and ingestion functionality.
"""

from .simple.chapter_generator import ChapterGenerator
from .simple.chapter_ingestion_pipeline import ChapterIngestionPipeline

__all__ = [
    "ChapterGenerator",
    "ChapterIngestionPipeline",
]
