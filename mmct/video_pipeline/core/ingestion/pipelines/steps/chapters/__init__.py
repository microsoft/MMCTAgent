"""
Chapter Generator Module

This module provides chapter generation and ingestion functionality.
"""

from .chapter_generator import ChapterGenerator
from .chapter_ingestion_pipeline import ChapterIngestionPipeline

__all__ = [
    "ChapterGenerator",
    "ChapterIngestionPipeline",
]
