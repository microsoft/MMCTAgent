"""Group summarization using LLM providers.

Generates concise summaries for chapter groups by:
1. Extracting individual chapter summaries
2. Building a prompt with group context (time range, topics)
3. Calling LLM to generate cohesive group summary
4. Falling back to first chapter summary on errors

Supports concurrent processing with configurable rate limiting.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mmct.video_pipeline.core.ingestion.models import ChapterGroup

from .config import (
    GROUP_SUMMARY_MAX_TOKENS,
    GROUP_SUMMARY_TEMPERATURE,
    MAX_CONCURRENT_SUMMARIES,
)


logger = logging.getLogger(__name__)


class GroupSummarizer:
    """Generates LLM-based summaries for chapter groups.
    
    Supports async processing with configurable concurrency and graceful error handling.
    """
    
    GROUP_SUMMARY_PROMPT = """Summarize the following video chapter group content.

GROUP INFORMATION:
- Chapters: {chapter_range}
- Time Range: {time_range}
- Topics: {topics}

CHAPTER SUMMARIES:
{chapter_summaries}

Generate a concise JSON response with:
{{
    "summary": "A 2-3 sentence summary capturing the main content, actions, and key information in this section of the video.",
    "main_theme": "The primary theme or topic of this group in 3-5 words"
}}

Focus on what viewers will learn or see in this section."""

    def __init__(
        self,
        max_tokens: int = GROUP_SUMMARY_MAX_TOKENS,
        temperature: float = GROUP_SUMMARY_TEMPERATURE,
        max_concurrent: int = MAX_CONCURRENT_SUMMARIES,
    ):
        """Initialize the group summarizer.
        
        Args:
            max_tokens: Maximum tokens for summary generation
            temperature: LLM temperature (lower = more deterministic)
            max_concurrent: Maximum concurrent LLM requests
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def generate_group_summaries(
        self,
        groups: List[ChapterGroup],
        chapters: List[Dict[str, Any]],
        llm_provider: Any,
    ) -> List[ChapterGroup]:
        """Generate summaries for all chapter groups.
        
        Processes groups concurrently up to the configured limit.
        Updates each group's summary field in place.
        
        Args:
            groups: List of ChapterGroup objects to summarize
            chapters: Original chapter dictionaries for context
            llm_provider: LLM provider with generate_json method
            
        Returns:
            List of ChapterGroup objects with summaries populated
        """
        if not groups:
            return groups
        
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        tasks = [
            self._summarize_group_with_semaphore(group, chapters, llm_provider)
            for group in groups
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and update groups
        updated_groups: List[ChapterGroup] = []
        for group, result in zip(groups, results):
            if isinstance(result, Exception):
                logger.warning(f"Summary generation failed for group {group.id}: {result}")
                # Keep group with fallback summary
                if not group.summary:
                    group.summary = self._generate_fallback_summary(group, chapters)
                updated_groups.append(group)
            else:
                updated_groups.append(result)
        
        return updated_groups
    
    async def generate_group_summary(
        self,
        chapters: List[Dict[str, Any]],
        llm_provider: Any,
    ) -> str:
        """Generate a summary for a single group of chapters.
        
        Standalone method for generating a summary from a list of chapters
        without requiring a pre-existing ChapterGroup object.
        
        Args:
            chapters: List of chapter dictionaries to summarize
            llm_provider: LLM provider with generate_json method
            
        Returns:
            Summary string for the chapter group
        """
        if not chapters:
            return ""
        
        # Extract chapter summaries
        chapter_summaries = self._extract_chapter_summaries(chapters)
        
        # Build prompt components
        chapter_range = f"Chapters 1-{len(chapters)}"
        time_range = self._format_time_range(chapters)
        topics = self._extract_topics_from_chapters(chapters)
        
        prompt = self.GROUP_SUMMARY_PROMPT.format(
            chapter_range=chapter_range,
            time_range=time_range,
            topics=", ".join(topics) if topics else "General content",
            chapter_summaries=chapter_summaries,
        )
        
        try:
            response = await llm_provider.generate_json(prompt)
            return response.get("summary", "")
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Return fallback from first chapter
            if chapter_summaries:
                first_line = chapter_summaries.split("\n")[0]
                return first_line[:200] if first_line else ""
            return ""
    
    async def _summarize_group_with_semaphore(
        self,
        group: ChapterGroup,
        chapters: List[Dict[str, Any]],
        llm_provider: Any,
    ) -> ChapterGroup:
        """Summarize a single group with semaphore for rate limiting.
        
        Args:
            group: ChapterGroup to summarize
            chapters: All chapter dictionaries
            llm_provider: LLM provider
            
        Returns:
            ChapterGroup with summary populated
        """
        async with self._semaphore:
            return await self._summarize_group(group, chapters, llm_provider)
    
    async def _summarize_group(
        self,
        group: ChapterGroup,
        chapters: List[Dict[str, Any]],
        llm_provider: Any,
    ) -> ChapterGroup:
        """Generate summary for a single chapter group.
        
        Args:
            group: ChapterGroup to summarize
            chapters: All chapter dictionaries
            llm_provider: LLM provider
            
        Returns:
            ChapterGroup with summary populated
        """
        # Get chapters for this group
        group_chapters = [
            chapters[i]
            for i in (group.chapter_indices or [])
            if i < len(chapters)
        ]
        
        if not group_chapters:
            group.summary = self._generate_fallback_summary(group, chapters)
            return group
        
        # Extract chapter summaries
        chapter_summaries = self._extract_chapter_summaries(group_chapters)
        
        # Build prompt
        chapter_range = f"Chapters {min(group.chapter_indices or [0]) + 1}-{max(group.chapter_indices or [0]) + 1}"
        time_range = self._format_group_time_range(group)
        topics = ", ".join(group.topics) if group.topics else "General content"
        
        prompt = self.GROUP_SUMMARY_PROMPT.format(
            chapter_range=chapter_range,
            time_range=time_range,
            topics=topics,
            chapter_summaries=chapter_summaries,
        )
        
        try:
            response = await llm_provider.generate_json(prompt)
            group.summary = response.get("summary", "")
            
            # Update metadata with theme if available
            if response.get("main_theme") and group.metadata is not None:
                group.metadata["main_theme"] = response["main_theme"]
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed for group {group.id}: {e}")
            group.summary = self._generate_fallback_summary(group, chapters)
        
        return group
    
    def _extract_chapter_summaries(
        self,
        chapters: List[Dict[str, Any]],
    ) -> str:
        """Extract and format summaries from chapter dictionaries.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            Formatted string of chapter summaries
        """
        summaries: List[str] = []
        
        for i, chapter in enumerate(chapters):
            chapter_data = chapter.get("chapter", chapter)
            
            if isinstance(chapter_data, dict):
                summary = (
                    chapter_data.get("detailed_summary")
                    or chapter_data.get("summary")
                    or ""
                )
            else:
                summary = str(chapter_data) if chapter_data else ""
            
            if summary:
                # Truncate long summaries
                if len(summary) > 500:
                    summary = summary[:497] + "..."
                summaries.append(f"Chapter {i + 1}: {summary}")
        
        return "\n\n".join(summaries) if summaries else "No chapter summaries available."
    
    def _format_group_time_range(
        self,
        group: ChapterGroup,
    ) -> str:
        """Format time range string for a group.
        
        Args:
            group: ChapterGroup with start_time and end_time
            
        Returns:
            Formatted time range string
        """
        start = group.start_time or 0.0
        end = group.end_time or 0.0
        
        return f"{self._format_timestamp(start)} - {self._format_timestamp(end)}"
    
    def _format_time_range(
        self,
        chapters: List[Dict[str, Any]],
    ) -> str:
        """Format time range string from chapters.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            Formatted time range string
        """
        if not chapters:
            return "0:00 - 0:00"
        
        # Get start time from first chapter
        start = self._get_time_from_chapter(chapters[0], "start")
        # Get end time from last chapter
        end = self._get_time_from_chapter(chapters[-1], "end")
        
        return f"{self._format_timestamp(start)} - {self._format_timestamp(end)}"
    
    def _get_time_from_chapter(
        self,
        chapter: Dict[str, Any],
        time_type: str,
    ) -> float:
        """Extract time value from chapter dictionary.
        
        Args:
            chapter: Chapter dictionary
            time_type: Either 'start' or 'end'
            
        Returns:
            Time in seconds
        """
        # Try direct keys
        if time_type in chapter:
            return float(chapter[time_type])
        
        full_key = f"{time_type}_time"
        if full_key in chapter:
            return float(chapter[full_key])
        
        return 0.0
    
    def _format_timestamp(
        self,
        seconds: float,
    ) -> str:
        """Format seconds as MM:SS or HH:MM:SS string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    def _extract_topics_from_chapters(
        self,
        chapters: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract unique topics from chapters.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            List of unique topic strings
        """
        topics: List[str] = []
        seen: set = set()
        
        for chapter in chapters:
            chapter_data = chapter.get("chapter", chapter)
            if isinstance(chapter_data, dict):
                for key in ["topics", "topic", "category"]:
                    value = chapter_data.get(key)
                    if value:
                        if isinstance(value, list):
                            for t in value:
                                t_lower = str(t).lower()
                                if t_lower not in seen and t_lower != "none":
                                    seen.add(t_lower)
                                    topics.append(str(t))
                        elif str(value).lower() not in seen and str(value).lower() != "none":
                            seen.add(str(value).lower())
                            topics.append(str(value))
        
        return topics[:5]
    
    def _generate_fallback_summary(
        self,
        group: ChapterGroup,
        chapters: List[Dict[str, Any]],
    ) -> str:
        """Generate a fallback summary when LLM call fails.
        
        Args:
            group: ChapterGroup being summarized
            chapters: All chapter dictionaries
            
        Returns:
            Fallback summary string
        """
        if not group.chapter_indices:
            return f"Video section covering {group.name or 'content'}"
        
        # Try to get first chapter's summary
        first_idx = group.chapter_indices[0]
        if first_idx < len(chapters):
            chapter = chapters[first_idx]
            chapter_data = chapter.get("chapter", chapter)
            if isinstance(chapter_data, dict):
                summary = chapter_data.get("detailed_summary") or chapter_data.get("summary", "")
                if summary:
                    # Take first sentence
                    first_sentence = summary.split(".")[0]
                    if len(first_sentence) > 150:
                        first_sentence = first_sentence[:147] + "..."
                    return f"{first_sentence}."
        
        # Generic fallback
        chapter_count = len(group.chapter_indices)
        return f"Video section containing {chapter_count} related chapters."
