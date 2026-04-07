"""Event extraction from video chapters using multimodal LLM.

Extracts atomic events from chapter content using frames + transcript,
then generates embeddings for semantic similarity search.

Architecture:
- Input: Keyframes (visual) + transcript (audio) per chapter
- Output: List of GraphEvent objects with embeddings
- Timestamps: Relative to chapter start, converted to absolute

Event Types:
- action: Physical movement, manipulation, or activity
- dialogue: Speech, narration, or verbal communication
- transition: Scene change, visual transition, or context shift
- state_change: Object or environment state modification
"""

import json
from typing import List, Dict, Any, Optional

from loguru import logger

from mmct.providers.base import BaseLLMProvider, BaseEmbeddingProvider
from mmct.video_pipeline.core.ingestion.models import GraphEvent

from .config import (
    MAX_EVENTS_PER_CHAPTER,
    MIN_EVENT_DURATION_MS,
    MAX_EXTRACTION_RETRIES,
    EVENT_EXTRACTION_TEMPERATURE,
    EMBEDDING_BATCH_SIZE,
)
from .prompts import build_event_messages


DEFAULT_MAX_FRAMES_PER_CHAPTER = 12


class EventExtractor:
    """Extracts atomic events from video chapters using multimodal LLM.
    
    Uses vision-capable LLM to identify discrete events from:
    - Visual content (keyframe sequence in chronological order)
    - Audio content (transcript)
    
    Generates embeddings using fastembed with BAAI/bge-small-en-v1.5 (384-dim).
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        max_events_per_chapter: int = MAX_EVENTS_PER_CHAPTER,
        min_event_duration_ms: int = MIN_EVENT_DURATION_MS,
        max_retries: int = MAX_EXTRACTION_RETRIES,
        max_frames_per_chapter: int = DEFAULT_MAX_FRAMES_PER_CHAPTER,
    ):
        """Initialize the EventExtractor.
        
        Args:
            llm_provider: Provider for LLM-based event extraction
            embedding_provider: Provider for generating event embeddings
            max_events_per_chapter: Maximum events per chapter (default: 10)
            min_event_duration_ms: Minimum event duration in ms (default: 500)
            max_retries: Max retry attempts (default: 3)
            max_frames_per_chapter: Max frames per chapter (default: 12)
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.max_events_per_chapter = max_events_per_chapter
        self.min_event_duration_ms = min_event_duration_ms
        self.max_retries = max_retries
        self.max_frames_per_chapter = max_frames_per_chapter
    
    async def extract_events_from_chapter(
        self,
        chapter_data: Dict[str, Any],
        keyframes: Dict[str, Any],
        chapter_index: int,
        video_id: str,
    ) -> List[GraphEvent]:
        """Extract events from a single chapter using multimodal analysis.
        
        Args:
            chapter_data: Chapter data containing transcript, summary, etc.
            keyframes: Keyframe data for this chapter
            chapter_index: Index of the chapter in the video
            video_id: Unique identifier for the video
            
        Returns:
            List of GraphEvent objects extracted from the chapter
        """
        start_time = chapter_data.get("start", 0.0)
        
        # Build multimodal messages with frames + transcript + summary
        messages = build_event_messages(
            chapter_data=chapter_data,
            keyframes=keyframes,
            max_frames=self.max_frames_per_chapter,
            max_events=self.max_events_per_chapter,
            min_duration_ms=self.min_event_duration_ms,
        )
        
        keyframes_count = len(keyframes.get("keyframes", []))
        logger.info(f"Chapter {chapter_index}: extracting events from {keyframes_count} keyframes")
        
        raw_events = await self._extract_with_retry(messages, chapter_index)
        
        if not raw_events:
            return []
        
        events = self._parse_events(
            raw_events=raw_events,
            chapter_index=chapter_index,
            video_id=video_id,
            chapter_start_time=start_time,
        )
        
        if self.embedding_provider and events:
            events = await self._generate_embeddings(events)
        
        return events
    
    async def extract_events_from_chapters(
        self,
        chapters: List[Dict[str, Any]],
        keyframes_list: List[Dict[str, Any]],
        video_id: str,
    ) -> List[GraphEvent]:
        """Extract events from multiple chapters using multimodal analysis.
        
        Args:
            chapters: List of chapter data dictionaries
            keyframes_list: List of keyframe data for each chapter
            video_id: Unique identifier for the video
            
        Returns:
            Combined list of GraphEvent objects from all chapters
        """
        all_events: List[GraphEvent] = []
        global_sequence = 0
        
        for idx, chapter in enumerate(chapters):
            logger.info(f"Extracting events from chapter {idx + 1}/{len(chapters)}")
            
            # Get keyframes for this chapter
            keyframes = keyframes_list[idx] if idx < len(keyframes_list) else {}
            
            chapter_events = await self.extract_events_from_chapter(
                chapter_data=chapter,
                keyframes=keyframes,
                chapter_index=idx,
                video_id=video_id,
            )
            
            for event in chapter_events:
                global_sequence += 1
                event.sequence_number = global_sequence
            
            all_events.extend(chapter_events)
        
        logger.info(f"Extracted {len(all_events)} total events from {len(chapters)} chapters")
        return all_events
    
    async def _extract_with_retry(
        self,
        messages: List[Dict[str, str]],
        chapter_index: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """Extract events with retry logic.
        
        Args:
            messages: List of messages for LLM chat completion
            chapter_index: Index of the chapter being processed (for logging)
            
        Returns:
            Parsed JSON response or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.llm_provider.chat_completion(
                    messages=messages,
                    temperature=EVENT_EXTRACTION_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                
                if response and "content" in response:
                    content = response["content"]
                    if isinstance(content, str):
                        return json.loads(content)
                    elif isinstance(content, dict):
                        return content
                
                logger.warning(f"Chapter {chapter_index}: unexpected response format")
                
            except json.JSONDecodeError as e:
                logger.error(f"Chapter {chapter_index}: JSON parse error - {e}")
            except Exception as e:
                logger.exception(f"Chapter {chapter_index}: extraction attempt {attempt + 1}/{self.max_retries} failed")
                
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def _parse_events(
        self,
        raw_events: Dict[str, Any],
        chapter_index: int,
        video_id: str,
        chapter_start_time: float,
    ) -> List[GraphEvent]:
        """Parse raw LLM response into GraphEvent objects.
        
        Args:
            raw_events: Raw JSON response from LLM
            chapter_index: Index of the chapter
            video_id: Video identifier
            chapter_start_time: Start time of the chapter
            
        Returns:
            List of validated GraphEvent objects
        """
        events_data = raw_events.get("events", [])
        parsed_events: List[GraphEvent] = []
        
        for idx, event_data in enumerate(events_data[:self.max_events_per_chapter]):
            try:
                relative_timestamp = float(event_data.get("timestamp", 0.0))
                absolute_timestamp = chapter_start_time + relative_timestamp
                
                duration = float(event_data.get("duration", 0.0))
                if duration * 1000 < self.min_event_duration_ms:
                    duration = self.min_event_duration_ms / 1000.0
                
                event = GraphEvent(
                    id=f"evt_{video_id}_{chapter_index}_{idx:03d}",
                    description=event_data.get("description", ""),
                    video_id=video_id,
                    timestamp=absolute_timestamp,
                    duration=duration,
                    event_type=event_data.get("event_type", "action"),
                    participants=event_data.get("participants", []),
                    chapter_index=chapter_index,
                    sequence_number=event_data.get("sequence_number", idx + 1),
                    embedding_vector=None,
                    metadata={
                        "relative_timestamp": relative_timestamp,
                        "extraction_source": "llm",
                    },
                )
                
                if event.description:
                    parsed_events.append(event)
                    
            except Exception as e:
                logger.warning(f"Failed to parse event {idx}: {e}")
                continue
        
        return parsed_events
    
    async def _generate_embeddings(
        self,
        events: List[GraphEvent],
    ) -> List[GraphEvent]:
        """Generate embeddings for events.
        
        Args:
            events: List of events to generate embeddings for
            
        Returns:
            Events with embedding_vector populated
        """
        if not self.embedding_provider or not events:
            return events
        
        descriptions = [event.description for event in events]
        
        try:
            all_embeddings: List[List[float]] = []
            for i in range(0, len(descriptions), EMBEDDING_BATCH_SIZE):
                batch = descriptions[i:i + EMBEDDING_BATCH_SIZE]
                batch_embeddings = await self.embedding_provider.batch_embedding(batch)
                all_embeddings.extend(batch_embeddings)
            
            for idx, event in enumerate(events):
                if idx < len(all_embeddings):
                    event.embedding_vector = all_embeddings[idx]
                    
        except Exception as e:
            logger.warning(f"Failed to generate event embeddings: {e}")
        
        return events
