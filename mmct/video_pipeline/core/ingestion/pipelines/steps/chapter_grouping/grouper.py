"""Chapter grouping logic using semantic similarity and temporal proximity.

Implements a sliding window algorithm that groups consecutive chapters when:
1. Their embeddings have cosine similarity >= configurable threshold
2. They are within the temporal window (max chapter index distance)

The algorithm maintains a running centroid (normalized average embedding)
and starts a new group when similarity drops or window is exceeded.
"""

import logging
import math
import uuid
from typing import Dict, Any, List, Optional, Tuple

from mmct.video_pipeline.core.ingestion.models import ChapterGroup

from .config import (
    CHAPTER_GROUPING_THRESHOLD,
    CHAPTER_TEMPORAL_WINDOW,
    EMBEDDING_DIMENSION,
)


logger = logging.getLogger(__name__)


class ChapterGrouper:
    """Groups chapters by semantic similarity and temporal proximity.
    
    Uses sliding window algorithm with configurable thresholds.
    Supports chapters with or without embeddings (falls back to temporal-only grouping).
    """
    
    def __init__(
        self,
        similarity_threshold: float = CHAPTER_GROUPING_THRESHOLD,
        temporal_window: int = CHAPTER_TEMPORAL_WINDOW,
    ):
        """Initialize the chapter grouper.
        
        Args:
            similarity_threshold: Minimum cosine similarity for grouping (0.0-1.0)
            temporal_window: Maximum chapter index distance for grouping
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
    
    def group_chapters(
        self,
        chapters: List[Dict[str, Any]],
        video_id: str,
        video_duration: float = 0.0,
        playlist_id: Optional[str] = None,
        playlist_order: Optional[int] = None,
    ) -> Tuple[List[ChapterGroup], List[str]]:
        """Group chapters by semantic similarity and temporal proximity.
        
        Uses a sliding window approach to group consecutive chapters that are
        semantically similar. A new group starts when:
        - The similarity between current chapter and group centroid drops below threshold
        - The chapter index distance exceeds the temporal window
        
        Args:
            chapters: List of chapter dictionaries with optional 'embeddings' key
            video_id: Video identifier for the chapter groups
            video_duration: Total duration of the video in seconds
            playlist_id: Optional playlist ID this video belongs to
            playlist_order: Optional 1-based position within the playlist
            
        Returns:
            Tuple of:
            - List of ChapterGroup objects with computed metadata
            - List of warning/info messages generated during grouping
        """
        messages: List[str] = []
        self._playlist_id = playlist_id
        self._playlist_order = playlist_order
        
        if not chapters:
            messages.append("No chapters provided for grouping")
            return [], messages
        
        # Extract embeddings and chapter metadata
        chapter_embeddings = self._extract_embeddings(chapters)
        has_embeddings = any(e is not None for e in chapter_embeddings)
        
        if not has_embeddings:
            messages.append(
                "No embeddings found in chapters. Creating sequential groups "
                f"using temporal window of {self.temporal_window} chapters."
            )
            return self._group_by_temporal_window(chapters, video_id, video_duration), messages
        
        # Group using sliding window with similarity
        groups = self._group_by_similarity(chapters, chapter_embeddings, video_id, video_duration)
        messages.append(
            f"Created {len(groups)} chapter groups from {len(chapters)} chapters "
            f"(threshold={self.similarity_threshold}, window={self.temporal_window})"
        )
        
        return groups, messages
    
    def compute_group_embedding(
        self,
        embeddings: List[List[float]],
    ) -> List[float]:
        """Compute normalized average embedding for a group.
        
        Averages all chapter embeddings in the group and normalizes
        the result to unit length for consistent similarity comparisons.
        
        Args:
            embeddings: List of chapter embedding vectors
            
        Returns:
            Normalized average embedding vector
        """
        if not embeddings:
            return [0.0] * EMBEDDING_DIMENSION
        
        # Filter out None embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            return [0.0] * EMBEDDING_DIMENSION
        
        # Compute average
        dim = len(valid_embeddings[0])
        avg_embedding = [0.0] * dim
        
        for emb in valid_embeddings:
            for i in range(min(dim, len(emb))):
                avg_embedding[i] += emb[i]
        
        for i in range(dim):
            avg_embedding[i] /= len(valid_embeddings)
        
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in avg_embedding))
        if norm > 0:
            avg_embedding = [x / norm for x in avg_embedding]
        
        return avg_embedding
    
    def _extract_embeddings(
        self,
        chapters: List[Dict[str, Any]],
    ) -> List[Optional[List[float]]]:
        """Extract embeddings from chapter dictionaries.
        
        Looks for embeddings in multiple possible locations:
        - Direct 'embeddings' key
        - Nested in 'chapter' dictionary
        - 'embedding_vector' key
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            List of embedding vectors (None for chapters without embeddings)
        """
        embeddings: List[Optional[List[float]]] = []
        
        for chapter in chapters:
            embedding = None
            
            # Try direct embeddings key
            if "embeddings" in chapter and chapter["embeddings"]:
                embedding = chapter["embeddings"]
            # Try nested in chapter dict
            elif "chapter" in chapter:
                chapter_data = chapter["chapter"]
                if isinstance(chapter_data, dict):
                    embedding = chapter_data.get("embeddings")
            # Try embedding_vector key
            if embedding is None and "embedding_vector" in chapter:
                embedding = chapter["embedding_vector"]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _group_by_similarity(
        self,
        chapters: List[Dict[str, Any]],
        embeddings: List[Optional[List[float]]],
        video_id: str,
        video_duration: float = 0.0,
    ) -> List[ChapterGroup]:
        """Group chapters using sliding window with similarity threshold.
        
        Args:
            chapters: List of chapter dictionaries
            embeddings: Corresponding embedding vectors
            video_id: Video identifier
            video_duration: Total duration of the video in seconds
            
        Returns:
            List of ChapterGroup objects
        """
        groups: List[ChapterGroup] = []
        current_group_indices: List[int] = []
        current_group_embeddings: List[List[float]] = []
        group_centroid: Optional[List[float]] = None
        
        for i, (chapter, embedding) in enumerate(zip(chapters, embeddings)):
            if not current_group_indices:
                # Start new group
                current_group_indices.append(i)
                if embedding:
                    current_group_embeddings.append(embedding)
                    group_centroid = embedding[:]
                continue
            
            # Check temporal window constraint
            if i - current_group_indices[0] > self.temporal_window:
                # Finalize current group and start new one
                groups.append(self._create_group(
                    chapters, current_group_indices, current_group_embeddings, video_id, video_duration
                ))
                current_group_indices = [i]
                current_group_embeddings = [embedding] if embedding else []
                group_centroid = embedding[:] if embedding else None
                continue
            
            # Check similarity constraint
            if group_centroid and embedding:
                similarity = self._cosine_similarity(group_centroid, embedding)
                
                if similarity < self.similarity_threshold:
                    # Finalize current group and start new one
                    groups.append(self._create_group(
                        chapters, current_group_indices, current_group_embeddings, video_id, video_duration
                    ))
                    current_group_indices = [i]
                    current_group_embeddings = [embedding]
                    group_centroid = embedding[:]
                    continue
            
            # Add to current group
            current_group_indices.append(i)
            if embedding:
                current_group_embeddings.append(embedding)
                # Update centroid incrementally
                group_centroid = self.compute_group_embedding(current_group_embeddings)
        
        # Finalize last group
        if current_group_indices:
            groups.append(self._create_group(
                chapters, current_group_indices, current_group_embeddings, video_id, video_duration
            ))
        
        # Assign sequential order to all groups
        for order_idx, group in enumerate(groups):
            group.order = order_idx
        
        return groups
    
    def _group_by_temporal_window(
        self,
        chapters: List[Dict[str, Any]],
        video_id: str,
        video_duration: float = 0.0,
    ) -> List[ChapterGroup]:
        """Group chapters by temporal window only (no embeddings).
        
        Creates groups of consecutive chapters up to the temporal window size.
        
        Args:
            chapters: List of chapter dictionaries
            video_id: Video identifier
            video_duration: Total duration of the video in seconds
            
        Returns:
            List of ChapterGroup objects
        """
        groups: List[ChapterGroup] = []
        
        for i in range(0, len(chapters), self.temporal_window):
            end_idx = min(i + self.temporal_window, len(chapters))
            group_indices = list(range(i, end_idx))
            groups.append(self._create_group(chapters, group_indices, [], video_id, video_duration))
        
        # Assign sequential order to all groups
        for order_idx, group in enumerate(groups):
            group.order = order_idx
        
        return groups
    
    def _create_group(
        self,
        chapters: List[Dict[str, Any]],
        indices: List[int],
        embeddings: List[List[float]],
        video_id: str,
        video_duration: float = 0.0,
    ) -> ChapterGroup:
        """Create a ChapterGroup from selected chapter indices.
        
        Args:
            chapters: All chapter dictionaries
            indices: Indices of chapters in this group
            embeddings: Embedding vectors for chapters in this group
            video_id: Video identifier
            video_duration: Total duration of the video in seconds
            
        Returns:
            ChapterGroup with computed metadata
        """
        group_chapters = [chapters[i] for i in indices]
        
        # Compute time bounds
        start_time = self._get_chapter_time(group_chapters[0], "start")
        end_time = self._get_chapter_time(group_chapters[-1], "end")
        
        # Extract topics from chapters
        topics = self._extract_topics(group_chapters)
        
        # Create group name from first chapter summary or index
        name = self._generate_group_name(group_chapters, len(indices))
        
        # Compute group embedding
        group_embedding = self.compute_group_embedding(embeddings) if embeddings else None
        
        return ChapterGroup(
            id=f"group_{uuid.uuid4().hex[:12]}",
            name=name,
            video_id=video_id,
            chapter_indices=indices,
            start_time=start_time,
            end_time=end_time,
            video_duration=video_duration,
            summary=None,  # Will be populated by summarizer
            topics=topics if topics else None,
            parent_group_id=None,
            metadata={
                "chapter_count": len(indices),
                "embedding_vector": group_embedding,
                **({"playlist_id": self._playlist_id} if self._playlist_id else {}),
                **({"playlist_order": self._playlist_order} if self._playlist_order is not None else {}),
            },
        )
    
    def _get_chapter_time(
        self,
        chapter: Dict[str, Any],
        time_key: str,
    ) -> float:
        """Extract start or end time from chapter dictionary.
        
        Args:
            chapter: Chapter dictionary
            time_key: Either 'start' or 'end'
            
        Returns:
            Time in seconds (0.0 if not found)
        """
        # Try direct keys
        if time_key in chapter:
            return float(chapter[time_key])
        
        # Try start_time/end_time variants
        full_key = f"{time_key}_time"
        if full_key in chapter:
            return float(chapter[full_key])
        
        return 0.0
    
    def _extract_topics(
        self,
        chapters: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract unique topics from group chapters.
        
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
                # Try various topic keys
                for key in ["topics", "topic", "category", "sub_category"]:
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
        
        return topics[:10]  # Limit to 10 topics
    
    def _generate_group_name(
        self,
        chapters: List[Dict[str, Any]],
        group_size: int,
    ) -> str:
        """Generate a descriptive name for the group.
        
        Args:
            chapters: Chapters in the group
            group_size: Number of chapters in the group
            
        Returns:
            Group name string
        """
        # Try to get summary from first chapter
        first_chapter = chapters[0]
        chapter_data = first_chapter.get("chapter", first_chapter)
        
        if isinstance(chapter_data, dict):
            summary = chapter_data.get("detailed_summary") or chapter_data.get("summary", "")
            if summary:
                # Take first sentence or truncate
                first_sentence = summary.split(".")[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:77] + "..."
                return first_sentence
        
        # Fallback to generic name
        return f"Chapter Group ({group_size} chapters)"
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        if not vec1 or not vec2:
            return 0.0
        
        min_len = min(len(vec1), len(vec2))
        
        dot_product = sum(vec1[i] * vec2[i] for i in range(min_len))
        norm1 = math.sqrt(sum(x * x for x in vec1[:min_len]))
        norm2 = math.sqrt(sum(x * x for x in vec2[:min_len]))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
