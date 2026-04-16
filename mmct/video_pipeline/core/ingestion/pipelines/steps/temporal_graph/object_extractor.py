"""Object extraction from video frames using vision LLM with transcript-based name resolution.

Extracts identifiable objects from visual content (frames) and uses transcript
to resolve actual names of people/objects when mentioned in dialogue.

Architecture:
- Input: Keyframes per chapter + transcript for name resolution
- Output: List of GraphObject objects with embeddings
- Name Resolution: Uses transcript to identify actual names (e.g., "Jitendra Kumar Kushwaha" not "man in blue shirt")
- Timestamps: first_seen and last_seen relative to chapter, converted to absolute
- Deduplication: Embedding-based similarity with configurable threshold

Object Types:
- person: Human individuals (uses actual names from transcript when available)
- item: Physical objects, tools, products
- animal: Animals, pets, wildlife
- text: On-screen text, signs, labels
- vehicle: Cars, bikes, transportation
- location: Identifiable places, rooms, settings
- food: Food items, ingredients, dishes
"""

import json
from typing import List, Dict, Any, Optional

from loguru import logger

from mmct.providers.base import BaseLLMProvider, BaseEmbeddingProvider
from mmct.video_pipeline.core.ingestion.models import GraphObject

from .config import (
    MAX_EXTRACTION_RETRIES,
    OBJECT_EXTRACTION_TEMPERATURE,
    OBJECT_SIMILARITY_THRESHOLD,
    EMBEDDING_BATCH_SIZE,
)
from .prompts import build_object_messages


DEFAULT_MAX_FRAMES_PER_CHAPTER = 12


class ObjectExtractor:
    """Extracts identifiable objects from video frames using vision LLM with transcript-based name resolution.
    
    Uses vision-capable LLM to identify objects from visual content, cross-referencing
    transcript to use actual names when people/objects are mentioned by name.
    Generates embeddings using fastembed with snowflake/snowflake-arctic-embed-s (384-dim).
    
    Supports two deduplication modes:
    - Incremental: Deduplicate as objects are extracted (for sequential processing)
    - Batch: Deduplicate all objects at once (for parallel processing)
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        max_objects_per_chapter: int = 15,
        similarity_threshold: float = OBJECT_SIMILARITY_THRESHOLD,
        max_retries: int = MAX_EXTRACTION_RETRIES,
        max_frames_per_chapter: int = DEFAULT_MAX_FRAMES_PER_CHAPTER,
    ):
        """Initialize the ObjectExtractor.
        
        Args:
            llm_provider: Provider for LLM-based object extraction (vision)
            embedding_provider: Provider for generating object embeddings
            max_objects_per_chapter: Maximum objects per chapter (default: 15)
            similarity_threshold: Cosine similarity threshold for dedup (default: 0.80)
            max_retries: Max retry attempts (default: 3)
            max_frames_per_chapter: Max frames per chapter (default: 12)
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.max_objects_per_chapter = max_objects_per_chapter
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.max_frames_per_chapter = max_frames_per_chapter
        
        self._seen_objects: Dict[str, GraphObject] = {}
        self._object_embeddings: Dict[str, List[float]] = {}
    
    def reset_deduplication_state(self) -> None:
        """Reset deduplication state for a new video."""
        self._seen_objects.clear()
        self._object_embeddings.clear()
    
    async def extract_objects_from_chapter(
        self,
        chapter_data: Dict[str, Any],
        keyframes: Dict[str, Any],
        chapter_index: int,
        video_id: str,
        skip_deduplication: bool = False,
    ) -> List[GraphObject]:
        """Extract objects from a single chapter using visual-only analysis.
        
        Args:
            chapter_data: Chapter data with start, end, video_id
            keyframes: Keyframe data for this chapter
            chapter_index: Index of the chapter in the video
            video_id: Unique identifier for the video
            skip_deduplication: If True, skip incremental deduplication (for batch dedup later)
            
        Returns:
            List of GraphObject objects extracted from the chapter (with embeddings)
        """
        start_time = chapter_data.get("start", 0.0)
        end_time = chapter_data.get("end", 0.0)
        
        # Add video_id to chapter_data for message building
        chapter_with_video_id = {**chapter_data, "video_id": video_id}
        
        # Build visual-only messages (frames only, no transcript/summary)
        messages = build_object_messages(
            chapter_data=chapter_with_video_id,
            keyframes=keyframes,
            max_frames=self.max_frames_per_chapter,
            max_objects=self.max_objects_per_chapter,
        )
        
        keyframes_count = len(keyframes.get("keyframes", []))
        logger.info(f"Chapter {chapter_index}: extracting objects from {keyframes_count} keyframes")
        
        raw_objects = await self._extract_with_retry(messages, chapter_index)
        
        if not raw_objects:
            return []
        
        objects = self._parse_objects(
            raw_objects=raw_objects,
            chapter_index=chapter_index,
            video_id=video_id,
            chapter_start_time=start_time,
            chapter_end_time=end_time,
        )
        
        # Generate embeddings for deduplication
        if self.embedding_provider and objects:
            objects = await self._generate_embeddings(objects)
        
        # Skip deduplication if batch dedup will happen later
        if not skip_deduplication:
            objects = await self._deduplicate_objects(objects)
        
        return objects
    
    async def _extract_with_retry(
        self,
        messages: List[Dict[str, str]],
        chapter_index: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """Extract objects with retry logic.
        
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
                    temperature=OBJECT_EXTRACTION_TEMPERATURE,
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
    
    def _parse_objects(
        self,
        raw_objects: Dict[str, Any],
        chapter_index: int,
        video_id: str,
        chapter_start_time: float,
        chapter_end_time: float,
    ) -> List[GraphObject]:
        """Parse raw LLM response into GraphObject objects.
        
        Args:
            raw_objects: Raw JSON response from LLM
            chapter_index: Index of the chapter
            video_id: Video identifier
            chapter_start_time: Start time of the chapter
            chapter_end_time: End time of the chapter
            
        Returns:
            List of validated GraphObject objects
        """
        objects_data = raw_objects.get("objects", [])
        parsed_objects: List[GraphObject] = []
        
        for idx, obj_data in enumerate(objects_data[:self.max_objects_per_chapter]):
            try:
                relative_first_seen = float(obj_data.get("first_seen", 0.0))
                absolute_first_seen = chapter_start_time + relative_first_seen
                
                # Parse last_seen from LLM response, fall back to chapter_end if not provided
                relative_last_seen = obj_data.get("last_seen")
                if relative_last_seen is not None:
                    absolute_last_seen = chapter_start_time + float(relative_last_seen)
                else:
                    absolute_last_seen = chapter_end_time
                
                appearance = obj_data.get("appearance", [])
                if isinstance(appearance, str):
                    appearance = [appearance]
                
                identity = obj_data.get("identity", [])
                if isinstance(identity, str):
                    identity = [identity]
                
                obj = GraphObject(
                    id=f"obj_{video_id}_{chapter_index}_{idx:03d}",
                    name=obj_data.get("name", "Unknown object"),
                    video_id=video_id,
                    first_seen=absolute_first_seen,
                    last_seen=absolute_last_seen,
                    object_type=obj_data.get("object_type", "item"),
                    appearance=appearance,
                    identity=identity,
                    embedding_vector=None,
                    metadata={
                        "source_chapter": chapter_index,
                        "relative_first_seen": relative_first_seen,
                        "relative_last_seen": relative_last_seen if relative_last_seen is not None else (chapter_end_time - chapter_start_time),
                        "extraction_source": "llm",
                    },
                )
                
                if obj.name:
                    parsed_objects.append(obj)
                    
            except Exception as e:
                logger.warning(f"Failed to parse object {idx}: {e}")
                continue
        
        return parsed_objects
    
    async def _generate_embeddings(
        self,
        objects: List[GraphObject],
    ) -> List[GraphObject]:
        """Generate embeddings for objects.
        
        Args:
            objects: List of objects to generate embeddings for
            
        Returns:
            Objects with embedding_vector populated
        """
        if not self.embedding_provider or not objects:
            return objects
        
        texts = [self._build_object_embedding_text(obj) for obj in objects]
        
        try:
            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + EMBEDDING_BATCH_SIZE]
                batch_embeddings = await self.embedding_provider.batch_embedding(batch)
                all_embeddings.extend(batch_embeddings)
            
            for idx, obj in enumerate(objects):
                if idx < len(all_embeddings):
                    obj.embedding_vector = all_embeddings[idx]
                    
        except Exception as e:
            logger.warning(f"Failed to generate object embeddings: {e}")
        
        return objects
    
    def _build_object_embedding_text(self, obj: GraphObject) -> str:
        """Build text representation of object for embedding.
        
        Args:
            obj: GraphObject to build text for
            
        Returns:
            Concatenated text for embedding generation
        """
        parts = [obj.name or ""]
        
        if obj.appearance:
            parts.append(" ".join(obj.appearance[:5]))
        
        if obj.identity:
            parts.append(" ".join(obj.identity[:5]))
        
        if obj.object_type:
            parts.append(obj.object_type)
        
        return " ".join(filter(None, parts))
    
    async def _deduplicate_objects(
        self,
        new_objects: List[GraphObject],
    ) -> List[GraphObject]:
        """Deduplicate objects within the video.
        
        Compares new objects against previously seen objects using
        embedding similarity. Updates existing objects with new appearance
        information if duplicate is found.
        
        Args:
            new_objects: Newly extracted objects
            
        Returns:
            List of unique objects (new or updated)
        """
        unique_new: List[GraphObject] = []
        
        for obj in new_objects:
            if not obj.embedding_vector:
                unique_new.append(obj)
                self._seen_objects[obj.id] = obj
                continue
            
            is_duplicate = False
            matched_id: Optional[str] = None
            highest_similarity = 0.0
            
            for existing_id, existing_embedding in self._object_embeddings.items():
                similarity = self._cosine_similarity(
                    obj.embedding_vector,
                    existing_embedding,
                )
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    if similarity >= self.similarity_threshold:
                        is_duplicate = True
                        matched_id = existing_id
            
            if is_duplicate and matched_id:
                existing_obj = self._seen_objects[matched_id]
                self._merge_object_info(existing_obj, obj)
                logger.debug(
                    f"Merged duplicate object '{obj.name}' with '{existing_obj.name}' "
                    f"(similarity: {highest_similarity:.2f})"
                )
            else:
                unique_new.append(obj)
                self._seen_objects[obj.id] = obj
                self._object_embeddings[obj.id] = obj.embedding_vector
        
        return unique_new
    
    def _merge_object_info(
        self,
        existing: GraphObject,
        new: GraphObject,
    ) -> None:
        """Merge new object information into existing object.
        
        Args:
            existing: Existing object to update
            new: New object with additional information
        """
        if new.last_seen and (not existing.last_seen or new.last_seen > existing.last_seen):
            existing.last_seen = new.last_seen
        
        if new.appearance:
            existing_appearances = set(existing.appearance or [])
            for app in new.appearance:
                if app not in existing_appearances:
                    existing.appearance = (existing.appearance or []) + [app]
                    existing_appearances.add(app)
        
        if new.identity:
            existing_identities = set(existing.identity or [])
            for ident in new.identity:
                if ident not in existing_identities:
                    existing.identity = (existing.identity or []) + [ident]
                    existing_identities.add(ident)
        
        if existing.metadata and new.metadata:
            source_chapters = existing.metadata.get("source_chapters", [existing.metadata.get("source_chapter")])
            new_chapter = new.metadata.get("source_chapter")
            if new_chapter and new_chapter not in source_chapters:
                source_chapters.append(new_chapter)
            existing.metadata["source_chapters"] = source_chapters
            existing.metadata["appearance_count"] = len(source_chapters)
    
    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity value between 0 and 1
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def deduplicate_objects_batch(
        self,
        all_objects: List[GraphObject],
    ) -> tuple[List[GraphObject], Dict[str, str]]:
        """Deduplicate all objects in batch using first-come canonical strategy.
        
        Processes objects in order (by chapter index), treating the first
        occurrence as canonical. Later duplicates are merged into the canonical.
        
        Args:
            all_objects: All objects from all chapters (with embeddings)
            
        Returns:
            Tuple of (deduplicated_objects, id_mapping) where id_mapping maps
            original object IDs to canonical object IDs
        """
        # Sort by chapter index to ensure consistent canonical selection
        sorted_objects = sorted(
            all_objects,
            key=lambda o: (o.metadata or {}).get("source_chapter", 0)
        )
        
        canonical_objects: Dict[str, GraphObject] = {}
        canonical_embeddings: Dict[str, List[float]] = {}
        id_mapping: Dict[str, str] = {}  # original_id -> canonical_id
        
        for obj in sorted_objects:
            if not obj.embedding_vector:
                # No embedding - treat as unique
                canonical_objects[obj.id] = obj
                id_mapping[obj.id] = obj.id
                continue
            
            # Find best matching canonical object
            best_match_id: Optional[str] = None
            highest_similarity = 0.0
            
            for canonical_id, canonical_embedding in canonical_embeddings.items():
                similarity = self._cosine_similarity(
                    obj.embedding_vector,
                    canonical_embedding,
                )
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    if similarity >= self.similarity_threshold:
                        best_match_id = canonical_id
            
            if best_match_id:
                # Merge into canonical object
                canonical_obj = canonical_objects[best_match_id]
                self._merge_object_info(canonical_obj, obj)
                id_mapping[obj.id] = best_match_id
                logger.debug(
                    f"Batch dedup: Merged '{obj.name}' into '{canonical_obj.name}' "
                    f"(similarity: {highest_similarity:.2f})"
                )
            else:
                # New canonical object
                canonical_objects[obj.id] = obj
                canonical_embeddings[obj.id] = obj.embedding_vector
                id_mapping[obj.id] = obj.id
        
        unique_objects = list(canonical_objects.values())
        logger.info(
            f"Batch deduplication: {len(all_objects)} objects -> {len(unique_objects)} unique"
        )
        
        return unique_objects, id_mapping
