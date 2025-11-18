import json
import uuid
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel, Field
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from mmct.providers.search_index_schema import create_object_collection_index_schema
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, ObjectResponse
from mmct.video_pipeline.core.ingestion.chapter_generator.video_summary import VideoSummary
from mmct.video_pipeline.core.ingestion.chapter_generator.utils import create_embedding


class MergedObjectCollectionResponse(BaseModel):
    """
    Response model for merged object collection.
    """
    model_config = {"extra": "forbid"}

    merged_objects: Optional[List[ObjectResponse]] = Field(
        default_factory=list,
        description="List of ObjectResponse objects containing name, appearance, identity, first_seen timestamp, and additional_details"
    )


class ObjectCollectionProcessor:
    """
    Processes and merges object collections from multiple video chapters.

    This class combines object collections from different chapters of the same video,
    merges duplicate objects, and indexes them for search.
    """

    def __init__(self, index_name: str):
        """
        Initialize the ObjectCollectionProcessor.

        Args:
            index_name: Name of the search index to use for object collection storage
        """
        self.config = MMCTConfig()
        self.llm_provider = provider_factory.create_llm_provider()
        self.search_provider = provider_factory.create_search_provider()
        self.index_name = index_name
        self.video_summary_processor = VideoSummary()

    async def run(
        self,
        chapter_responses: List[ChapterCreationResponse],
        video_id: str,
        url: Optional[str] = None,
        video_duration: Optional[float] = None
    ) -> Optional[List[ObjectResponse]]:
        """
        Main method to process chapter responses and create merged object collection and video summary.

        Args:
            chapter_responses: List of ChapterCreationResponse objects containing object collections
            video_id: Unique identifier for the video
            url: Optional URL of the video
            video_duration: Duration of the video in seconds

        Returns:
            Merged object collection as a list of ObjectResponse objects, or None if no objects found
        """
        # Extract all object collections from chapters
        registries = self._extract_registries(chapter_responses)

        if not registries:
            logger.info("No object collections found in chapters")
            merged_registry = None
        else:
            # Merge registries using LLM
            merged_registry = await self._merge_registries(registries)

            if not merged_registry:
                logger.warning("Failed to merge object collections")

        # Create merged video summary from all chapter summaries
        # Note: chapter_responses are already sorted chronologically by chapter_generator.py
        video_summary = await self.video_summary_processor.create_video_summary(
            chapter_responses=chapter_responses
        )

        if not video_summary:
            logger.warning("Failed to create video summary")
            video_summary = ""

        # Index the merged registry, video summary, and video duration
        await self._index_registry(merged_registry, video_id, url, video_summary, video_duration)

        return merged_registry

    def _extract_registries(
        self,
        chapter_responses: List[ChapterCreationResponse]
    ) -> List[List[ObjectResponse]]:
        """
        Extract object collections from chapter responses.

        Args:
            chapter_responses: List of ChapterCreationResponse objects

        Returns:
            List of object collection lists (List[ObjectResponse])
        """
        registries = []

        for idx, chapter in enumerate(chapter_responses):
            if chapter.object_collection:
                registries.append(chapter.object_collection)
                logger.debug(f"Extracted collection from chapter {idx}: {len(chapter.object_collection)} objects")

        logger.info(f"Extracted {len(registries)} non-empty collections from {len(chapter_responses)} chapters")
        return registries

    async def _merge_registries(self, registries: List[List[ObjectResponse]]) -> Optional[List[ObjectResponse]]:
        """
        Merge multiple object collections using LLM to handle duplicates.
        Uses batch processing for large numbers of collections.

        Args:
            registries: List of object collection lists (List[ObjectResponse])

        Returns:
            Merged object collection as a list of ObjectResponse objects
        """
        if len(registries) == 0:
            return None

        if len(registries) == 1:
            logger.info("Only one registry found, no merging needed")
            return registries[0]

        # Use batch merging for multiple registries
        return await self._merge_registries_in_batches(registries, batch_size=3)

    async def _merge_registries_in_batches(
        self,
        registries: List[List[ObjectResponse]],
        batch_size: int = 3
    ) -> Optional[List[ObjectResponse]]:
        """
        Merge object collections in batches, passing the result of the previous batch
        to maintain cohesion across the entire merge process.

        Args:
            registries: List of object collection lists (List[ObjectResponse])
            batch_size: Number of collections to process at once (default: 3)

        Returns:
            Merged object collection as a list of ObjectResponse objects
        """
        logger.info(f"Starting registry merge in batches of {batch_size}...")
        
        # Split registries into groups of batch_size
        registry_batches = [
            registries[i:i + batch_size]
            for i in range(0, len(registries), batch_size)
        ]
        
        logger.info(f"Processing {len(registries)} registries in {len(registry_batches)} merge batches")
        
        # Track the accumulated merged result
        accumulated_merged_registry = None
        
        # Process each batch
        for batch_idx, current_batch in enumerate(registry_batches):
            logger.info(f"Processing merge batch {batch_idx + 1}/{len(registry_batches)} with {len(current_batch)} registries")
            
            # For the first batch, merge without prior context
            if accumulated_merged_registry is None:
                accumulated_merged_registry = await self._merge_and_enrich_objects(
                    current_batch,
                    prev_merged_registry=None
                )
            else:
                # For subsequent batches, pass the accumulated result as previous context
                # This ensures cohesion by passing context forward
                accumulated_merged_registry = await self._merge_and_enrich_objects(
                    current_batch,
                    prev_merged_registry=accumulated_merged_registry
                )
        
        if accumulated_merged_registry:
            logger.info(f"Final merged collection contains {len(accumulated_merged_registry)} objects")
        else:
            logger.warning("No objects found after batch merging")

        return accumulated_merged_registry

    async def _merge_and_enrich_objects(
        self,
        current_registries: List[List[ObjectResponse]],
        prev_merged_registry: Optional[List[ObjectResponse]] = None
    ) -> Optional[List[ObjectResponse]]:
        """
        Perform a dedicated LLM call to merge and enrich object collections,
        ensuring exhaustive extraction of all objects with detailed attributes.

        Args:
            current_registries: List of object collection lists to merge in this batch
            prev_merged_registry: Optional list of previously merged objects from earlier batches

        Returns:
            Merged object collection as a list of ObjectResponse objects
        """
        logger.info(f"Performing dedicated object collection merge and enrichment for {len(current_registries)} collections...")
        
        # Prepare all object collections
        all_objects = []
        has_previous_context = prev_merged_registry is not None and len(prev_merged_registry) > 0

        # If we have previous merged results, add them first
        if has_previous_context:
            all_objects.append({
                'batch_number': 'Previous Merged Results',
                'objects': [obj.model_dump() for obj in prev_merged_registry]
            })

        # Add all current registries
        for i, registry in enumerate(current_registries):
            if registry:
                all_objects.append({
                    'batch_number': i + 1,
                    'objects': [obj.model_dump() for obj in registry]
                })

        if not all_objects:
            logger.info("No objects found in any collection, skipping merge")
            return None
        
        # Adjust merge prompt based on whether we have previous context
        context_instruction = ""
        if has_previous_context:
            context_instruction = """
            NOTE: The first batch contains PREVIOUSLY MERGED objects from earlier chapters.
            Your task is to:
            1. Keep ALL objects from the previous merged results
            2. Add any NEW objects from the new chapter collections
            3. If an object from new collections matches one in previous results, MERGE their attributes intelligently
            4. Maintain cohesion by ensuring the final collection is consistent and comprehensive
            """

        system_prompt = f"""You are an ObjectMergerGPT specialized in creating exhaustive, detailed object collections.
        Your task is to merge object information from multiple video chapters into a single comprehensive collection.
        {context_instruction}

        MERGING RULES:
        1. EXTRACT ALL OBJECTS: Include every person, object, animal, item, or entity mentioned across all collections
        2. IDENTIFY DUPLICATES: Recognize when the same object appears in multiple chapters (same name, similar descriptions)
        3. MERGE DUPLICATES INTELLIGENTLY:
           - Choose the most descriptive or complete name (prefer specific names over generic ones)
           - Combine all unique appearance descriptions (remove exact duplicates but keep variations)
           - Combine all unique identity descriptions (remove exact duplicates but keep variations)
           - Keep the EARLIEST first_seen timestamp
           - Merge additional_details into a comprehensive, non-redundant description
        4. ENRICH ATTRIBUTES: For each object, ensure maximum detail:
           - People: clothing colors/styles/patterns, accessories (glasses, jewelry, hats), physical features (hair color/style, height, build), roles, activities
           - Objects: colors, sizes, brands, models, materials, conditions, positions, purposes, quantities
           - Animals: species, breeds, colors, markings, sizes, behaviors, conditions
           - Vehicles: make, model, color, type, distinctive features, license plates
           - Text/Signs: exact text content, location, context, purpose
        5. CONSISTENT NAMING: Assign clear, descriptive names (e.g., "Person in blue shirt", "Red Toyota Camry", "iPhone 15 Pro", "Welcome sign")
        6. COMPLETENESS: Don't drop any object even if it seems minor or appears in only one chapter

        OUTPUT: Return a list of merged objects with these fields:
        - name: string (the object's name or identifier)
        - appearance: list of strings (visual characteristics)
        - identity: list of strings (type, category, role, etc.)
        - first_seen: float (timestamp in seconds when object first appeared)
        - additional_details: string or null (any extra context)

        CRITICAL: Be EXHAUSTIVE and DETAILED. Preserve all unique objects while intelligently merging duplicates.
        """

        registries_json = json.dumps(all_objects, indent=2)
        
        user_prompt = f"""Here are the object collections from {len(all_objects)} different video chapters to merge:

{registries_json}

Please create a single, exhaustive, merged object collection that includes ALL objects with detailed attributes.
Carefully identify duplicate objects that refer to the same entity and merge them according to the rules."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=0.0,
                response_format=MergedObjectCollectionResponse,
            )

            # Extract the parsed response
            merged_response: MergedObjectCollectionResponse = result['content']
            merged_registry = merged_response.merged_objects

            if merged_registry:
                logger.info(f"Object merge complete: {len(merged_registry)} objects in merged collection")
            else:
                logger.warning("Object merge returned empty collection")

            return merged_registry

        except Exception as e:
            logger.error(f"Error during object merge: {e}. Returning None")
            return None

    async def _index_registry(
        self,
        registry: Optional[List[ObjectResponse]],
        video_id: str,
        url: Optional[str] = None,
        video_summary: str = "",
        video_duration: Optional[float] = None
    ) -> bool:
        """
        Index the merged object collection, video summary, and video duration into search index as a single combined document.

        Args:
            registry: Merged object collection as a list of ObjectResponse objects (can be None)
            video_id: Unique identifier for the video
            url: Optional URL of the video
            video_summary: Overall summary of the entire video
            video_duration: Duration of the video in seconds

        Returns:
            True if indexing succeeded, False otherwise
        """
        try:
            # Check if index exists
            index_exists = await self.search_provider.index_exists(self.index_name)

            if not index_exists:
                logger.info(f"Creating object collection index '{self.index_name}'...")
                # Create index schema for object collection
                await self._create_object_collection_index()

            # Serialize the entire merged object_collection to JSON string
            object_collection_json = "[]"
            object_count = 0
            if registry:
                try:
                    # Convert the List[ObjectResponse] to JSON-serializable list
                    object_collection_list = [obj.model_dump() for obj in registry]
                    object_collection_json = json.dumps(object_collection_list)
                    object_count = len(registry)
                except Exception as e:
                    logger.warning(f"Failed to serialize merged object_collection: {e}")
                    object_collection_json = "[]"

            # Create embedding for video summary if it exists
            video_summary_embedding = []
            if video_summary:
                try:
                    logger.info("Creating embedding for video summary...")
                    video_summary_embedding = await create_embedding(video_summary)
                    logger.info(f"Successfully created video summary embedding with dimension {len(video_summary_embedding)}")
                except Exception as e:
                    logger.error(f"Failed to create video summary embedding: {e}")
                    video_summary_embedding = []

            # Create a single document with the combined object collection, video summary, embedding, and duration
            doc = {
                "id": str(uuid.uuid4()),
                "video_id": video_id,
                "url": url,
                "object_collection": object_collection_json,
                "object_count": object_count,
                "video_summary": video_summary,
                "video_summary_embedding": video_summary_embedding,
                "video_duration": video_duration if video_duration is not None else 0.0
            }

            # Index the single combined document
            await self.search_provider.upload_documents(
                documents=[doc],
                index_name=self.index_name
            )
            logger.info(f"Successfully indexed combined object collection with {object_count} objects, video summary, embedding, and duration ({video_duration}s) for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index object collection: {e}")
            return False

    async def _create_object_collection_index(self):
        """Create the search index for object collection if it doesn't exist."""
        try:
            # Use the schema utility function from search_index_schema.py
            index_schema = create_object_collection_index_schema(self.index_name)

            created = await self.search_provider.create_index(self.index_name, index_schema)
            if created:
                logger.info(f"Object collection index '{self.index_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to create object collection index: {e}")
            raise
