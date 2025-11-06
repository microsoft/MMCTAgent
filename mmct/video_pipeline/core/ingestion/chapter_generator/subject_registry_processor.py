import json
import uuid
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel, Field
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from mmct.providers.search_index_schema import create_subject_registry_index_schema
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, SubjectResponse
from mmct.video_pipeline.core.ingestion.chapter_generator.video_summary import VideoSummary
from mmct.video_pipeline.core.ingestion.chapter_generator.utils import create_embedding


class MergedSubjectRegistryResponse(BaseModel):
    """
    Response model for merged subject registry.
    """
    model_config = {"extra": "forbid"}

    merged_subjects: Optional[List[SubjectResponse]] = Field(
        default_factory=list,
        description="List of SubjectResponse objects containing name, appearance, identity, first_seen timestamp, and additional_details"
    )


class SubjectRegistryProcessor:
    """
    Processes and merges subject registries from multiple video chapters.

    This class combines subject registries from different chapters of the same video,
    merges duplicate subjects, and indexes them for search.
    """

    def __init__(self, index_name: str):
        """
        Initialize the SubjectRegistryProcessor.

        Args:
            index_name: Name of the search index to use for subject registry storage
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
    ) -> Optional[List[SubjectResponse]]:
        """
        Main method to process chapter responses and create merged subject registry and video summary.

        Args:
            chapter_responses: List of ChapterCreationResponse objects containing subject registries
            video_id: Unique identifier for the video
            url: Optional URL of the video
            video_duration: Duration of the video in seconds

        Returns:
            Merged subject registry as a list of SubjectResponse objects, or None if no subjects found
        """
        # Extract all subject registries from chapters
        registries = self._extract_registries(chapter_responses)

        if not registries:
            logger.info("No subject registries found in chapters")
            merged_registry = None
        else:
            # Merge registries using LLM
            merged_registry = await self._merge_registries(registries)

            if not merged_registry:
                logger.warning("Failed to merge subject registries")

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
    ) -> List[List[SubjectResponse]]:
        """
        Extract subject registries from chapter responses.

        Args:
            chapter_responses: List of ChapterCreationResponse objects

        Returns:
            List of subject registry lists (List[SubjectResponse])
        """
        registries = []

        for idx, chapter in enumerate(chapter_responses):
            if chapter.subject_registry:
                registries.append(chapter.subject_registry)
                logger.debug(f"Extracted registry from chapter {idx}: {len(chapter.subject_registry)} subjects")

        logger.info(f"Extracted {len(registries)} non-empty registries from {len(chapter_responses)} chapters")
        return registries

    async def _merge_registries(self, registries: List[List[SubjectResponse]]) -> Optional[List[SubjectResponse]]:
        """
        Merge multiple subject registries using LLM to handle duplicates.
        Uses batch processing for large numbers of registries.

        Args:
            registries: List of subject registry lists (List[SubjectResponse])

        Returns:
            Merged subject registry as a list of SubjectResponse objects
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
        registries: List[List[SubjectResponse]],
        batch_size: int = 3
    ) -> Optional[List[SubjectResponse]]:
        """
        Merge subject registries in batches, passing the result of the previous batch
        to maintain cohesion across the entire merge process.

        Args:
            registries: List of subject registry lists (List[SubjectResponse])
            batch_size: Number of registries to process at once (default: 3)

        Returns:
            Merged subject registry as a list of SubjectResponse objects
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
                accumulated_merged_registry = await self._merge_and_enrich_subjects(
                    current_batch,
                    prev_merged_registry=None
                )
            else:
                # For subsequent batches, pass the accumulated result as previous context
                # This ensures cohesion by passing context forward
                accumulated_merged_registry = await self._merge_and_enrich_subjects(
                    current_batch,
                    prev_merged_registry=accumulated_merged_registry
                )
        
        if accumulated_merged_registry:
            logger.info(f"Final merged registry contains {len(accumulated_merged_registry)} subjects")
        else:
            logger.warning("No subjects found after batch merging")
        
        return accumulated_merged_registry

    async def _merge_and_enrich_subjects(
        self,
        current_registries: List[List[SubjectResponse]],
        prev_merged_registry: Optional[List[SubjectResponse]] = None
    ) -> Optional[List[SubjectResponse]]:
        """
        Perform a dedicated LLM call to merge and enrich subject registries,
        ensuring exhaustive extraction of all subjects with detailed attributes.

        Args:
            current_registries: List of subject registry lists to merge in this batch
            prev_merged_registry: Optional list of previously merged subjects from earlier batches

        Returns:
            Merged subject registry as a list of SubjectResponse objects
        """
        logger.info(f"Performing dedicated subject registry merge and enrichment for {len(current_registries)} registries...")
        
        # Prepare all subject registries
        all_subjects = []
        has_previous_context = prev_merged_registry is not None and len(prev_merged_registry) > 0
        
        # If we have previous merged results, add them first
        if has_previous_context:
            all_subjects.append({
                'batch_number': 'Previous Merged Results',
                'subjects': [subject.model_dump() for subject in prev_merged_registry]
            })
        
        # Add all current registries
        for i, registry in enumerate(current_registries):
            if registry:
                all_subjects.append({
                    'batch_number': i + 1,
                    'subjects': [subject.model_dump() for subject in registry]
                })
        
        if not all_subjects:
            logger.info("No subjects found in any registry, skipping merge")
            return None
        
        # Adjust merge prompt based on whether we have previous context
        context_instruction = ""
        if has_previous_context:
            context_instruction = """
            NOTE: The first batch contains PREVIOUSLY MERGED subjects from earlier chapters.
            Your task is to:
            1. Keep ALL subjects from the previous merged results
            2. Add any NEW subjects from the new chapter registries
            3. If a subject from new registries matches one in previous results, MERGE their attributes intelligently
            4. Maintain cohesion by ensuring the final registry is consistent and comprehensive
            """
        
        system_prompt = f"""You are a SubjectMergerGPT specialized in creating exhaustive, detailed subject registries.
        Your task is to merge subject information from multiple video chapters into a single comprehensive registry.
        {context_instruction}
        
        MERGING RULES:
        1. EXTRACT ALL SUBJECTS: Include every person, object, animal, item, or entity mentioned across all registries
        2. IDENTIFY DUPLICATES: Recognize when the same subject appears in multiple chapters (same name, similar descriptions)
        3. MERGE DUPLICATES INTELLIGENTLY:
           - Choose the most descriptive or complete name (prefer specific names over generic ones)
           - Combine all unique appearance descriptions (remove exact duplicates but keep variations)
           - Combine all unique identity descriptions (remove exact duplicates but keep variations)
           - Keep the EARLIEST first_seen timestamp
           - Merge additional_details into a comprehensive, non-redundant description
        4. ENRICH ATTRIBUTES: For each subject, ensure maximum detail:
           - People: clothing colors/styles/patterns, accessories (glasses, jewelry, hats), physical features (hair color/style, height, build), roles, activities
           - Objects: colors, sizes, brands, models, materials, conditions, positions, purposes, quantities
           - Animals: species, breeds, colors, markings, sizes, behaviors, conditions
           - Vehicles: make, model, color, type, distinctive features, license plates
           - Text/Signs: exact text content, location, context, purpose
        5. CONSISTENT NAMING: Assign clear, descriptive names (e.g., "Person in blue shirt", "Red Toyota Camry", "iPhone 15 Pro", "Welcome sign")
        6. COMPLETENESS: Don't drop any subject even if it seems minor or appears in only one chapter
        
        OUTPUT: Return a list of merged subjects with these fields:
        - name: string (the subject's name or identifier)
        - appearance: list of strings (visual characteristics)
        - identity: list of strings (type, category, role, etc.)
        - first_seen: float (timestamp in seconds when subject first appeared)
        - additional_details: string or null (any extra context)
        
        CRITICAL: Be EXHAUSTIVE and DETAILED. Preserve all unique subjects while intelligently merging duplicates.
        """
        
        registries_json = json.dumps(all_subjects, indent=2)
        
        user_prompt = f"""Here are the subject registries from {len(all_subjects)} different video chapters to merge:

{registries_json}

Please create a single, exhaustive, merged subject registry that includes ALL subjects with detailed attributes.
Carefully identify duplicate subjects that refer to the same entity and merge them according to the rules."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=0.0,
                response_format=MergedSubjectRegistryResponse,
            )
            
            # Extract the parsed response
            merged_response: MergedSubjectRegistryResponse = result['content']
            merged_registry = merged_response.merged_subjects
            
            if merged_registry:
                logger.info(f"Subject merge complete: {len(merged_registry)} subjects in merged registry")
            else:
                logger.warning("Subject merge returned empty registry")
            
            return merged_registry
                
        except Exception as e:
            logger.error(f"Error during subject merge: {e}. Returning None")
            return None

    async def _index_registry(
        self,
        registry: Optional[List[SubjectResponse]],
        video_id: str,
        url: Optional[str] = None,
        video_summary: str = "",
        video_duration: Optional[float] = None
    ) -> bool:
        """
        Index the merged subject registry, video summary, and video duration into search index as a single combined document.

        Args:
            registry: Merged subject registry as a list of SubjectResponse objects (can be None)
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
                logger.info(f"Creating subject registry index '{self.index_name}'...")
                # Create index schema for subject registry
                await self._create_subject_registry_index()

            # Serialize the entire merged subject_registry to JSON string
            subject_registry_json = "[]"
            subject_count = 0
            if registry:
                try:
                    # Convert the List[SubjectResponse] to JSON-serializable list
                    subject_registry_list = [subject.model_dump() for subject in registry]
                    subject_registry_json = json.dumps(subject_registry_list)
                    subject_count = len(registry)
                except Exception as e:
                    logger.warning(f"Failed to serialize merged subject_registry: {e}")
                    subject_registry_json = "[]"

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

            # Create a single document with the combined subject registry, video summary, embedding, and duration
            doc = {
                "id": str(uuid.uuid4()),
                "video_id": video_id,
                "url": url,
                "subject_registry": subject_registry_json,
                "subject_count": subject_count,
                "video_summary": video_summary,
                "video_summary_embedding": video_summary_embedding,
                "video_duration": video_duration if video_duration is not None else 0.0
            }

            # Index the single combined document
            await self.search_provider.upload_documents(
                documents=[doc],
                index_name=self.index_name
            )
            logger.info(f"Successfully indexed combined subject registry with {subject_count} subjects, video summary, embedding, and duration ({video_duration}s) for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index subject registry: {e}")
            return False

    async def _create_subject_registry_index(self):
        """Create the search index for subject registry if it doesn't exist."""
        try:
            # Use the schema utility function from search_index_schema.py
            index_schema = create_subject_registry_index_schema(self.index_name)

            created = await self.search_provider.create_index(self.index_name, index_schema)
            if created:
                logger.info(f"Subject registry index '{self.index_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to create subject registry index: {e}")
            raise
