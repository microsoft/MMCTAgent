import json
import uuid
from typing import List, Dict, Optional
from loguru import logger
from pydantic import BaseModel, Field
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from mmct.providers.search_index_schema import create_subject_registry_index_schema
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, SubjectResponse


class MergedSubjectRegistryResponse(BaseModel):
    """Response model for merged subject registry.

    Uses properly typed SubjectResponse objects to satisfy Azure OpenAI's
    strict validation requirements for nested structures.
    """
    model_config = {"extra": "forbid"}

    merged_subjects: Optional[Dict[str, SubjectResponse]] = Field(
        default_factory=dict,
        description="Dictionary mapping subject names (keys) to SubjectResponse objects (values) containing name, appearance, identity, first_seen timestamp, and additional_details"
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

    async def run(
        self,
        chapter_responses: List[ChapterCreationResponse],
        video_id: str
    ) -> Optional[Dict[str, SubjectResponse]]:
        """
        Main method to process chapter responses and create merged subject registry.

        Args:
            chapter_responses: List of ChapterCreationResponse objects containing subject registries
            video_id: Unique identifier for the video

        Returns:
            Merged subject registry dictionary mapping subject names to SubjectResponse objects, or None if no subjects found
        """
        # Extract all subject registries from chapters
        registries = self._extract_registries(chapter_responses)

        if not registries:
            logger.info("No subject registries found in chapters")
            return None

        # Merge registries using LLM
        merged_registry = await self._merge_registries(registries)

        if not merged_registry:
            logger.warning("Failed to merge subject registries")
            return None

        # Index the merged registry
        await self._index_registry(merged_registry, video_id)

        return merged_registry

    def _extract_registries(
        self,
        chapter_responses: List[ChapterCreationResponse]
    ) -> List[Dict[str, SubjectResponse]]:
        """
        Extract subject registries from chapter responses.

        Args:
            chapter_responses: List of ChapterCreationResponse objects

        Returns:
            List of subject registry dictionaries (Dict[str, SubjectResponse])
        """
        registries = []

        for idx, chapter in enumerate(chapter_responses):
            if chapter.subject_registry:
                # Keep as Dict[str, SubjectResponse]
                if chapter.subject_registry:
                    registries.append(chapter.subject_registry)
                    logger.debug(f"Extracted registry from chapter {idx}: {len(chapter.subject_registry)} subjects")

        logger.info(f"Extracted {len(registries)} non-empty registries from {len(chapter_responses)} chapters")
        return registries

    async def _merge_registries(self, registries: List[Dict[str, SubjectResponse]]) -> Optional[Dict[str, SubjectResponse]]:
        """
        Merge multiple subject registries using LLM to handle duplicates.

        Args:
            registries: List of subject registry dictionaries (Dict[str, SubjectResponse])

        Returns:
            Merged subject registry dictionary (Dict[str, SubjectResponse])
        """
        if len(registries) == 0:
            return None

        if len(registries) == 1:
            logger.info("Only one registry found, no merging needed")
            return registries[0]

        try:
            # Convert Pydantic models to dicts for JSON serialization
            registries_dicts = []
            for registry in registries:
                registry_dict = {}
                for subject_name, subject_obj in registry.items():
                    registry_dict[subject_name] = subject_obj.model_dump()
                registries_dicts.append(registry_dict)

            registries_json = json.dumps(registries_dicts, indent=2)

            system_prompt = """You are a SubjectRegistryMergerGPT. Your task is to merge multiple partial subject registries from different clips of the same video into one coherent registry.

MERGE RULES:
1. PRESERVE ALL UNIQUE SUBJECTS - Do not lose any information or subjects from the input registries
2. IDENTIFY AND MERGE DUPLICATES - If two or more subjects clearly refer to the same entity (same person, object, or animal), merge them intelligently:
   - Choose the most descriptive or complete name (prefer specific names over generic ones)
   - Keep the EARLIEST `first_seen` timestamp across all occurrences
   - Combine ALL appearance descriptions from all instances (remove exact duplicates, but keep meaningful variations)
   - Combine ALL identity descriptions from all instances (remove exact duplicates, but keep meaningful variations)
   - Merge additional_details into a comprehensive, coherent description that includes all relevant information
3. MAINTAIN STRUCTURE - Each merged subject must have:
   - name: string (the subject's name or identifier)
   - appearance: list of strings (visual characteristics)
   - identity: list of strings (type, category, role, etc.)
   - first_seen: float (timestamp in seconds)
   - additional_details: string or null (any extra context)
4. OUTPUT FORMAT - Return a dictionary where keys are subject names and values are subject objects with the above fields

EXAMPLES OF MERGING:
- If "red car" appears at 10s and "red sports car" at 30s with similar descriptions → merge into "red sports car" with first_seen=10.0
- If "presenter" and "main host" clearly refer to the same person → merge into one with combined descriptions
- If "iPhone" and "smartphone" both refer to the same device → merge appropriately"""

            user_prompt = f"""Merge these partial subject registries from the same video into a single coherent registry:

{registries_json}

Carefully identify duplicate subjects that refer to the same entity and merge them according to the rules. Preserve all unique subjects."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            logger.info(f"Merging {len(registries)} subject registries using LLM...")

            result = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=0.0,
                response_format=MergedSubjectRegistryResponse,
            )

            # Extract the parsed response
            merged_response: MergedSubjectRegistryResponse = result['content']
            merged_registry = merged_response.merged_subjects

            logger.info(f"Successfully merged registries: {len(merged_registry)} unique subjects")
            return merged_registry

        except Exception as e:
            logger.error(f"Failed to merge subject registries: {e}")
            return None

    async def _index_registry(self, registry: Dict[str, SubjectResponse], video_id: str) -> bool:
        """
        Index the merged subject registry into search index.

        Args:
            registry: Merged subject registry dictionary (Dict[str, SubjectResponse])
            video_id: Unique identifier for the video

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

            # Prepare documents for indexing
            documents = []
            for subject_name, subject_obj in registry.items():
                # Create searchable document
                doc = {
                    "id":str(uuid.uuid4()),
                    "video_id": video_id,
                    "name": subject_obj.name,
                    "appearance": " | ".join(subject_obj.appearance),
                    "identity": " | ".join(subject_obj.identity),
                    "first_seen": subject_obj.first_seen,
                    "additional_details": subject_obj.additional_details
                }
                documents.append(doc)

            # Index documents
            if documents:
                await self.search_provider.upload_documents(
                    documents=documents,
                    index_name=self.index_name
                )
                logger.info(f"Successfully indexed {len(documents)} subjects for video {video_id}")
                return True
            else:
                logger.warning("No documents to index")
                return False

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
