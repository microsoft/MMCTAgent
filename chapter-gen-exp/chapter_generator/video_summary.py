import json
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel, Field
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse


class MergedVideoSummaryResponse(BaseModel):
    """Response model for merged video summary.

    Combines all chapter summaries into a single coherent video-level summary.
    """
    model_config = {"extra": "forbid"}

    video_summary: str = Field(
        ...,
        description="Comprehensive summary of the entire video, created by merging all chapter summaries into a coherent narrative"
    )


class VideoSummary:
    """
    Processes and merges chapter summaries to create an overall video summary.

    This class combines detailed summaries from different chapters of the same video
    into a single comprehensive video-level summary.
    """

    def __init__(self):
        """Initialize the VideoSummary processor."""
        self.config = MMCTConfig()
        self.llm_provider = provider_factory.create_llm_provider()

    async def create_video_summary(
        self,
        chapter_responses: List[ChapterCreationResponse]
    ) -> Optional[str]:
        """
        Main method to process chapter responses and create merged video summary.

        Args:
            chapter_responses: List of ChapterCreationResponse objects containing chapter summaries
                              (already sorted chronologically by chapter_generator.py line 203)

        Returns:
            Merged video summary as a string, or None if no summaries found
        """
        # Extract all summaries from chapters (already in chronological order from chapter_generator.py)
        summaries = self._extract_summaries(chapter_responses)

        if not summaries:
            logger.info("No chapter summaries found")
            return None

        # Merge summaries using LLM with batch processing
        merged_summary = await self._merge_summaries(summaries)

        if not merged_summary:
            logger.warning("Failed to merge chapter summaries")
            return None

        return merged_summary

    def _extract_summaries(
        self,
        chapter_responses: List[ChapterCreationResponse]
    ) -> List[str]:
        """
        Extract detailed summaries from chapter responses in order.

        Args:
            chapter_responses: List of ChapterCreationResponse objects (assumed to be in chronological order)

        Returns:
            List of chapter summary strings in chronological order
        """
        summaries = []

        for idx, chapter in enumerate(chapter_responses):
            if chapter.detailed_summary:
                summaries.append(chapter.detailed_summary)
                logger.debug(f"Extracted summary from chapter {idx}")

        logger.info(f"Extracted {len(summaries)} summaries from {len(chapter_responses)} chapters in chronological order")
        return summaries

    async def _merge_summaries(self, summaries: List[str]) -> Optional[str]:
        """
        Merge multiple chapter summaries using LLM with batch processing for large inputs.

        Args:
            summaries: List of chapter summary strings

        Returns:
            Merged video summary as a string
        """
        if len(summaries) == 0:
            return None

        if len(summaries) == 1:
            logger.info("Only one summary found, no merging needed")
            return summaries[0]

        try:
            MAX_SUMMARIES_PER_BATCH = 10

            # If summaries fit in one batch, process directly
            if len(summaries) <= MAX_SUMMARIES_PER_BATCH:
                logger.info(f"Processing {len(summaries)} summaries in a single batch")
                return await self._process_summary_batch(summaries, None)

            # Split into batches and process iteratively
            logger.info(f"Processing {len(summaries)} summaries in multiple batches of {MAX_SUMMARIES_PER_BATCH}")

            summary_batches = [
                summaries[i:i + MAX_SUMMARIES_PER_BATCH]
                for i in range(0, len(summaries), MAX_SUMMARIES_PER_BATCH)
            ]

            previous_analysis = None

            for i, batch in enumerate(summary_batches):
                logger.info(f"Processing batch {i + 1}/{len(summary_batches)} with {len(batch)} summaries")

                batch_result = await self._process_summary_batch(batch, previous_analysis)

                if not batch_result:
                    logger.error(f"Failed to process batch {i + 1}")
                    return None

                previous_analysis = batch_result
                logger.info(f"Completed batch {i + 1}/{len(summary_batches)}")

            logger.info(f"Successfully merged all {len(summaries)} summaries into video summary")
            return previous_analysis

        except Exception as e:
            logger.error(f"Failed to merge chapter summaries: {e}")
            return None

    async def _process_summary_batch(
        self,
        summaries: List[str],
        previous_analysis: Optional[str]
    ) -> Optional[str]:
        """
        Process a single batch of summaries with LLM.

        Args:
            summaries: List of chapter summaries to merge in this batch
            previous_analysis: Previous merged summary from earlier batches (if any)

        Returns:
            Merged summary string for this batch
        """
        summaries_json = json.dumps(summaries, indent=2)

        system_prompt = """You are a VideoSummaryMergerGPT. Your task is to merge multiple chapter summaries from different segments of the same video into one comprehensive, coherent video-level summary.

        MERGE RULES:
        1. CREATE A COHESIVE NARRATIVE - Combine all chapter summaries into a single flowing narrative that describes the entire video
        2. PRESERVE KEY INFORMATION - Maintain all important details, topics, actions, and insights from all chapters
        3. ELIMINATE REDUNDANCY - Remove duplicate information that appears across multiple chapters
        4. MAINTAIN LOGICAL FLOW - Organize the information in a way that makes sense for understanding the whole video
        5. BE COMPREHENSIVE YET CONCISE - Include all significant points while avoiding unnecessary repetition
        6. FOCUS ON THE VIDEO'S PURPOSE - Highlight the main themes, objectives, and key takeaways of the entire video

        OUTPUT FORMAT:
        - Return a single comprehensive paragraph or multi-paragraph summary that captures the essence of the entire video
        - The summary should be detailed enough to give someone a complete understanding of what the video covers without watching it
        - Do NOT use bullet points or lists - create a narrative summary"""

        if previous_analysis:
            # For subsequent batches, include context from previous analysis
            context = f"""You've already analyzed earlier chapters of this video and created a partial summary.

            Previous summary: {previous_analysis}

            Now merge these additional chapter summaries with your previous analysis to create a more complete video summary.
            Maintain consistency with your previous analysis and integrate the new information seamlessly."""

            user_prompt = f"""{context}

            Additional chapter summaries to integrate:
            {summaries_json}

            Create a comprehensive, updated video summary that includes both the previous analysis and the new chapter information."""
        else:
            # First batch uses standard prompt
            user_prompt = f"""Merge these chapter summaries from the same video into a single comprehensive video summary:

            {summaries_json}

            Create a coherent, comprehensive summary that captures the essence of the entire video."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = await self.llm_provider.chat_completion(
            messages=messages,
            temperature=0.0,
            response_format=MergedVideoSummaryResponse,
        )

        # Extract the parsed response
        merged_response: MergedVideoSummaryResponse = result['content']
        return merged_response.video_summary
