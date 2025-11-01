from typing import List, Optional
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse
import os
import json
import aiofiles
import re
from dotenv import load_dotenv
from mmct.video_pipeline.utils.helper import get_media_folder

load_dotenv(override=True)

class MergeVisualSummaryWithTranscript:
    def __init__(
        self,
        chapter_responses: List[ChapterCreationResponse],
        transcripts: List[str],
        video_id: str,
        full_transcript_string: str,
    ):
        """
        Initialize the SummaryExtraction class with chapter responses and transcripts.

        Args:
            chapter_responses (List[ChapterCreationResponse]): List of ChapterCreationResponse objects
            transcripts (List[str]): List of transcript segments corresponding to each chapter
            video_id (str): The video identifier
            full_transcript_string (str): The complete transcript of the video
        """
        self.AZURE_STORAGE_ACCOUNT_URL = os.getenv("BLOB_ACCOUNT_URL")
        self.CONTAINER_NAME = os.getenv("VIDEO_DESCRIPTION_CONTAINER_NAME")
        self.merged = {}
        self.transcripts = transcripts
        self.chapters = chapter_responses
        self.video_id = video_id
        self.full_transcript_string = full_transcript_string
        self.chapters_result={}

    async def extract_timestamp(self, transcript_text: str) -> Optional[str]:
        """
        Extract timestamp from transcript text.

        Args:
            transcript_text (str): Transcript text containing timestamp information

        Returns:
            Optional[str]: Extracted timestamp or None if not found
        """
        if not transcript_text:
            return None

        match = re.search(r"\[(\d{1,2}:\d{2}:\d{2}\.\d{3,6})\]", transcript_text)
        if match:
            return match.group(1)

        match = re.search(
            r"(\d{1,2}:\d{2}:\d{2},\d{3})\s*-->?\s*(\d{1,2}:\d{2}:\d{2},\d{3})",
            transcript_text,
        )
        if match:
            return f"{match.group(1)} --> {match.group(2)}"

        return None

    async def _process_chapters(self):
        """Process chapter responses and transcripts to extract summaries and actions."""
        summaries = ""
        actions_taken = ""

        for chapter, transcript in zip(self.chapters, self.transcripts):
            # Extract relevant data directly from the ChapterCreationResponse object
            detailed_summary = chapter.Detailed_summary if chapter.Detailed_summary else ""
            action_taken = chapter.Action_taken if chapter.Action_taken else ""
            
            # Extract timestamp from the transcript
            timestamp = await self.extract_timestamp(transcript)

            if timestamp:
                timestamp = timestamp.replace("\n", "").replace("\r", "").replace(" ", "")
                summaries += f"\n[{timestamp}]\n{detailed_summary}\n"
                actions_taken += f"\n[{timestamp}]\n{action_taken}\n"

        self.chapters_result["transcript"] = self.full_transcript_string
        self.chapters_result["summaries"] = {
            "detailed_description": summaries,
            "Action_taken": actions_taken
        }
        
        json_data = json.dumps(self.chapters_result, indent=4)
        async with aiofiles.open(
            os.path.join(await get_media_folder(),f"{self.video_id}.json"), "w", encoding="utf-8"
        ) as f:
            await f.write(json_data)

    async def __call__(self) ->None:
        """Execute the summary extraction and upload process."""
        await self._process_chapters()
        print("Chapters processed & uploaded!")
