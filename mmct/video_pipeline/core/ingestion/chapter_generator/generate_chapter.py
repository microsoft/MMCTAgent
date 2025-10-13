import os
import re
import base64
from datetime import time
from typing import List, Dict, Any, Union, Optional
from PIL import Image
from mmct.llm_client import LLMClient
from mmct.video_pipeline.core.ingestion.models import (
    ChapterCreationResponse,
    SubjectVarietyResponse,
)
from azure.identity import AzureCliCredential, DefaultAzureCredential
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.video_pipeline.utils.helper import create_stacked_frames_base64
from azure.search.documents.aio import SearchClient
from loguru import logger
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class ChapterGeneration:
    def __init__(self, keyframe_index, frame_stacking_grid_size=4):
        self.llm_client = LLMClient(
            service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True
        ).get_client()
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.index_client = SearchClient(
            endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=keyframe_index,
            credential=self._get_credential()
        )

    def _get_credential(self):
        try:
            cli_credential = AzureCliCredential()
            cli_credential.get_token("https://search.azure.com/.default")
            return cli_credential
        except Exception:
            return DefaultAzureCredential()

    async def _get_frames(self, transcript_seg:str, video_id: str) -> List[str]:
        """
        Fetch and load video frames for a transcript segment.

        Args:
            transcript_seg (str): Transcript segment with timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm text).
            video_id (str): Unique video identifier.

        Returns:
            List[str]: List of base64 encoded frame images.
        """
        frames_file_name = []
        match = re.search(r'(\d{2}:\d{2}:\d{2}),\d+\s*-->\s*(\d{2}:\d{2}:\d{2}),\d+', transcript_seg)
        start_time, end_time = match.groups()
        # Convert string timestamps to time objects if needed
        if isinstance(start_time, str):
            start_time = time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = time.fromisoformat(end_time)

        # Convert time objects to seconds
        start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

        time_filter = f"timestamp_seconds ge {start_seconds} and timestamp_seconds le {end_seconds}"
        video_filter = f"video_id eq '{video_id}'"
        combined_filter = f"{time_filter} and {video_filter}"

        results = await self.index_client.search(
        search_text="*",
        filter=combined_filter,
        order_by=["created_at asc"])

        async for result in results:
            frames_file_name.append(result['keyframe_filename'])

        # Load the keyframes from local
        base_dir = await get_media_folder()
        frames_file_paths = [os.path.join(base_dir, "keyframes", video_id, fname) for fname in frames_file_name]

        # Load and convert to base64 directly
        base64_frames = []
        for fpath in frames_file_paths:
            if os.path.exists(fpath):
                with open(fpath, "rb") as img_file:
                    base64_frames.append(base64.b64encode(img_file.read()).decode("utf-8"))

        return base64_frames
        
    async def subject_and_variety(self, transcript: str) -> str:
        """
        Extract subject and variety information from a video transcript using an AI model.

        Args:
            transcript (str): The text transcription of the video.

        Returns:
            str: A JSON-formatted string containing subject and variety information, or error details.
        """
        try:
            system_prompt = f"""
            You are a TranscriptAnalyzerGPT. Your job is to find all the details from the transcripts of every 2 seconds and from the audio.
            Mention only the English name or the text into the response. If the text mentioned in the video is in Hindi or any other language, then convert it into English.
            If any text from transcript is in Hindi or any other language, translate it into English and include it in the response.
            Topics to include in the response:
            1. Main subject or item being discussed in the video.
            2. Specific variety or type of the subject (e.g., model numbers, versions, specific types) discussed.
            If the transcript does not contain any specific subject or variety, assign 'None'.
            Ensure the response language is only English, not Hinglish or Hindi or any other language.
            Include the English-translated name of subjects and their variety only if certain.
            """

            prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The audio transcription is: {transcript}",
                        }
                    ],
                },
            ]

            response = await self.llm_client.beta.chat.completions.parse(
                model=os.getenv(
                    "LLM_MODEL_NAME"
                    if os.getenv("LLM_PROVIDER") == "azure"
                    else "OPENAI_MODEL_NAME"
                ),
                messages=prompt,
                temperature=0,
                response_format=SubjectVarietyResponse,
            )
            # Get the parsed Pydantic model from the response
            parsed_response: SubjectVarietyResponse = response.choices[0].message.parsed
            # Return the model as JSON string
            return parsed_response.model_dump_json()
        except Exception as e:
            return SubjectVarietyResponse(
                subject="None", variety_of_subject="None"
            ).model_dump_json()

    async def Chapters_creation(
        self,
        transcript: str,
        video_id: str,
        categories: str,
        subject_variety: str,
    ) -> ChapterCreationResponse:
        """
        Extract chapter information from video frames and transcript.

        Args:
            transcript (str): The video transcript text with timestamps
            video_id (str): Unique video identifier
            categories (str): Category and subcategory information in JSON format
            subject_variety (str): Subject and variety information in JSON format

        Returns:
            ChapterCreationResponse: A Pydantic model instance containing chapter information
        """
        try:
            frames = await self._get_frames(transcript, video_id)
            # Apply frame stacking if enabled (grid_size > 1)
            if self.frame_stacking_grid_size > 1 and len(frames) > self.frame_stacking_grid_size:
                logger.info(f"Applying frame stacking with grid_size={self.frame_stacking_grid_size}")
                processed_frames = await create_stacked_frames_base64(
                    frames, 
                    grid_size=self.frame_stacking_grid_size,
                    enable_stacking=True
                )
            else:
                processed_frames = frames
                logger.info("Frame stacking disabled or insufficient frames for stacking")
            # Add frame stacking information to system prompt if enabled
            frame_stacking_info = ""
            if self.frame_stacking_grid_size > 1 and len(processed_frames) < len(frames):
                frame_stacking_info = f"""
            NOTE: The video frames have been stacked in grids of {self.frame_stacking_grid_size} frames per image to optimize processing. Each image shows multiple sequential frames arranged in a grid format. Analyze all frames within each grid image to understand the temporal progression of the video content.
            """
            
            system_prompt = f""" 
            You are a VideoAnalyzerGPT. Your job is to find all the details from the video frames of every 2 seconds and from the audio.{frame_stacking_info}
            Below is the category and sub-category information for the provided transcript: 
                Category and sub category: {categories}
            Below are the main subjects and varieties mentioned, if you find other subjects or varieties then add them with comma separation:
                Subject and specific variety is {subject_variety}.
            Mention only the English name or text in the response, if text mentioned in the video is in Hindi or any other language then convert them into English.
            If any text from video frames or transcript is in Hindi or any other language then translate them into English and include it in the response.
            Topics that you have to find and include in the response:
            1. Topic of the video or scene theme.
            2. Main subject or item being discussed in the video.
            3. Specific variety or type of the subject (e.g. model numbers, versions, types, etc.) that is being discussed.
            4. A detailed summary which contains all the information discussed and analyzed from the frames.
            5. Actions taken in the video.
            6. Text from the images and scenes.
            Make sure the response language is only English, not Hinglish or Hindi or any other language.
            """

            # Handle large inputs by batching only frames, sending full transcript each time
            MAX_FRAMES_PER_BATCH = 20

            # Process frames in batches if needed, always sending full transcript
            if len(processed_frames) > MAX_FRAMES_PER_BATCH:

                # Split frames into batches
                frame_batches = [
                    processed_frames[i : i + MAX_FRAMES_PER_BATCH]
                    for i in range(0, len(processed_frames), MAX_FRAMES_PER_BATCH)
                ]

                results = []
                previous_analysis = ""

                # Process each batch
                for i, batch in enumerate(frame_batches):
                    # First batch uses standard prompt
                    if i == 0:
                        batch_prompt = [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    *map(
                                        lambda x: {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpg;base64,{x}",
                                                "detail": "high",
                                            },
                                        },
                                        batch,
                                    ),
                                    {
                                        "type": "text",
                                        "text": f"The audio transcription is: {transcript}",
                                    },
                                ],
                            },
                        ]
                    else:
                        # For subsequent batches, include context from previous results
                        context = f"""You've already analyzed the first {i * MAX_FRAMES_PER_BATCH} frames of this video. 
                        These are frames {i * MAX_FRAMES_PER_BATCH + 1} to {min((i + 1) * MAX_FRAMES_PER_BATCH, len(processed_frames))}.
                        
                        Previous analysis results: {previous_analysis}
                        
                        Continue your analysis with these additional frames, focusing on new information not captured in previous analyses.
                        Maintain consistency with your previous analysis for the same elements (subject, variety, etc.) unless new visual evidence contradicts it.
                        Pay special attention to any text, actions, or visual elements that appear in these new frames."""

                        batch_prompt = [
                            {"role": "system", "content": system_prompt + "\n\n" + context},
                            {
                                "role": "user",
                                "content": [
                                    *map(
                                        lambda x: {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpg;base64,{x}",
                                                "detail": "high",
                                            },
                                        },
                                        batch,
                                    ),
                                    {
                                        "type": "text",
                                        "text": f"The audio transcription is: {transcript}",
                                    },
                                ],
                            },
                        ]

                    try:
                        batch_response = await self.llm_client.beta.chat.completions.parse(
                            model=os.getenv(
                                "LLM_MODEL_NAME"
                                if os.getenv("LLM_PROVIDER") == "azure"
                                else "OPENAI_MODEL_NAME"
                            ),
                            messages=batch_prompt,
                            temperature=0,
                            response_format=ChapterCreationResponse,
                        )

                        batch_result: ChapterCreationResponse = batch_response.choices[
                            0
                        ].message.parsed
                        results.append(batch_result)
                        logger.info(f"single batch result:{batch_result}")
                        # Update previous analyses for next batch
                        previous_analysis = batch_result

                    except Exception as e:
                        print(f"Error processing frame batch {i+1}: {e}")
                        # Continue with other batches even if one fails
                logger.info(f"batch results:{results}")
                # Combine the results from all batches
                if len(results) > 1:
                    # Create a summary prompt to combine all results
                    summary_prompt = [
                        {
                            "role": "system",
                            "content": f"""You are tasked with combining multiple analyses of the same video into a single coherent analysis.
                            Below you'll find analyses from different frame batches of the same video.
                            Create a single comprehensive JSON that combines all the information without redundancy.
                            
                            When integrating information:
                            1. For factual fields (topic, subject, variety), use the most detailed and accurate version
                            2. For summary fields, synthesize all information into a cohesive narrative
                            3. For actions and text from scene, include all unique observations across analyses
                            """,
                        },
                        {
                            "role": "user",
                            "content": f"Here are the analyses from different frame batches to combine:\n\n"
                            + "\n\n".join(
                                [f"Batch {i+1}:\n{result}" for i, result in enumerate(results)]
                            ),
                        },
                    ]

                    combined_response = await self.llm_client.beta.chat.completions.parse(
                        model=os.getenv(
                            "LLM_MODEL_NAME"
                            if os.getenv("LLM_PROVIDER") == "azure"
                            else "OPENAI_MODEL_NAME"
                        ),
                        messages=summary_prompt,
                        temperature=0,
                        response_format=ChapterCreationResponse,
                    )

                    logger.info(f"combined batch response:{combined_response}")
                    final_result: ChapterCreationResponse = combined_response.choices[
                        0
                    ].message.parsed

                else:
                    final_result: ChapterCreationResponse = results[0]

                # Return ChapterCreationResponse instance directly
                return final_result
            # Original implementation for smaller inputs
            prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        *map(
                            lambda x: {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpg;base64,{x}",
                                    "detail": "high",
                                },
                            },
                            processed_frames,
                        ),
                        {"type": "text", "text": f"The audio transcription is: {transcript}"},
                    ],
                },
            ]

            response = await self.llm_client.beta.chat.completions.parse(
                model=os.getenv(
                    "LLM_MODEL_NAME"
                    if os.getenv("LLM_PROVIDER") == "azure"
                    else "OPENAI_MODEL_NAME"
                ),
                messages=prompt,
                temperature=0,
                response_format=ChapterCreationResponse,
            )

            response_object: ChapterCreationResponse = response.choices[0].message.parsed

            # Return ChapterCreationResponse instance directly
            return response_object
        except Exception as e:
            logger.exception(f"Error Creating chapters: {e}")
            raise
