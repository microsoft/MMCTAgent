import os
from typing import List, Dict, Any, Union, Optional
from mmct.llm_client import LLMClient
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, SpeciesVarietyResponse
from loguru import logger
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

class ChapterGeneration:
    def __init__(self):
        self.llm_client = LLMClient(
            service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True
        ).get_client()

    async def species_and_variety(self, transcript:str)->str:
        """
        Extract species and variety information from a video transcript using an AI model.

        Args:
            transcript (str): The text transcription of the video.

        Returns:
            str: A JSON-formatted string containing species and variety information, or error details.
        """
        try:
            system_prompt = f"""
            You are a TranscriptAnalyzerGPT. Your job is to find all the details from the transcripts of every 2 seconds and from the audio.
            Mention only the English name or the text into the response. If the text mentioned in the video is in Hindi or any other language, then convert it into English.
            If any text from transcript is in Hindi or any other language, translate it into English and include it in the response.
            Topics to include in the response:
            1. Species name talked about in the video.
            2. Specific variety of species (e.g., IPA 15-06, IPL 203, IPH 15-03) discussed.
            If the transcript does not contain any species or variety, assign 'None'.
            Ensure the response language is only English, not Hinglish or Hindi or any other language.
            Include the English-translated name of species and their variety only if certain.
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
                    "AZURE_OPENAI_MODEL"
                    if os.getenv("LLM_PROVIDER") == "azure"
                    else "OPENAI_MODEL"
                ),
                messages=prompt,
                temperature=0,
                response_format=SpeciesVarietyResponse,
            )
            # Get the parsed Pydantic model from the response
            parsed_response: SpeciesVarietyResponse = response.choices[0].message.parsed
            # Return the model as JSON string
            return parsed_response.model_dump_json()
        except Exception as e:
            return SpeciesVarietyResponse(
            species="None", Variety_of_species="None"
            ).model_dump_json()

    async def Chapters_creation(
        self,
        transcript: str,
        frames: List[str],
        categories: str,
        species_variety: str,
    ) -> ChapterCreationResponse:
        """
        Extract chapter information from video frames and transcript.

        Args:
            transcript (str): The video transcript text
            frames (List[str]): List of Base64 encoded frame images
            categories (str): Category and subcategory information in JSON format
            species_variety (str): Species and variety information in JSON format

        Returns:
            ChapterCreationResponse: A Pydantic model instance containing chapter information
        """
        try:
            system_prompt = f""" 
            You are a VideoAnalyzerGPT. Your job is to find all the details from the video frames of every 2 seconds and from the audio.
            below is the category and a sub category from which the provided transcript is belongs to in terms of agriculture: 
                Category and sub category: {categories}
            Below are the main species and the varity of species, If you found another species or varity of species then add the species and varity of species with the comma seprated:
                species and specific varity of species is {species_variety}.
            Mention only the english name or the text into the response, if the text is mention in the video is in hindi or any other language then convert them into english language.
            If any text from anywhere in video frames or transcript is in hindi or any other language then translate them into english and then include it into response.
            Topics that you have to find and given in the response:
            1. Topic of the video or a scene theme.
            2. Species name which is talked in the video.
            3. Specific Variety of species(e.g. IPA 15-06, IPL 203, etc) on which they are talking.
            4. A detailed summary which can contain all the information which is talked and analyse from the frames.
            5. Actions taken into the video.
            6. text from the images and the scene.
            Make sure include response languge is only english. not hinglish or hindi or any other language etc.
            """

            # Handle large inputs by batching only frames, sending full transcript each time
            MAX_FRAMES_PER_BATCH = 20

            # Process frames in batches if needed, always sending full transcript
            if len(frames) > MAX_FRAMES_PER_BATCH:

                # Split frames into batches
                frame_batches = [
                    frames[i : i + MAX_FRAMES_PER_BATCH]
                    for i in range(0, len(frames), MAX_FRAMES_PER_BATCH)
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
                        These are frames {i * MAX_FRAMES_PER_BATCH + 1} to {min((i + 1) * MAX_FRAMES_PER_BATCH, len(frames))}.
                        
                        Previous analysis results: {previous_analysis}
                        
                        Continue your analysis with these additional frames, focusing on new information not captured in previous analyses.
                        Maintain consistency with your previous analysis for the same elements (species, variety, etc.) unless new visual evidence contradicts it.
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
                            model=os.getenv("AZURE_OPENAI_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_MODEL"),
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
                            1. For factual fields (topic, species, variety), use the most detailed and accurate version
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
                        model=os.getenv("AZURE_OPENAI_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_MODEL"),
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
            prompt = [{"role":"system","content":system_prompt},
                        {"role":"user","content":[
                                *map(
                                        lambda x: {
                                            "type": "image_url", 
                                            "image_url": {
                                                "url": f'data:image/jpg;base64,{x}', 
                                                "detail": "high"
                                            }
                                        }, 
                                        frames
                                    ),
                                {"type": "text", "text": f"The audio transcription is: {transcript}"}
                            ],
                        }]

            response = await self.llm_client.beta.chat.completions.parse(
                    model=os.getenv("AZURE_OPENAI_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_MODEL"),
                    messages=prompt,
                    temperature=0,
                    response_format=ChapterCreationResponse
                )

            response_object: ChapterCreationResponse = response.choices[0].message.parsed

            # Return ChapterCreationResponse instance directly
            return response_object
        except Exception as e:
            logger.exception(f"Error Creating chapters: {e}")
            raise
    