"""
This tool allows to do the query on selected frames around a given timestamp
"""

# Importing Libaries
import os
from typing_extensions import Annotated
from typing import Optional
import time
import asyncio
from mmct.video_pipeline.utils.helper import download_blobs, load_images, encode_image_to_base64, get_media_folder
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from dotenv import load_dotenv, find_dotenv
from loguru import logger
# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Initialize configuration and providers
config = MMCTConfig()
llm_provider = provider_factory.create_llm_provider(
    config.llm.provider,
    config.llm.model_dump()
)

async def query_vision_llm(
    query: Annotated[str, "A natural language question about the visual content of the video."], 
    timestamp: Annotated[str, "The timestamp (in %H:%M:%S format) around which to sample frames for visual analysis."], 
    video_id: Annotated[str, "Unique identifier for the video from which frames will be extracted."],
    llm_provider: Optional[object] = None,
    vision_provider: Optional[object] = None
) -> str:
    """
    A visual analysis tool that leverages Vision model to answer user queries based on still frames extracted 
    around a given timestamp in a video.

    - This tool is crucial for grounding user queries in visual evidence, making it highly suitable for timestamp-level
      video inspections, content moderation, or factual visual Q&A tasks.
      
    Returns:
    - str: The response from Vision LLM, based strictly on the visible content of the frames. If relevant visual evidence 
           is not found in the frames, the model may respond with uncertainty or indicate lack of sufficient context.
    """
    try:
        logger.info("Utilizing query vision llm tool")
        # Convert timestamp to milliseconds
        h, m, s = map(int, timestamp.split(":"))
        index_of_frame = int(h * 3600 + m * 60 + s)

        start_index = max(0, index_of_frame - 4)
        end_index = index_of_frame + 5
        frame_indices = list(range(start_index, end_index + 1))
        frame_indices = [idx for idx in frame_indices if idx>=0]

        # Prepare download paths
        base_dir = os.path.join(await get_media_folder(),"Frames",f"{video_id}")
        frames_download_path = base_dir
        os.makedirs(frames_download_path, exist_ok=True)
        if not os.path.exists(frames_download_path):
            logger.error(f"File path do not exists: {frames_download_path}")
        # Download blobs
        blob_names = [f"frames/{video_id}/frame_{i}.png" for i in frame_indices]
        downloaded_files = await download_blobs(blob_names=blob_names, output_dir=frames_download_path)
        if len(downloaded_files)==0:
            return "Provided timestamp does not present in the video,"
        # file_paths = [for file_path in file]
        logger.info("Loading PNG Images")
        images = await load_images(file_paths=[os.path.join(frames_download_path, file) for file in downloaded_files])
        logger.info("PNG Images successfully loaded")
        base64Frames = [await encode_image_to_base64(image=img) for img in images]
    
        # Preparing the payload with prompts and frames.
        content = []
        for i in base64Frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{i}",
                    "detail":"high",
                    
                }
            })

        content.append(
            {
                "type":"text",
                "text" :f"{query}"
            }
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a vision model that can describe still frames provided by the user "
                                "from a short video clip in extreme detail. The user has attached frames sampled at 1 fps for you"
                                "to analyze. For every user query, you must carefully examine the frames for the relevant information "
                                "corresponding to the user query and respond accordingly. You are not allowed to hallucinate. Understand the frames carefully, If you have doubt then do not provide answer."
                                "Provide answers only available in the frames."
                                "You are given frames around a timestamp, may be frames are not relevant to the query, you need to be careful."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "top_p":0.1
        }
        logger.info("Payload ready")

        retry_intervals = [10,15]
        for attempt, wait_time in enumerate(retry_intervals,start=1):
            try:
                logger.info("Initiating the call to query vision llm")
                response = await llm_provider.chat_completion(
                    messages=payload['messages'],
                    temperature=payload["temperature"],
                    top_p=payload['top_p'],
                )
                return response["content"]
            except Exception as e:
                logger.error("Error when performing query vision llm call")
                error_message = str(e)
                if attempt<len(retry_intervals):
                    print(f"Attempt {attempt} failed: {error_message}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else: 
                    print(f"Final attempt failed: {error_message}")
                    return error_message
    except Exception as e:
        raise Exception(e)
    
if __name__ == "__main__":
    # Example usage - replace with your actual values
    query = "example question about the video frames"
    timestamps = "00:01:30"  # HH:MM:SS format
    video_id = "example-video-id"
    res = asyncio.run(query_vision_llm(query=query, timestamp=timestamps, video_id=video_id))
    print(res)
