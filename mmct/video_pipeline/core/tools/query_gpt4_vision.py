"""
This tool allows to do the query on selected frames around a given timestamp
"""

# Importing Libaries
import os
from typing_extensions import Annotated
import time
import asyncio
from mmct.video_pipeline.utils.helper import download_blobs, load_images, encode_image_to_base64, get_media_folder
from mmct.llm_client import LLMClient
from dotenv import load_dotenv, find_dotenv
from loguru import logger
# Load environment variables
load_dotenv(find_dotenv(), override=True)

service_provider = os.getenv("LLM_PROVIDER", "azure")
openai_client = LLMClient(service_provider=service_provider, isAsync=True)
openai_client = openai_client.get_client()

async def query_gpt4_vision(
    query: Annotated[str, "A natural language question about the visual content of the video."], 
    timestamp: Annotated[str, "The timestamp (in %H:%M:%S format) around which to sample frames for visual analysis."], 
    video_id: Annotated[str, "Unique identifier for the video from which frames will be extracted."]
) -> str:
    """
    A visual analysis tool that leverages GPT-4 Vision model to answer user queries based on still frames extracted 
    around a given timestamp in a video.

    - This tool is crucial for grounding user queries in visual evidence, making it highly suitable for timestamp-level
      video inspections, content moderation, or factual visual Q&A tasks.
      
    Returns:
    - str: The response from GPT-4 Vision, based strictly on the visible content of the frames. If relevant visual evidence 
           is not found in the frames, the model may respond with uncertainty or indicate lack of sufficient context.
    """
    try:
        logger.info("Utilizing query gpt 4v tool")
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
                                "You are `gpt-4`, the OpenAI model that can describe still frames provided by the user "
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
                logger.info("Initiating the call to query GPT 4 vision")
                response =  await openai_client.chat.completions.create(
                        model=os.getenv("AZURE_OPENAI_VISION_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_VISION_MODEL"),
                        temperature=payload["temperature"],
                        messages=payload['messages'],
                        top_p=payload['top_p'],
                        max_tokens=payload["max_tokens"]
                    )
                return response.choices[0].message.content
            except Exception as e:
                logger.error("Error when performing query gpt 4v call")
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
    query = "What type of content is in the image?"
    timestamps = "00:10:11"
    video_id = "009d738d0b4bb8374830a7894c7f3cd4134c6c89a440bfd656cea819c9bf4565"
    res = asyncio.run(query_gpt4_vision(query=query, timestamp=timestamps, video_id=video_id))
    print(res)
