"""Query specific video frames to extract detailed information and answer questions - Optimized version.

This tool analyzes video frames with optimized I/O operations for faster image loading and encoding.
Features concurrent processing, efficient compression, and memory-optimized operations.
"""
import os
import asyncio
import base64
from datetime import time
from typing import Annotated, Optional
from loguru import logger
from mmct.providers.factory import provider_factory
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher

# Initialize providers
llm_provider = provider_factory.create_llm_provider()

storage_provider = provider_factory.create_storage_provider()

async def download_and_encode_blob(file_name: str, folder_name: str, save_locally: bool = False, local_dir: str = "./debug_frames") -> Optional[str]:
    """Download JPG blob using storage_provider and encode to base64."""
    try:
        # Load blob data using storage provider
        image_data = await storage_provider.load_file_to_memory(folder=folder_name, file_name=file_name)

        # Optionally save to local disk for debugging
        if save_locally:
            os.makedirs(local_dir, exist_ok=True)
            # Create safe filename from blob_name
            safe_filename = file_name.replace('/', '_')
            local_path = os.path.join(local_dir, safe_filename)
            with open(local_path, 'wb') as f:
                f.write(image_data)
            print(f"Saved frame to: {local_path}")

        # Direct base64 encoding (no processing needed for JPG)
        return base64.b64encode(image_data).decode('utf-8')

    except Exception as e:
        print(f"Failed to download and encode file {file_name}: {e}")
        return None

async def query_frame(
    query: Annotated[str, "user query according to which video content has to be analyzeds. e.g. 'What materials are required to prepare the chilly nursery bed, and what are their uses?','count the person doing exercise in the video?'"],
    index_name: Annotated[str, "search index name"],
    frame_ids: Annotated[Optional[list], "List of specific frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])"] = None,
    video_id: Annotated[Optional[str], "Unique video identifier hash for frame retrieval"] = None,
    timestamps: Annotated[Optional[list], "List of time range pairs in HH:MM:SS format like [start_time, end_time], e.g., [['00:07:45', '00:09:44']]. Only 1 start_time, end)time pair"] = None,
    provider_name: Annotated[Optional[str], "Optional search provider name to override config (e.g. 'local_faiss')"] = None
) -> str:
    """
    This is query_frame tool which takes a user query and either specific frame IDs or
    timestamps to search for relevant frames in a video.
    """

    # Handle video_id validation and truncation for compatibility
    # if video_id and len(video_id) > 64:
    #     video_id = video_id[:64]
    save_frames_locally  = False
    # Get search endpoint from environment
    search_endpoint = os.getenv('SEARCH_ENDPOINT')

    # If there is a FAISS index directory in examples/ (e.g. from exported indices), prefer it
    provider_config = None
    alt_faiss_dir = os.path.join(os.getcwd(), "examples", "mmct_faiss_indices")
    default_faiss_dir = os.path.join(os.getcwd(), "mmct_faiss_indices")
    if os.path.isdir(alt_faiss_dir) and any(os.scandir(alt_faiss_dir)):
        provider_config = {"index_path": alt_faiss_dir}
    elif os.path.isdir(default_faiss_dir) and any(os.scandir(default_faiss_dir)):
        provider_config = {"index_path": default_faiss_dir}

    # Initialize searcher
    searcher = KeyframeSearcher(
        search_endpoint=search_endpoint,
        index_name=f"keyframes-{index_name}",
        provider_name=provider_name,
        provider_config=provider_config,
    )

    # Determine which frames to use
    frame_filenames = []

    if timestamps:
        # Search for frames based on timestamps
        for timestamp in timestamps:
            start_time, end_time = timestamp
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
            print(combined_filter)

            # Search for relevant frames with similarity filtering
            results = await searcher.search_keyframes(
                query=query,
                top_k=5,
                video_filter=combined_filter
            )

            # check for query matching and check fetched frames

            for result in results:
                keyframe_filename = result.get('keyframe_filename', '')
                if keyframe_filename:
                    frame_filenames.append(keyframe_filename)
    else:
        # Use provided frame_ids
        frame_filenames = [f"{video_id}_{frame_id}" for frame_id in frame_ids if frame_id is not None]

    # Make frame_filenames unique
    frame_filenames = sorted(
        list(dict.fromkeys(frame_filenames)),
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[-1].split('.')[0])) or 0)
    )
    print(f"Processing {len(frame_filenames)} frames directly from storage provider")

    # Prepare blob paths
    folder_name = "keyframes"
    file_paths = [f"{video_id}/{j}" for j in frame_filenames if j is not None]

    # Download and encode images directly from storage provider (no disk I/O)
    logger.info(f"Downloading and encoding {len(file_paths)} images from storage provider...")

    # Process blobs concurrently - direct blob to base64
    tasks = [download_and_encode_blob(file_name=file_name,folder_name=folder_name, save_locally=save_frames_locally) for file_name in file_paths]
    encoded_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    encoded_images = [result for result in encoded_results
                     if isinstance(result, str) and result is not None]

    print(f"Successfully processed {len(encoded_images)} images directly from storage provider")

    if not encoded_images:
        return "No valid images could be processed."

    # Prepare content for LLM query
    content = []
    for encoded_image in encoded_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                "detail": "high"
            }
        })

    content.append({
    "type": "text",
    "text": f"Query: {query}"
    })

    payload = {
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are an expert visual analysis assistant trained to extract detailed, visually grounded information from a set of video frames.

            Your task is to analyze the provided frames — each representing a distinct moment — and answer the **user’s query** based only on what is visible in these frames.

            ### Core Objectives
            1. Accurately **answer the provided query** using only visible evidence from the frames.
            2. Identify **key visual elements or events** (e.g., objects, people, actions, materials, or text).
            3. Focus strictly on **what is visible** — not on assumptions or external knowledge.
            4. **Ignore irrelevant frames** (blurry, duplicated, or contextually unrelated) and base conclusions only on meaningful visuals.
            5. If there are conflicting visuals, **weigh clarity and relevance** to the query in your analysis.

            ### Guidelines
            - Examine **each frame independently**
            - Highlight **objects, people, actions, or materials** relevant to the query.
            - When some frames are unclear, **prioritize clarity and relevance** — focus on those that help answer the question.
            - Ensure your final answer is **fully grounded in visible evidence**.
            - Do **not** speculate or use any information not directly seen in the frames.

            ### Output Format
            Provide:
            1. A short **summary of relevant frames**.
            2. A **description of key visual observations** from those frames.
            3. A **final answer to the user’s query**, based solely on what is visually evident.
            """
                    }
                ]
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0,
        "top_p": 0.1,
    }


    response = await llm_provider.chat_completion(
        messages=payload['messages'],
        temperature=payload["temperature"],
        #top_p=payload['top_p'],
        max_tokens=500
    )

    # Clean up memory after LLM call
    del encoded_images
    del content
    del payload

    return response["content"]


if __name__ == "__main__":
    import asyncio

    async def main():
        # Use the concrete inputs provided for debugging
        query = "count the fruits on the christmas tree"
        index_name = "local_search_index"
        video_id = "808ef24205b8bfe7181818699675f5a4dbfe5974baf5ded99ab5b5b3c8b6f15d"
        # Use timestamps mode (list of [start, end]) as in your function call
        timestamps = [["00:00:00", "00:00:46"]]

        # Call the tool using timestamps mode (frame_ids=None)
        result = await query_frame(
            query=query,
            index_name=index_name,
            frame_ids=None,
            video_id=video_id,
            timestamps=timestamps
        )

        print("query_frame result:", result)

    asyncio.run(main())