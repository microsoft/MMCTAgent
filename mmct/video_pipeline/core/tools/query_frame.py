"""Query specific video frames to extract detailed information and answer questions.

This tool analyzes video frames to provide comprehensive answers to queries. It can work with
frame IDs or timestamp ranges and uses advanced image processing for optimal analysis.
"""
from typing import Annotated, Optional
from datetime import time
from mmct.video_pipeline.utils.helper import download_blobs, get_media_folder, encode_image_to_base64, load_images, stack_images_horizontally
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher
import os
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig

# Initialize configuration and providers
config = MMCTConfig(model_name = os.getenv("LLM_VISION_DEPLOYMENT_NAME", "gpt-4o"))
llm_provider = provider_factory.create_llm_provider(
    config.llm.provider,
    config.llm.model_dump()
)


async def query_frame(
    query: Annotated[str, "Natural language question about video content to analyze"],
    frame_ids: Annotated[Optional[list], "List of specific frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])"] = None,
    video_id: Annotated[Optional[str], "Unique video identifier hash for frame retrieval"] = None,
    timestamps: Annotated[Optional[list], "List of time range pairs in HH:MM:SS format, e.g., [['00:07:45', '00:09:44'], ['00:21:22', '00:23:17']]"] = None
) -> str:
    """Analyze video frames to answer specific questions about visual content.

    This function processes video frames using advanced AI vision models to provide detailed
    answers to queries. It supports both direct frame selection via frame_ids and automatic
    frame discovery via timestamp ranges.

    Args:
        query: Natural language question about the video content (e.g., "What objects are visible?",
               "Describe the scene in detail", "What actions are being performed?")
        frame_ids: Optional list of specific frame filenames to analyze. If provided, only these
                  frames will be processed. Use when you have exact frame references.
        video_id: Required video identifier for frame retrieval from storage. Must be provided
                 if using timestamps parameter.
        timestamps: Optional list of time range pairs in HH:MM:SS format. The system will
                   automatically find and analyze relevant frames within these time windows.
                   Example: [['00:07:45', '00:09:44'], ['00:21:22', '00:23:17']]

    Returns:
        str: Comprehensive analysis and answer to the query based on the processed frames.
             The response includes detailed descriptions, object identification, scene analysis,
             and any other relevant information found in the frames.

    Note:
        - Either frame_ids OR (video_id + timestamps) must be provided
        - When using timestamps, the system searches for the most relevant frames within the specified ranges
        - Frames are processed with high-resolution analysis for maximum detail extraction
        - Large frame sets are optimized through intelligent stacking to maintain processing efficiency
    """

    # Handle video_id validation and truncation for compatibility
    if video_id and len(video_id) > 64:
        video_id = video_id[:64]

    # Get search endpoint from environment
    search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT', 'https://osaistemp.search.windows.net')

    # Initialize searcher
    searcher = KeyframeSearcher(
        search_endpoint=search_endpoint,
        index_name=os.getenv("KEYFRAME_INDEX_NAME", "video-keyframes-index-2")
    )
        
    # Determine which frames to use
    frame_filenames = []

    if timestamps:
        # Search for frames based on timestamps
        for timestamp in timestamps:
            start_time, end_time = timestamp
            # create filter for the timestamp range and video id; timestamp is format HH:MM:SS
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

            # Search for relevant frames
            results = await searcher.search_keyframes(
                query=query,
                top_k=10,
                video_filter=combined_filter
            )

            for result in results:
                keyframe_filename = result.get('keyframe_filename', '')
                if keyframe_filename:
                    frame_filenames.append(keyframe_filename)
    else:
        # Use provided frame_ids
        frame_filenames = frame_ids if frame_ids else []

    # Make frame_filenames unique
    frame_filenames = list(dict.fromkeys(frame_filenames))  # Preserves order while removing duplicates
    print(frame_filenames)
    # Create relevant folder
    relevant_folder = os.path.join(await get_media_folder(),'relevant_frames',video_id)

    # Download the frames from blob in the relevant folder
    container_name = "keyframes"
    blob_paths = [f"{video_id}/{j}" for j in frame_filenames if j is not None]
    await download_blobs(blob_paths, relevant_folder, container_name=container_name)

    # Load and process the downloaded frames for LLM query
    frame_paths = []
    for frame_filename in frame_filenames:
        frame_filename_only = os.path.basename(frame_filename)
        frame_path = os.path.join(relevant_folder, frame_filename_only)
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)

    # Load images from paths
    loaded_images = await load_images(frame_paths)

    # Process frames: keep first 10 as individual frames, stack remaining in groups of 3
    processed_images = []
    stacking_info = ""

    if len(loaded_images) <= 10:
        # Use all frames as individual images
        processed_images = loaded_images
    else:
        # Keep first 10 frames as individual images
        processed_images = loaded_images[:10]
        remaining_frames = loaded_images[10:]

        # Stack remaining frames in groups of 3 horizontally
        stacked_count = 0
        for i in range(0, len(remaining_frames), 3):
            group = remaining_frames[i:i+3]
            if group:
                # Stack horizontally
                stacked_image = await stack_images_horizontally(group, type="pil")
                processed_images.append(stacked_image)
                stacked_count += len(group)

        if stacked_count > 0:
            stacking_info = f"\n\nNote: {stacked_count} additional frames have been horizontally stacked in groups of 3 to optimize processing."

    # Encode the processed images to base64
    encoded_images = []
    for image in processed_images:
        encoded_image = await encode_image_to_base64(image)
        encoded_images.append(encoded_image)

    # Prepare content for LLM query
    content = []
    for i in encoded_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{i}",
                "detail":"high"
            }
        })

    content.append(
        {
            "type":"text",
            "text" :f"{query}{stacking_info}"
        }
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """You are an AI assistant that find detailed information from a set of images to answer the question. Note that some images may be horizontally stacked combinations of multiple frames to optimize processing. When analyzing such stacked images, consider each section as a separate frame.""",
                        
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

    response = await llm_provider.chat_completion(
                     messages=payload['messages'],
                    temperature=payload["temperature"],
                    top_p=payload['top_p'],
                )
    return response["content"]
    
    


if __name__ == "__main__":
    import asyncio
    
    async def main():
        query = "What materials are required to prepare the chilly nursery bed, and what are their uses?"
        video_id = "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45"
        frame_ids = [
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_1827.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_2436.jpg",
            "d5bbc45fb  8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_3770.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_4901.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_5133.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_8845.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_9802.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_10121.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_10643.jpg",
            "d5bbc45fb8d284082f84b788b9dce1a931052e1e650daa4889de78654dbb9d45_11774.jpg"
        ]
        
        result = await query_frame(query, frame_ids, video_id)
        print(f"Query result: {result}")
    
    asyncio.run(main())