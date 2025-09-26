"""Query specific video frames to extract detailed information and answer questions - Optimized version.

This tool analyzes video frames with optimized I/O operations for faster image loading and encoding.
Features concurrent processing, efficient compression, and memory-optimized operations.
"""
import os
import asyncio
import base64
from datetime import time
from typing import Annotated, Optional

from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.identity import DefaultAzureCredential, AzureCliCredential
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher

# Initialize configuration and providers
config = MMCTConfig(model_name=os.getenv("LLM_VISION_DEPLOYMENT_NAME", "gpt-4o"))
llm_provider = provider_factory.create_llm_provider(
    config.llm.provider,
    config.llm.model_dump()
)

def _get_credential():
    """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
    try:
        # Try Azure CLI credential first
        cli_credential = AzureCliCredential()
        # Test if CLI credential works by getting a token
        cli_credential.get_token("https://storage.azure.com/.default")
        return cli_credential
    except Exception:
        return DefaultAzureCredential()

async def download_and_encode_blob(blob_name: str, container_name: str) -> Optional[str]:
    """Download JPG blob directly to memory and encode to base64."""
    try:
        credential = _get_credential()
        blob_service_client = AsyncBlobServiceClient(
            os.getenv("BLOB_ACCOUNT_URL"), credential
        )
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        # Download blob data directly to memory
        stream = await blob_client.download_blob()
        image_data = await stream.readall()

        # Direct base64 encoding (no processing needed for JPG)
        return base64.b64encode(image_data).decode('utf-8')

    except Exception as e:
        print(f"Failed to download and encode blob {blob_name}: {e}")
        return None

async def query_frame(
    query: Annotated[str, "Natural language question about video content to analyze"],
    frame_ids: Annotated[Optional[list], "List of specific frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])"] = None,
    video_id: Annotated[Optional[str], "Unique video identifier hash for frame retrieval"] = None,
    timestamps: Annotated[Optional[list], "List of time range pairs in HH:MM:SS format, e.g., [['00:07:45', '00:09:44'], ['00:21:22', '00:23:17']]"] = None
) -> str:
    """
    This is query_frame tool with optimized I/O operations.
    """

    # Handle video_id validation and truncation for compatibility
    if video_id and len(video_id) > 64:
        video_id = video_id[:64]

    # Get search endpoint from environment
    search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT', 'https://osaistemp.search.windows.net')

    # Initialize searcher
    searcher = KeyframeSearcher(
        search_endpoint=search_endpoint,
        index_name=os.getenv("KEYFRAME_INDEX_NAME", "farming-video-bihar-index")
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
    frame_filenames = list(dict.fromkeys(frame_filenames))
    print(f"Processing {len(frame_filenames)} frames directly from blob storage")

    # Prepare blob paths
    container_name = "keyframes"
    blob_paths = [f"{video_id}/{j}" for j in frame_filenames if j is not None]

    # Download and encode images directly from blob storage (no disk I/O)
    print(f"Downloading and encoding {len(blob_paths)} images from blob storage...")

    # Process blobs concurrently - direct blob to base64
    tasks = [download_and_encode_blob(blob_path, container_name) for blob_path in blob_paths]
    encoded_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    encoded_images = [result for result in encoded_results
                     if isinstance(result, str) and result is not None]

    print(f"Successfully processed {len(encoded_images)} images directly from blob storage")

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
                        "text": """You are an expert visual analysis assistant specialized in extracting detailed information from video frames. Your task is to analyze the provided images and answer queries based solely on the visual content observed.

                        Instructions:
                        - Analyze all visual elements present in the images thoroughly
                        - Provide comprehensive and accurate descriptions based only on what you can see
                        - Extract relevant details that would be useful for downstream processing
                        - Do not include any information from your training knowledge that is not visible in the images
                        - Be specific and detailed in your observations
                        - Each image represents a separate individual frame from the video""",
                    }
                ]
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0,
        "top_p": 0.1
    }

    response = await llm_provider.chat_completion(
        messages=payload['messages'],
        temperature=payload["temperature"],
        top_p=payload['top_p'],
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