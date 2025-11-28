"""Query specific video frames to extract detailed information and answer questions - Optimized version.

This tool analyzes video frames with optimized I/O operations for faster image loading and encoding.
Features concurrent processing, efficient compression, and memory-optimized operations.
"""
import os
import asyncio
import base64
from typing import Annotated, List, Optional
from urllib.parse import urlparse
from loguru import logger
from mmct.providers.factory import provider_factory

# Initialize providers
llm_provider = provider_factory.create_llm_provider()

storage_provider = provider_factory.create_storage_provider()

MAX_FRAMES_PER_REQUEST = 5

def _parse_blob_url(blob_url: str) -> Optional[tuple[str, str]]:
    """Convert a blob URL into container and blob path components."""
    if not blob_url:
        return None

    parsed = urlparse(blob_url)
    blob_path = parsed.path.lstrip("/")
    if not blob_path or "/" not in blob_path:
        return None

    container, blob_name = blob_path.split("/", 1)
    return container, blob_name


async def download_and_encode_blob(blob_url: str, save_locally: bool = False, local_dir: str = "./debug_frames") -> Optional[str]:
    """Download JPG blob via its URL using storage_provider and encode to base64."""
    parsed = _parse_blob_url(blob_url)
    if not parsed:
        logger.warning(f"Unable to parse blob URL: {blob_url}")
        return None

    folder_name, file_name = parsed

    try:
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


def _chunk_frames(items, chunk_size: int):
    """Yield successive chunks from a list."""
    for idx in range(0, len(items), chunk_size):
        yield items[idx: idx + chunk_size]

async def get_frame_analysis(
    query: Annotated[str, "Detailed description of the visual information needed from the frames."],
    blob_urls: Annotated[List[str], "List of blob URLs pointing to keyframe images."],
) -> str:
    """
    Analyze the provided frame blobs with respect to the user's query.

    All frame discovery is expected to happen upstream; this tool focuses solely on
    downloading the given blobs, batching them into groups of at most five images, and
    invoking the vision LLM to reason about visible details.
    """
    if not blob_urls:
        return "blob_urls is required."

    save_frames_locally = False

    logger.info(f"Preparing to analyze {len(blob_urls)} provided frame blobs")

    tasks = [
        download_and_encode_blob(
            blob_url=blob_url,
            save_locally=save_frames_locally,
        )
        for blob_url in blob_urls
    ]
    encoded_results = await asyncio.gather(*tasks, return_exceptions=True)

    encoded_frames = []
    for idx, (blob_url, result) in enumerate(zip(blob_urls, encoded_results)):
        if isinstance(result, str) and result is not None:
            encoded_frames.append({
                "blob_url": blob_url,
                "image_b64": result,
                "index": idx,
            })

    logger.info(f"Successfully processed {len(encoded_frames)} frames from provided blob URLs")

    if not encoded_frames:
        return "No valid images could be processed."

    system_prompt = """You are an expert visual analysis assistant trained to extract detailed, visually grounded information from a set of video frames.

Your task is to analyze the provided frames — each representing a distinct moment — and answer the user's query based only on what is visible in these frames.

### Core Objectives
1. Accurately answer the provided query using only visible evidence from the frames.
2. Identify key visual elements or events (objects, people, actions, materials, or text).
3. Focus strictly on what is visible — not on assumptions or external knowledge.
4. Ignore irrelevant frames (blurry, duplicated, or contextually unrelated) and base conclusions only on meaningful visuals.
5. If there are conflicting visuals, weigh clarity and relevance to the query in your analysis.

### Output Format
Provide:
1. A short summary of the relevant frames.
2. Key visual observations tied to the query.
3. A final answer to the user's query, based solely on visible evidence.
"""

    batch_responses = []
    for batch_index, frame_batch in enumerate(_chunk_frames(encoded_frames, MAX_FRAMES_PER_REQUEST), start=1):
        content = []
        metadata_lines = []

        for frame in frame_batch:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_b64']}",
                    "detail": "high"
                }
            })
            metadata_lines.append(f"- Frame index {frame['index']}")

        content.append({
            "type": "text",
            "text": (
                f"Query: {query}\n"
                f"Frame details:\n{os.linesep.join(metadata_lines)}"
            )
        })

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
        }

        response = await llm_provider.chat_completion(
            messages=payload['messages'],
            temperature=payload["temperature"],
            max_tokens=500,
        )

        batch_responses.append(f"Batch {batch_index} insight:\n{response['content']}")

    return "\n\n".join(batch_responses)


if __name__ == "__main__":
    import asyncio

    async def main():
        # Sample inputs for local debugging
        query = "Describe all the frames"
        blob_urls = [
            "https://geckostorageaccount.blob.core.windows.net/kv-new-chapter-gen-frames/9r8ph2pb9aw.mp4/of_scene_frames_chunk0024_0001.jpg",
            "https://geckostorageaccount.blob.core.windows.net/kv-new-chapter-gen-frames/9r8ph2pb9aw.mp4/of_scene_frames_chunk0025_0000.jpg",
            "https://geckostorageaccount.blob.core.windows.net/kv-new-chapter-gen-frames/9r8ph2pb9aw.mp4/of_scene_frames_chunk0025_0008.jpg"
        ]

        result = await get_frame_analysis(
            query=query,
            blob_urls=blob_urls,
        )

        print("get_frame_analysis result:", result)

    asyncio.run(main())