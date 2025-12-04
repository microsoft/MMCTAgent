"""
Query specific video frames to extract detailed information and answer questions.
This tool analyzes video frames.
"""

import os
import asyncio
import base64
from typing import Annotated, Optional
from loguru import logger
from mmct.providers.base import BaseLLMProvider, BaseStorageProvider, BaseKeyframesVectorDBProvider, BaseImageEmbeddingProvider
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher


class QueryFrameTool:
    def __init__(self, llm_provider:BaseLLMProvider, storage_provider:BaseStorageProvider, vectordb_keyframes: BaseKeyframesVectorDBProvider, image_embedding_provider:BaseImageEmbeddingProvider):
        """Initialize QueryFrameTool with provider configuration.
        
        Args:
            providers: VideoAgentProviderConfig containing all required providers
        """
        self.llm_provider = llm_provider
        self.storage_provider = storage_provider
        self.vectordb_keyframes = vectordb_keyframes
        self.image_embedding_provider = image_embedding_provider  # For CLIP embeddings

    async def query_frame(
        self,
        query: Annotated[
            str,
            "user query according to which video content has to be analyzed. If options are available and relevant with the query, they should also be passed. e.g. 'What materials are required to prepare the chilly nursery bed, and what are their uses?','count the person doing exercise in the video?'",
        ],
        frame_ids: Annotated[
            Optional[list],
            "List of frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])",
        ] = None,
        video_id: Annotated[
            Optional[str],
            "Unique hash video id as an identifier for frame retrieval. Mandatory if frame_ids are provided. Do extract it from the URL",
        ] = None,
        start_time: Annotated[Optional[float], "start time in seconds"] = None,
        end_time: Annotated[Optional[float], "end time in seconds"] = None,
    ) -> str:
        """
        Analyze specific video frames using vision models for visual verification.

        Description:
            Uses vision models to analyze video frames and extract visual information.
            Can work with either specific frame IDs or timestamp ranges.

        Input Parameters:
            - query (str): Detailed description of what to look for in frames
                        (e.g., "Count people doing exercises", "What color shirt is the person wearing?")
            - frame_ids (Optional[list]): List of specific frame filenames to analyze (from get_relevant_frames)
            - video_id (Optional[str]): Video identifier (required if using start_time/end_time)
            - start_time (Optional[float]): Start time in seconds (use from get_context or object's first_seen in get_object_collection output.)
            - end_time (Optional[float]): End time in seconds (start_time + 5 seconds, if start_time is from the get_objection_collection tool else use what is)

        Output:
            String containing visual analysis results including:
            - Detailed observations about visible content
            - Object positions, counts, and spatial relationships
            - Actions, poses, gestures, expressions
            - Colors, appearances, visual attributes
            - Text visible in frames
            - Any other visual details relevant to query
        """
        if len(video_id) == 64:
            parent_id = video_id
        else:
            parent_id = video_id[:64]

        save_frames_locally = False # variable to debug the frames
        # If there is a FAISS index directory in examples/ (e.g. from exported indices), prefer it
        provider_config = None
        alt_faiss_dir = os.path.join(os.getcwd(), "examples", "mmct_faiss_indices")
        default_faiss_dir = os.path.join(os.getcwd(), "mmct_faiss_indices")
        if os.path.isdir(alt_faiss_dir) and any(os.scandir(alt_faiss_dir)):
            provider_config = {"index_path": alt_faiss_dir}
        elif os.path.isdir(default_faiss_dir) and any(os.scandir(default_faiss_dir)):
            provider_config = {"index_path": default_faiss_dir}

        # Initialize searcher with injected providers
        searcher = KeyframeSearcher(
            search_provider=self.vectordb_keyframes,
            image_embedding_provider=self.image_embedding_provider,
            provider_config=provider_config,
        )

        # Determine which frames to use
        frame_filenames = []

        if not (None in (start_time, end_time)):
            combined_filter = dict()
            combined_filter["timestamp_seconds"] = {"ge": start_time, "le": end_time}
            combined_filter["parent_id"] = {"eq": parent_id}

            # Search for relevant frames with similarity filtering
            results = await searcher.search_keyframes(
                query=query, top_k=5, video_filter=combined_filter
            )

            # check for query matching and check fetched frames

            for result in results:
                keyframe_filename = result.get("keyframe_filename", "")
                if keyframe_filename:
                    frame_filenames.append(keyframe_filename)
        else:
            # Use provided frame_ids
            frame_filenames = [
                f"{video_id}_{frame_id}" for frame_id in frame_ids if frame_id is not None
            ]

        # Make frame_filenames unique
        frame_filenames = sorted(
            list(dict.fromkeys(frame_filenames)),
            key=lambda x: int(
                "".join(filter(str.isdigit, os.path.basename(x).split("_")[-1].split(".")[0])) or 0
            ),
        )
        logger.info(f"Processing {len(frame_filenames)} frames directly from storage provider")

        # Prepare blob paths
        folder_name = "keyframes"
        file_paths = [f"{j.split('_')[0]}/{j}" for j in frame_filenames if j is not None]

        # Download and encode images directly from storage provider (no disk I/O)
        logger.info(f"Downloading and encoding {len(file_paths)} images from storage provider...")

        # Process blobs concurrently - direct blob to base64
        tasks = [
            self.download_and_encode_blob(
                file_name=file_name, folder_name=folder_name, save_locally=save_frames_locally
            )
            for file_name in file_paths
        ]
        encoded_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        encoded_images = [
            result for result in encoded_results if isinstance(result, str) and result is not None
        ]

        logger.info(
            f"Successfully processed {len(encoded_images)} images directly from storage provider"
        )

        if not encoded_images:
            return "No valid images could be processed."

        # Prepare content for LLM query
        content = []
        for encoded_image in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "high",
                    },
                }
            )

        content.append({"type": "text", "text": f"Query: {query}"})

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
                """,
                        }
                    ],
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "top_p": 0.1,
        }

        response = await self.llm_provider.chat_completion(
            messages=payload["messages"],
            temperature=payload["temperature"],
            # top_p=payload['top_p'],
            max_tokens=500,
        )

        # Clean up memory after LLM call
        del encoded_images
        del content
        del payload

        return response["content"]

    async def download_and_encode_blob(
        self,
        file_name: str,
        folder_name: str,
        save_locally: bool = False,
        local_dir: str = "./debug_frames",
    ) -> Optional[str]:
        """Download JPG blob using storage_provider and encode to base64."""
        try:
            # Load blob data using storage provider
            image_data = await self.storage_provider.load_file_to_memory(
                folder=folder_name, file_name=file_name
            )

            # Optionally save to local disk for debugging
            if save_locally:
                os.makedirs(local_dir, exist_ok=True)
                # Create safe filename from blob_name
                safe_filename = file_name.replace("/", "_")
                local_path = os.path.join(local_dir, safe_filename)
                with open(local_path, "wb") as f:
                    f.write(image_data)
                print(f"Saved frame to: {local_path}")

            # Direct base64 encoding (no processing needed for JPG)
            return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            print(f"Failed to download and encode file {file_name}: {e}")
            return None


if __name__ == "__main__":
    import asyncio

    async def main():

        # query = "<sample-query>"
        # index_name ="<index-name>"
        # video_id = "<hash-video-id>"
        # start_time = "<start time in seconds>"
        # end_time = "<end time in seconds>"

        query = "Which animal appears at 00:07? Options: (A) Manatee, (B) Sea turtle, (C) Lobster, (D) Clownfish."
        index_name = "test_offset_index"
        video_id = "fc31fd6e96bd6ea524c0753244303e3fa32738756ad88236612bf0df6d7f986cB"
        start_time = 0
        end_time = 12

        # Call the tool using timestamps mode (frame_ids=None)
        result = await query_frame(
            query=query,
            index_name=index_name,
            frame_ids=None,
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
        )

        print("query_frame result:", result)

    asyncio.run(main())
