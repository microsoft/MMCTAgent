"""V4 Image Agent - Visual frame analysis.

Adapted from v2 ImageAgent for the V4 query pipeline.
Handles visual analysis of keyframe images:
1. Downloads frames from blob storage
2. Analyzes using ViT, OCR, object detection
3. Returns findings to Planner

Uses AutoGen's handoff mechanism for agent communication.
"""

import os
import re
import asyncio
from typing import Annotated, Optional, Tuple
from urllib.parse import urlparse
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext
from mmct.image_pipeline.core.tools.vit import VitTool
from mmct.image_pipeline.core.tools.recog import RecogTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core.tools.ocr import OcrTool
from mmct.config.providers import ImageAgentProviderConfig
from mmct.providers.base import BaseStorageProvider
from loguru import logger


def _normalize_blob_url(url: str) -> str:
    """Normalize blob URL by removing double slashes (except after protocol).
    
    Handles URLs like:
    - https://account.blob.core.windows.net//container/path -> https://account.blob.core.windows.net/container/path
    """
    # Replace multiple slashes with single slash, but preserve ://
    return re.sub(r'(?<!:)/{2,}', '/', url)


def _extract_blob_path_from_url(url: str, container_name: str) -> Tuple[str, str]:
    """Extract the blob path from a full Azure Blob Storage URL.
    
    Args:
        url: Full blob URL or relative path
        container_name: The container name to look for in the URL
        
    Returns:
        Tuple of (container_name, blob_path) where blob_path is the path within the container
        
    Examples:
        >>> _extract_blob_path_from_url(
        ...     "https://account.blob.core.windows.net/keyframes/video_id/keyframes/8/8_kf_0005.jpg",
        ...     "keyframes"
        ... )
        ('keyframes', 'video_id/keyframes/8/8_kf_0005.jpg')
    """
    # Normalize URL first (fix double slashes)
    url = _normalize_blob_url(url)
    
    # Check if it's a full URL or just a path
    if url.startswith("http://") or url.startswith("https://"):
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")  # Remove leading slash
        
        # Path format: container_name/blob_path
        # e.g., "keyframes/video_id/keyframes/8/8_kf_0005.jpg"
        if path.startswith(container_name + "/"):
            blob_path = path[len(container_name) + 1:]  # Remove "container_name/"
            return container_name, blob_path
        else:
            # Container name not in expected position, return path as-is
            logger.warning(f"Container '{container_name}' not found in URL path: {path}")
            return container_name, path
    else:
        # It's already a relative path
        return container_name, url


V4_IMAGE_AGENT_SYSTEM_PROMPT = """
You are the **ImageAgent** in a multi-agent Video QA system. VideoAgent or Planner delegates visual analysis to you.

# CAPABILITIES
- `analyze_image_with_vit`: Answer visual questions about an image (PRIMARY TOOL)
- `detect_objects`: Detect and list objects in the image
- `perform_ocr`: Extract text from the image
- `recognize_entities`: Recognize specific entities/details

# WORKFLOW

## Step 1: Receive Task
You receive keyframe blob URLs with metadata (video_id, timestamp, chapter_id) and a question.

## Step 2: Call Tools (Batch ALL in ONE response)
- For visual questions → `analyze_image_with_vit`
- For text extraction → `perform_ocr`  
- For object listing → `detect_objects`
- **Call tools for ALL images in a SINGLE response**

## Step 3: Summarize & Handoff (CRITICAL)
After receiving ALL tool results, provide ONE concise summary:
```
**Visual Analysis Results:**

**Frame 1** (video_id: X, timestamp: Y)
- [Finding 1]
- [Finding 2]

**Frame 2** (video_id: X, timestamp: Z)
- [Finding 1]

**Summary:** [Direct answer to the visual question]
```
Then IMMEDIATELY handoff to planner.

# RULES - CRITICAL

1. **ONE analysis per image** - NEVER re-analyze the same image with the same tool
2. **Batch tool calls** - Call tools for ALL images in ONE response
3. **Include metadata** - Always report video_id, timestamp, chapter_id with findings
4. **Handoff immediately** - After summarizing, handoff to planner. Do NOT call more tools.
5. **NO LOOPS** - If you have already analyzed an image, do NOT analyze it again
6. **Be concise** - Summarize findings briefly and answer the question directly
7. **Single response** - After tool results, give ONE summary then handoff. No iterating.
"""


class V4ImageAgent:
    """V4 Image Agent for visual frame analysis.
    
    Analyzes keyframe images using vision tools:
    - ViT for visual question answering
    - OCR for text extraction
    - Object detection for entity listing
    - Recognition for specific details
    
    Handles blob storage download and local file cleanup.
    
    Attributes:
        provider: ImageAgentProviderConfig with LLM settings.
        model_client: AutoGen model client for LLM calls.
        storage_provider: Optional blob storage provider.
        local_frame_dir: Directory for downloaded frames.
        agent: The underlying AutoGen AssistantAgent.
    """
    
    def __init__(
        self, 
        provider: ImageAgentProviderConfig, 
        model_client, 
        storage_provider: Optional[BaseStorageProvider] = None, 
        local_frame_dir: str = "./downloaded_frames",
        model_context: Optional[ChatCompletionContext] = None
    ):
        """Initialize the Image agent.
        
        Args:
            provider: ImageAgentProviderConfig with LLM provider.
            model_client: AutoGen model client.
            storage_provider: Optional blob storage for downloading frames.
            local_frame_dir: Directory to store downloaded frames.
            model_context: Optional shared model context for KV cache.
        """
        self.provider = provider
        self.model_client = model_client
        self.storage_provider = storage_provider
        self.local_frame_dir = local_frame_dir
        self.model_context = model_context
        self._downloaded_files: list[str] = []
        self.tools = self._create_tool_wrappers()
        self.agent = self._create_agent()

    async def _download_frame_from_blob(self, blob_url_or_path: str) -> str:
        """Download a frame from blob storage and save locally.
        
        Args:
            blob_url_or_path: Full blob URL or relative path in blob storage.
                Handles both:
                - Full URL: "https://account.blob.core.windows.net/keyframes/video_id/keyframes/8/8_kf_0005.jpg"
                - Relative path: "video_id/keyframes/8/8_kf_0005.jpg"
            
        Returns:
            Local file path where the frame was saved.
        """
        if self.storage_provider is None:
            raise ValueError("Storage provider required to download frames")
        
        try:
            os.makedirs(self.local_frame_dir, exist_ok=True)
            
            # Extract container and blob path from URL
            container_name, blob_path = _extract_blob_path_from_url(
                blob_url_or_path, 
                self.storage_provider.keyframe_container_name
            )
            
            # Create safe local filename from the blob path (not full URL)
            safe_filename = blob_path.replace("/", "_")
            local_path = os.path.join(self.local_frame_dir, safe_filename)
            
            if os.path.exists(local_path):
                logger.info(f"Frame already exists locally: {local_path}")
                return local_path
            
            logger.info(f"Downloading frame from blob: container={container_name}, path={blob_path}")
            image_data = await self.storage_provider.load_file_to_memory(
                folder=container_name,
                file_name=blob_path
            )
            
            with open(local_path, "wb") as f:
                f.write(image_data)
            
            logger.info(f"Saved frame to: {local_path}")
            self._downloaded_files.append(local_path)
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download frame {blob_url_or_path}: {e}")
            raise

    def cleanup(self) -> int:
        """Delete all downloaded frames from this session.
        
        Returns:
            Number of files deleted.
        """
        deleted_count = 0
        for file_path in self._downloaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up frame: {file_path}")
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        self._downloaded_files.clear()
        
        try:
            if os.path.exists(self.local_frame_dir) and not os.listdir(self.local_frame_dir):
                os.rmdir(self.local_frame_dir)
                logger.info(f"Removed empty directory: {self.local_frame_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove directory: {e}")
        
        return deleted_count

    async def _resolve_image_path(self, image_path: str) -> str:
        """Resolve image path - download from blob if needed.
        
        Args:
            image_path: Local file path or blob storage path/URL
            
        Returns:
            Local file path to the image
        """
        if os.path.exists(image_path):
            return image_path
        
        if self.storage_provider is not None:
            # Normalize URL to fix double slashes
            normalized_path = _normalize_blob_url(image_path)
            return await self._download_frame_from_blob(normalized_path)
        
        raise FileNotFoundError(f"Image not found and no storage provider: {image_path}")

    def _create_tool_wrappers(self):
        """Create tool wrapper functions for AutoGen."""
        
        async def analyze_image_with_vit(
            image_path: Annotated[str, "Path to the image/frame (local or blob path)"], 
            query: Annotated[str, "Question about the image"], 
            metadata: Optional[Annotated[dict, "Frame metadata (video_id, timestamp, chapter_id)"]] = None
        ) -> str:
            """Analyze an image using Vision Transformer to answer a visual question."""
            local_path = await self._resolve_image_path(image_path)
            tool_instance = VitTool(llm_provider=self.provider.llm_provider, img_path=local_path)
            result = await tool_instance.vit_tool(query)
            return f"{metadata}\n\n{result}" if metadata else result

        async def detect_objects(
            image_path: Annotated[str, "Path to the image/frame (local or blob path)"], 
            metadata: Optional[Annotated[dict, "Frame metadata (video_id, timestamp, chapter_id)"]] = None
        ) -> str:
            """Detect and list objects in the image."""
            local_path = await self._resolve_image_path(image_path)
            tool_instance = ObjectDetectTool(img_path=local_path)
            result = await tool_instance.object_detect_tool()
            return f"{metadata}\n\n{result}" if metadata else result

        async def perform_ocr(
            image_path: Annotated[str, "Path to the image/frame (local or blob path)"], 
            metadata: Optional[Annotated[dict, "Frame metadata (video_id, timestamp, chapter_id)"]] = None
        ) -> str:
            """Extract text from the image using OCR."""
            local_path = await self._resolve_image_path(image_path)
            tool_instance = OcrTool(img_path=local_path)
            result = await tool_instance.ocr_tool()
            return f"{metadata}\n\n{result}" if metadata else result

        async def recognize_entities(
            image_path: Annotated[str, "Path to the image/frame (local or blob path)"], 
            metadata: Optional[Annotated[dict, "Frame metadata (video_id, timestamp, chapter_id)"]] = None
        ) -> str:
            """Recognize specific entities or details in the image."""
            local_path = await self._resolve_image_path(image_path)
            tool_instance = RecogTool(img_path=local_path)
            result = await tool_instance.recog_tool()
            return f"{metadata}\n\n{result}" if metadata else result

        return [analyze_image_with_vit, detect_objects, perform_ocr, recognize_entities]

    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen AssistantAgent."""
        return AssistantAgent(
            name="ImageAgent",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Agent that analyzes keyframe images using Vision tools.",
            system_message=V4_IMAGE_AGENT_SYSTEM_PROMPT,
            tools=self.tools,
            reflect_on_tool_use=False,  # Wait for ALL batch results
            handoffs=["planner"],
        )
