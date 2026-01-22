import os
import asyncio
from typing import Annotated, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext
from mmct.image_pipeline.core.tools.vit import VitTool
from mmct.image_pipeline.core.tools.recog import RecogTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core.tools.ocr import OcrTool
from mmct.config.providers import ImageAgentProviderConfig
from mmct.providers.base import BaseStorageProvider
from loguru import logger

IMAGE_WORKER_SYSTEM_PROMPT = """
You are an Image Analysis Agent. Your goal is to analyze specific images or video frames provided by the Planner.
You have access to tools for visual analysis, object detection, OCR, and recognition.

When answering:
1. Analyze the image at the provided path (can be a local path or blob path like "video_id/frame_name.jpg").
2. Answer the specific question asked about the image.
3. Be precise and factual.
"""

class ImageAgent:
    def __init__(
        self, 
        provider: ImageAgentProviderConfig, 
        model_client, 
        storage_provider: Optional[BaseStorageProvider] = None, 
        local_frame_dir: str = "./downloaded_frames",
        model_context: Optional[ChatCompletionContext] = None
    ):
        self.provider = provider
        self.model_client = model_client
        self.storage_provider = storage_provider
        self.local_frame_dir = local_frame_dir
        self.model_context = model_context
        self._downloaded_files: list[str] = []  # Track downloaded files for cleanup
        self.tools = self._create_tool_wrappers()
        self.agent = self._create_agent()

    async def _download_frame_from_blob(self, blob_path: str) -> str:
        """
        Download a frame from Azure Blob Storage and save it locally.
        
        Args:
            blob_path: Path in blob storage (e.g., "video_id/frame_name.jpg" or just "frame_name.jpg")
            
        Returns:
            Local file path where the frame was saved.
        """
        if self.storage_provider is None:
            raise ValueError("Storage provider is required to download frames from blob storage")
        
        try:
            # Create local directory if it doesn't exist
            os.makedirs(self.local_frame_dir, exist_ok=True)
            
            # Create safe filename from blob_path
            safe_filename = blob_path.replace("/", "_")
            local_path = os.path.join(self.local_frame_dir, safe_filename)
            
            # Check if already downloaded
            if os.path.exists(local_path):
                logger.info(f"Frame already exists locally: {local_path}")
                return local_path
            
            # Download from blob storage
            logger.info(f"Downloading frame from blob storage: {blob_path}")
            image_data = await self.storage_provider.load_file_to_memory(
                folder=self.storage_provider.keyframe_container_name,
                file_name=blob_path
            )
            
            # Save locally
            with open(local_path, "wb") as f:
                f.write(image_data)
            
            logger.info(f"Saved frame to: {local_path}")
            self._downloaded_files.append(local_path)  # Track for cleanup
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download frame {blob_path}: {e}")
            raise

    def cleanup(self) -> int:
        """
        Delete all downloaded frames from this session.
        Should be called after query processing is complete.
        
        Returns:
            Number of files deleted.
        """
        deleted_count = 0
        for file_path in self._downloaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up downloaded frame: {file_path}")
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        self._downloaded_files.clear()
        
        # Optionally remove the directory if empty
        try:
            if os.path.exists(self.local_frame_dir) and not os.listdir(self.local_frame_dir):
                os.rmdir(self.local_frame_dir)
                logger.info(f"Removed empty frame directory: {self.local_frame_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove directory {self.local_frame_dir}: {e}")
        
        return deleted_count

    async def _resolve_image_path(self, image_path: str) -> str:
        """
        Resolve image path - download from blob if needed, or return local path.
        
        Args:
            image_path: Either a local file path or a blob storage path
            
        Returns:
            Local file path to the image
        """
        # If file exists locally, use it directly
        if os.path.exists(image_path):
            return image_path
        
        # If storage provider is available, try to download from blob
        if self.storage_provider is not None:
            return await self._download_frame_from_blob(image_path)
        
        # No storage provider and file doesn't exist locally
        raise FileNotFoundError(f"Image not found locally and no storage provider configured: {image_path}")

    def _create_tool_wrappers(self):
        
        async def analyze_image_with_vit(image_path: Annotated[str, "Path to the image/frame (local path or blob path like 'video_id/frame.jpg')"], query: Annotated[str, "Question about the image"]) -> str:
            """
            Analyzes an image using Vision Transformer (ViT) to answer a specific query.
            Downloads the frame from blob storage if needed.
            """
            local_path = await self._resolve_image_path(image_path)
            tool_instance = VitTool(llm_provider=self.provider.llm_provider, img_path=local_path)
            return await tool_instance.vit_tool(query)

        async def detect_objects(image_path: Annotated[str, "Path to the image/frame (local path or blob path like 'video_id/frame.jpg')"]) -> str:
            """
            Detects objects in the image.
            Downloads the frame from blob storage if needed.
            """
            local_path = await self._resolve_image_path(image_path)
            tool_instance = ObjectDetectTool(img_path=local_path)
            return await tool_instance.object_detect_tool()

        async def perform_ocr(image_path: Annotated[str, "Path to the image/frame (local path or blob path like 'video_id/frame.jpg')"]) -> str:
            """
            Extracts text from the image using OCR.
            Downloads the frame from blob storage if needed.
            """
            local_path = await self._resolve_image_path(image_path)
            tool_instance = OcrTool(img_path=local_path)
            return await tool_instance.ocr_tool()

        async def recognize_entities(image_path: Annotated[str, "Path to the image/frame (local path or blob path like 'video_id/frame.jpg')"]) -> str:
            """
            Recognizes specific entities or details in the image using RecogTool.
            Downloads the frame from blob storage if needed.
            """
            local_path = await self._resolve_image_path(image_path)
            tool_instance = RecogTool(img_path=local_path)
            return await tool_instance.recog_tool()

        return [analyze_image_with_vit, detect_objects, perform_ocr, recognize_entities]

    def _create_agent(self):
        return AssistantAgent(
            name="ImageAgent",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Agent that can analyze images using Vision tools.",
            system_message=IMAGE_WORKER_SYSTEM_PROMPT,
            tools=self.tools,
            reflect_on_tool_use=True,
            handoffs=["planner"],
        )
