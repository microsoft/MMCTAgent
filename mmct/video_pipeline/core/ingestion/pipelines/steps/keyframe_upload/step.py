"""Keyframe Upload pipeline step.

Uploads keyframe images from the keyframes step to blob storage with metadata.
Creates blob URLs for each keyframe that can be used by downstream steps.

Data flow:
- Input: keyframes_per_chunk from keyframes step
- Output: Updated keyframes with blob_url for each image
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.providers.base.storage_provider import BaseStorageProvider


@register_step("ingestion.keyframe_upload")
class KeyframeUploadStep(PipelineStep):
    """Pipeline step for uploading keyframe images to blob storage.
    
    Reads keyframes from keyframes step and uploads each image
    to blob storage, updating the metadata with blob URLs.
    
    Params:
        source_keyframes_step: Step ID for keyframes (default: "keyframes")
        max_concurrent_uploads: Maximum parallel uploads (default: 20)
        container_name: Blob container name (default: from storage_provider)
        skip_if_uploaded: Skip keyframes that already have blob_url (default: True)
    """
    
    step_type = "ingestion.keyframe_upload"
    description = "Upload keyframe images to blob storage"
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute keyframe upload.
        
        Args:
            context: Pipeline step context with data store and providers.
            
        Returns:
            StepResult with upload statistics.
        """
        # Get source step
        source_step: str = self.get_param(
            "source_keyframes_step", context, default="keyframes"
        )
        
        # Get keyframes data
        keyframes_per_chunk: List[Dict[str, Any]] = context.data_store.get(
            source_step, "keyframes_per_chunk"
        ) or context.data_store.get("dense_keyframes", "keyframes_per_chunk") or []
        
        if not keyframes_per_chunk:
            logger.warning("No keyframes found to upload")
            return StepResult(
                step_id=self.step_id,
                outputs={"upload_complete": False, "error": "No keyframes found"},
                metrics={"keyframes_uploaded": 0},
            )
        
        video_id: str = getattr(context, "video_id", "unknown")
        
        # Count total keyframes
        total_keyframes = sum(
            len(chunk.get("keyframes", [])) for chunk in keyframes_per_chunk
        )
        logger.info(f"Starting keyframe upload for video {video_id}: {total_keyframes} keyframes")
        
        # Get configuration
        max_concurrent: int = self.get_param("max_concurrent_uploads", context, default=20)
        skip_if_uploaded: bool = self.get_param("skip_if_uploaded", context, default=True)
        
        # Get storage provider
        storage_provider: Optional[BaseStorageProvider] = getattr(
            context.provider, "storage_provider", None
        )
        
        if not storage_provider:
            logger.error("No storage provider configured")
            return StepResult(
                step_id=self.step_id,
                outputs={"upload_complete": False, "error": "No storage provider"},
                metrics={"keyframes_uploaded": 0},
            )
        
        # Upload keyframes
        metrics = await self._upload_all_keyframes(
            keyframes_per_chunk=keyframes_per_chunk,
            video_id=video_id,
            storage_provider=storage_provider,
            max_concurrent=max_concurrent,
            skip_if_uploaded=skip_if_uploaded,
        )
        
        # Store updated keyframes back to data store
        context.data_store.set(
            self.step_id, "keyframes_per_chunk", keyframes_per_chunk
        )
        
        logger.info(
            f"Keyframe upload complete: {metrics['keyframes_uploaded']}/{metrics['total_keyframes']} uploaded"
        )
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "upload_complete": True,
                "keyframes_per_chunk": keyframes_per_chunk,
                "keyframes_uploaded": metrics["keyframes_uploaded"],
            },
            metrics=metrics,
        )
    
    async def _upload_all_keyframes(
        self,
        keyframes_per_chunk: List[Dict[str, Any]],
        video_id: str,
        storage_provider: BaseStorageProvider,
        max_concurrent: int,
        skip_if_uploaded: bool,
    ) -> Dict[str, Any]:
        """Upload all keyframes to blob storage.
        
        Args:
            keyframes_per_chunk: List of chunk data with keyframes.
            video_id: Video identifier for blob path.
            storage_provider: Blob storage provider.
            max_concurrent: Maximum concurrent uploads.
            skip_if_uploaded: Skip if already has blob_url.
            
        Returns:
            Dict with upload metrics.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        total_keyframes = 0
        uploaded_count = 0
        skipped_count = 0
        failed_count = 0
        
        async def upload_single_keyframe(
            keyframe: Dict[str, Any],
            chunk_id: str,
        ) -> bool:
            """Upload a single keyframe and update its blob_url."""
            nonlocal uploaded_count, skipped_count, failed_count
            
            async with semaphore:
                # Check if already uploaded
                if skip_if_uploaded and keyframe.get("blob_url"):
                    skipped_count += 1
                    return True
                
                filepath = keyframe.get("filepath")
                if not filepath or not os.path.exists(filepath):
                    logger.warning(f"Keyframe file not found: {filepath}")
                    failed_count += 1
                    return False
                
                try:
                    # Generate blob path: video_id/keyframes/chunk_id/filename
                    filename = os.path.basename(filepath)
                    blob_name = f"{video_id}/keyframes/{chunk_id}/{filename}"
                    
                    # Get container name from storage provider
                    container_name = getattr(
                        storage_provider, "keyframe_container_name", "keyframes"
                    )
                    
                    # Upload to blob storage
                    blob_url = await storage_provider.upload_file(
                        file_name=blob_name,
                        src_file_path=filepath,
                        folder_name=container_name,
                    )
                    
                    # Update keyframe with blob info
                    keyframe["blob_name"] = blob_name
                    keyframe["blob_url"] = blob_url
                    keyframe["uploaded_at"] = datetime.now(timezone.utc).isoformat()
                    
                    uploaded_count += 1
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to upload keyframe {filepath}: {e}")
                    failed_count += 1
                    return False
        
        # Create upload tasks for all keyframes
        tasks = []
        for chunk_data in keyframes_per_chunk:
            chunk_id = chunk_data.get("chunk_id", "unknown")
            keyframes = chunk_data.get("keyframes", [])
            total_keyframes += len(keyframes)
            
            for keyframe in keyframes:
                tasks.append(upload_single_keyframe(keyframe, chunk_id))
        
        # Execute all uploads
        if tasks:
            await asyncio.gather(*tasks)
            
            # Log progress
            logger.info(
                f"Upload complete: {uploaded_count} uploaded, "
                f"{skipped_count} skipped, {failed_count} failed"
            )
        
        return {
            "total_keyframes": total_keyframes,
            "keyframes_uploaded": uploaded_count,
            "keyframes_skipped": skipped_count,
            "keyframes_failed": failed_count,
        }
