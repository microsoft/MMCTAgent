"""Image analysis executor — programmatic keyframe analysis.

Downloads keyframes from blob storage and runs ViT analysis
in parallel. No LLM routing — tools are called directly.
"""

import os
import re
import asyncio
from typing import Annotated, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime

from loguru import logger


_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{_GRAY}[{now}]{_RESET} {_CYAN}[state:image]{_RESET} {msg}", flush=True)


def _normalize_blob_url(url: str) -> str:
    return re.sub(r"(?<!:)/{2,}", "/", url)


def _extract_blob_path(url: str, container_name: str) -> Tuple[str, str]:
    url = _normalize_blob_url(url)
    if url.startswith("http://") or url.startswith("https://"):
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if path.startswith(container_name + "/"):
            return container_name, path[len(container_name) + 1 :]
        return container_name, path
    return container_name, url


class ImageAnalysisExecutor:
    """Batch keyframe analysis — parallel ViT calls.

    Downloads frames from blob storage, runs vision tools,
    and returns structured results. No LLM routing involved.
    """

    def __init__(
        self,
        image_llm_provider=None,
        storage_provider=None,
        local_frame_dir: Annotated[str, "Directory used for temporary keyframe downloads"] = "./downloaded_frames",
    ):
        self.image_llm_provider = image_llm_provider
        self.storage_provider = storage_provider
        self.local_frame_dir = local_frame_dir
        self._downloaded_files: List[str] = []

    async def analyze_keyframes(
        self,
        keyframes: Annotated[List[Dict[str, Any]], "Keyframes with blob URLs and metadata"],
        query: Annotated[str, "User visual query to evaluate against each keyframe"],
    ) -> List[Dict[str, Any]]:
        """Analyze keyframes using ViT in parallel.

        Args:
            keyframes: List of keyframe dicts with blob_url, video_id, timestamp.
            query: User's visual query.

        Returns:
            List of analysis results with metadata.
        """
        if not keyframes:
            return []

        if self.image_llm_provider is None:
            logger.warning("Image LLM provider not configured")
            return []

        _log(f"Analyzing {len(keyframes)} keyframes for: '{query[:50]}...'")

        tasks = [
            self._analyze_single(kf, query)
            for kf in keyframes
            if kf.get("blob_url")
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        analyses = []
        for kf, result in zip(keyframes, results):
            if isinstance(result, Exception):
                logger.warning(f"Analysis failed for {kf.get('blob_url')}: {result}")
                continue
            analyses.append({
                "video_id": kf.get("video_id"),
                "timestamp": kf.get("timestamp"),
                "chapter_id": kf.get("chapter_id"),
                "analysis": result,
            })

        _log(f"{_YELLOW}Completed {len(analyses)} image analyses{_RESET}")
        return analyses

    async def _analyze_single(
        self,
        keyframe: Annotated[Dict[str, Any], "Single keyframe metadata entry"],
        query: Annotated[str, "User visual query"],
    ) -> str:
        """Analyze a single keyframe with ViT."""
        from mmct.image_pipeline.core.tools.vit import VitTool

        blob_url = keyframe.get("blob_url", "")
        local_path = await self._resolve_image(blob_url)

        tool = VitTool(llm_provider=self.image_llm_provider, img_path=local_path)
        return await tool.vit_tool(query)

    async def _resolve_image(
        self,
        image_path: Annotated[str, "Existing local path or blob URL"],
    ) -> str:
        if os.path.exists(image_path):
            return image_path
        if self.storage_provider is not None:
            return await self._download_from_blob(_normalize_blob_url(image_path))
        raise FileNotFoundError(f"Image not found: {image_path}")

    async def _download_from_blob(
        self,
        blob_url: Annotated[str, "Blob URL to download locally"],
    ) -> str:
        os.makedirs(self.local_frame_dir, exist_ok=True)
        container_name, blob_path = _extract_blob_path(
            blob_url, self.storage_provider.keyframe_container_name
        )
        safe_filename = blob_path.replace("/", "_")
        local_path = os.path.join(self.local_frame_dir, safe_filename)

        if os.path.exists(local_path):
            return local_path

        image_data = await self.storage_provider.load_file_to_memory(
            folder=container_name, file_name=blob_path
        )
        with open(local_path, "wb") as f:
            f.write(image_data)

        self._downloaded_files.append(local_path)
        return local_path

    def cleanup(self) -> int:
        """Delete downloaded frames."""
        deleted = 0
        for fp in self._downloaded_files:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
                    deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {fp}: {e}")

        self._downloaded_files.clear()

        try:
            if os.path.exists(self.local_frame_dir) and not os.listdir(self.local_frame_dir):
                os.rmdir(self.local_frame_dir)
        except Exception:
            pass

        return deleted
