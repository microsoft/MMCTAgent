"""Export step that uploads extracted frames into Azure Blob Storage and indexes them."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import uuid

from loguru import logger

from providers.azure.storage_provider import AzureStorageProvider
from providers.factory import provider_factory
from providers.search_document_models import KeyframeDocument
from settings import MMCTConfig

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step

_config = MMCTConfig()


@dataclass
class _FrameEmbedding:
    """Lightweight carrier for frame metadata paired with its embedding."""

    frame: Dict[str, Any]
    embedding: List[float]


@register_step("export.frame-blob-upload")
class FrameBlobUploadStep(PipelineStep):
    """Uploads frame artifacts to blob storage and indexes them in search."""

    description = "Uploads extracted frames to blob storage and pushes them to Azure AI Search."

    def run(self, context: StepContext) -> StepResult:
        frames_step = self.params.get("frames_step")
        if not frames_step:
            raise ValueError("'frames_step' parameter is required")

        payload = context.data_store.get(frames_step, {}) or {}
        frames: List[Dict[str, Any]] = payload.get("frames", [])
        if not frames:
            raise ValueError(
                f"Step '{frames_step}' did not produce any frames to upload."
            )

        container_name = self._resolve_container_name(context)
        video_folder = self._resolve_video_folder(context.video_uri)
        # Upload configuration
        concurrency = max(1, int(self.params.get("upload_concurrency", 8)))

        # Index configuration
        video_id = self._resolve_video_id(context)
        parent_id = (
            self.params.get("parent_id")
            or context.metadata.get("parent_id")
            or video_id
        )
        parent_duration = float(
            self.params.get("parent_duration_seconds")
            or context.metadata.get("parent_duration_seconds")
            or (context.video_duration_seconds or 0.0)
        )
        video_duration = float(
            self.params.get("video_duration_seconds")
            or context.metadata.get("video_duration_seconds")
            or (context.video_duration_seconds or 0.0)
        )
        index_name = self._resolve_index_name(video_id, context.metadata)
        blob_base_url_param = self.params.get("blob_base_url") or context.metadata.get("frame_blob_base_url", "")
        embedding_concurrency = int(self.params.get("embedding_concurrency", 4))
        upload_batch_size = int(self.params.get("upload_batch_size", 64))
        embedding_dimensions = int(self.params.get("embedding_dimensions", 512))

        provider = self._create_storage_provider()

        logger.info(
            "[{}] Uploading {} frames to container '{}' under prefix '{}'",
            self.step_id,
            len(frames),
            container_name,
            video_folder,
        )

        try:
            upload_stats = asyncio.run(
                self._upload_frames(
                    storage_provider=provider,
                    frames=frames,
                    container_name=container_name,
                    video_folder=video_folder,
                    concurrency=concurrency,
                )
            )
        finally:
            self._cleanup_provider(provider)

        base_url = upload_stats.get("base_url") or blob_base_url_param
        if base_url:
            context.metadata["frame_blob_base_url"] = base_url

        image_provider = provider_factory.create_image_embedding_provider()
        search_provider = provider_factory.create_search_provider()
        try:
            index_stats = asyncio.run(
                self._index_frames(
                    frames=frames,
                    video_id=video_id,
                    parent_id=parent_id,
                    parent_duration=parent_duration,
                    video_duration=video_duration,
                    index_name=index_name,
                    blob_base_url=base_url,
                    embedding_concurrency=max(1, embedding_concurrency),
                    upload_batch_size=max(1, upload_batch_size),
                    embedding_dimensions=max(1, embedding_dimensions),
                    image_provider=image_provider,
                    search_provider=search_provider,
                )
            )
        finally:
            self._cleanup_index_providers(image_provider, search_provider)

        produced = {
            "frames": frames,
            "container_name": container_name,
            "video_blob_prefix": video_folder,
            "frame_blob_base_url": base_url,
            "index_name": index_name,
            "frames_uploaded": upload_stats["uploaded"],
            "frames_failed": upload_stats["failed"],
            "frames_missing": upload_stats["missing"],
            "frames_indexed": index_stats["documents_indexed"],
            "frames_skipped": index_stats["frames_skipped"],
        }
        metrics = {
            "frames_uploaded": float(upload_stats["uploaded"]),
            "frames_failed": float(upload_stats["failed"]),
            "frames_missing": float(upload_stats["missing"]),
            "frames_indexed": float(index_stats["documents_indexed"]),
            "frames_processed_for_index": float(index_stats["frames_processed"]),
        }

        return StepResult(
            step_id=self.step_id,
            produced=produced,
            metrics=metrics,
        )

    def _resolve_container_name(self, context: StepContext) -> str:
        explicit = self.params.get("container_name") or context.metadata.get("frame_blob_container")
        fallback = _config.storage.container_name
        container = (explicit or fallback or "").strip()
        if not container:
            raise ValueError("A 'container_name' must be provided via params, metadata, or storage config.")
        return container

    def _resolve_video_folder(self, video_uri: str) -> str:
        name = Path(video_uri).name
        if not name:
            return "video"
        return name

    def _create_storage_provider(self) -> AzureStorageProvider:
        provider_name = (self.params.get("provider") or _config.storage.provider or "azure").lower()
        if provider_name not in {"azure", "azure_blob", "azure-storage"}:
            raise ValueError(f"Unsupported storage provider '{provider_name}' for frame blob export")

        config_payload = _config.storage.model_dump()
        overrides = self.params.get("provider_config") or {}
        config_payload.update(overrides)
        return AzureStorageProvider(config_payload)

    async def _upload_frames(
        self,
        *,
        storage_provider: AzureStorageProvider,
        frames: List[Dict[str, Any]],
        container_name: str,
        video_folder: str,
        concurrency: int,
    ) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(concurrency)

        async def upload_single(frame: Dict[str, Any]) -> Dict[str, Any]:
            local_path = frame.get("path")
            if not local_path:
                return {"status": "missing"}
            src = Path(local_path)
            if not src.is_file():
                return {"status": "missing"}

            blob_name = f"{video_folder}/{src.name}"
            try:
                async with semaphore:
                    url = await storage_provider.save_file(
                        file_name=blob_name,
                        src_file_path=str(src),
                        folder_name=container_name,
                    )
                frame["blob_url"] = url
                return {
                    "status": "uploaded",
                    "url": url,
                }
            except Exception as exc:
                logger.warning(
                    "[{}] Failed to upload frame {}: {}",
                    self.step_id,
                    src,
                    exc,
                )
                return {"status": "failed"}

        results = await asyncio.gather(*(upload_single(frame) for frame in frames))

        uploaded = 0
        failed = 0
        missing = 0
        base_url: Optional[str] = None

        for item in results:
            status = item.get("status")
            if status == "uploaded":
                uploaded += 1
                url = item.get("url")
                if url and not base_url:
                    base_url = url.rsplit("/", 1)[0]
            elif status == "missing":
                missing += 1
            else:
                failed += 1

        return {
            "uploaded": uploaded,
            "failed": failed,
            "missing": missing,
            "base_url": base_url,
        }

    def _cleanup_provider(self, provider: AzureStorageProvider) -> None:
        try:
            asyncio.run(provider.close())
        except RuntimeError:
            # Ignore cleanup issues if already inside an event loop.
            pass
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug("[{}] Storage provider cleanup failed: {}", self.step_id, exc)

    async def _index_frames(
        self,
        *,
        frames: List[Dict[str, Any]],
        video_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        index_name: str,
        blob_base_url: str,
        embedding_concurrency: int,
        upload_batch_size: int,
        embedding_dimensions: int,
        image_provider,
        search_provider,
    ) -> Dict[str, int]:
        eligible_frames = [frame for frame in frames if self._frame_exists(frame) and frame.get("blob_url")]
        skipped_missing = len(frames) - len(eligible_frames)
        if not eligible_frames:
            raise RuntimeError("No frames with uploaded blob URLs were found to index.")

        await self._ensure_index(search_provider, index_name, embedding_dimensions)
        embeddings = await self._generate_embeddings(eligible_frames, embedding_concurrency, image_provider)

        documents: List[Dict[str, Any]] = []
        for item in embeddings:
            doc = self._build_document(
                item,
                video_id=video_id,
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=video_duration,
                blob_base_url=blob_base_url,
            )
            if doc:
                documents.append(doc)

        if not documents:
            raise RuntimeError("Failed to generate embeddings for any frames")

        uploaded = await self._upload_documents(search_provider, documents, index_name, upload_batch_size)

        return {
            "frames_processed": len(eligible_frames),
            "frames_skipped": skipped_missing + (len(eligible_frames) - len(documents)),
            "documents_indexed": uploaded,
        }

    async def _ensure_index(self, search_provider, index_name: str, embedding_dimensions: int) -> None:
        try:
            if await search_provider.index_exists(index_name):
                return
        except Exception as exc:
            logger.warning("[{}] Index existence check failed: {}", self.step_id, exc)
            return

        logger.info("[{}] Creating keyframe index '{}'", self.step_id, index_name)
        await search_provider.create_index(
            index_name,
            {"type": "keyframe", "dim": embedding_dimensions},
        )

    async def _generate_embeddings(
        self,
        frames: List[Dict[str, Any]],
        concurrency: int,
        image_provider,
    ) -> List[_FrameEmbedding]:
        semaphore = asyncio.Semaphore(concurrency)
        embeddings: List[_FrameEmbedding] = []

        async def embed(frame: Dict[str, Any]) -> Optional[_FrameEmbedding]:
            path = frame.get("path")
            if not path:
                return None
            try:
                async with semaphore:
                    vector = await image_provider.image_embedding(path)
                return _FrameEmbedding(frame=frame, embedding=vector)
            except Exception as exc:
                logger.warning("[{}] Embedding failed for {}: {}", self.step_id, path, exc)
                return None

        for batch in _chunked(frames, max(1, concurrency) * 2):
            results = await asyncio.gather(*(embed(frame) for frame in batch))
            embeddings.extend(item for item in results if item is not None)

        return embeddings

    async def _upload_documents(
        self,
        search_provider,
        documents: List[Dict[str, Any]],
        index_name: str,
        batch_size: int,
    ) -> int:
        uploaded = 0
        for batch in _chunked(documents, batch_size):
            await search_provider.upload_documents(batch, index_name=index_name)
            uploaded += len(batch)
            logger.info(
                "[{}] Uploaded {} / {} frame documents to '{}'",
                self.step_id,
                uploaded,
                len(documents),
                index_name,
            )
        return uploaded

    def _build_document(
        self,
        item: _FrameEmbedding,
        *,
        video_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        blob_base_url: str,
    ) -> Optional[Dict[str, Any]]:
        frame = item.frame
        embedding = item.embedding
        frame_path = frame.get("path")
        if not frame_path:
            return None
        timestamp = float(frame.get("timestamp") or 0.0)
        motion_score = float(frame.get("motion_score") or 0.0)
        frame_identifier = frame.get("frame_id") or Path(frame_path).stem
        deterministic_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{frame_identifier}")
        file_name = Path(frame_path).name
        blob_url = frame.get("blob_url")
        if not blob_url and blob_base_url:
            blob_url = f"{blob_base_url.rstrip('/')}/{file_name}"
        elif not blob_url:
            blob_url = Path(frame_path).resolve().as_uri()

        document = KeyframeDocument(
            id=str(deterministic_id),
            video_id=video_id,
            keyframe_filename=file_name,
            embeddings=embedding,
            created_at=datetime.now(timezone.utc),
            motion_score=motion_score,
            timestamp_seconds=timestamp,
            blob_url=blob_url,
            parent_id=parent_id,
            parent_duration=float(parent_duration or 0.0),
            video_duration=float(video_duration or 0.0),
        )

        return document.model_dump(mode="json")

    def _resolve_video_id(self, context: StepContext) -> str:
        if self.params.get("video_id"):
            return str(self.params["video_id"])
        video_id = context.video_id
        if video_id:
            return str(video_id)
        raise ValueError(
            "'video_id' must be provided via params or experiment metadata before uploading frames."
        )

    def _resolve_index_name(self, video_id: str, metadata: Dict[str, Any]) -> str:
        if self.params.get("index_name"):
            return str(self.params["index_name"])
        if metadata.get("keyframe_index_name"):
            return str(metadata["keyframe_index_name"])
        configured = _config.search.index_name
        if configured:
            configured = str(configured)
            if configured.startswith("keyframes-"):
                return configured
            return f"keyframes-{configured}"
        return f"keyframes-{_slugify(video_id)}"

    @staticmethod
    def _frame_exists(frame: Dict[str, Any]) -> bool:
        path = frame.get("path")
        if not path:
            return False
        return Path(path).is_file()

    def _cleanup_index_providers(self, image_provider, search_provider) -> None:
        try:
            if hasattr(image_provider, "close"):
                image_provider.close()
        except Exception as exc:
            logger.debug("[{}] Image provider cleanup failed: {}", self.step_id, exc)

        try:
            asyncio.run(search_provider.close())
        except RuntimeError:
            pass
        except Exception as exc:
            logger.debug("[{}] Search provider cleanup failed: {}", self.step_id, exc)


def _chunked(sequence: Iterable[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        size = 1
    chunk: List[Any] = []
    for item in sequence:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _slugify(value: str) -> str:
    lowered = value.lower()
    cleaned = [ch if ch.isalnum() else "-" for ch in lowered]
    slug = "".join(cleaned).strip("-")
    slug = "-".join(filter(None, slug.split("-")))
    return slug or "video"
