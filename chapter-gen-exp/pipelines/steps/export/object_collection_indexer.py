"""Export step that indexes merged object collections into Azure AI Search."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from providers.factory import provider_factory
from providers.search_document_models import ObjectCollectionDocument

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


@register_step("export.object-collection-index")
class ObjectCollectionSearchIndexExporter(PipelineStep):
    """Uploads the enriched object roster and video summary to Azure AI Search."""

    description = "Pushes merged object collection JSON into an Azure AI Search index."

    def run(self, context: StepContext) -> StepResult:
        source_step = self.params.get("source_step")
        if not source_step:
            raise ValueError("'source_step' parameter is required")

        payload = context.data_store.get(source_step, {}) or {}
        collection_key = self.params.get("collection_key", "object_collection")
        object_collection = payload.get(collection_key) or []
        if not object_collection:
            raise ValueError(
                f"Step '{source_step}' did not provide '{collection_key}' payload for export."
            )

        video_summary = self._resolve_summary(payload, context)
        video_id = self._resolve_video_id(context)
        hash_video_id = str(
            self.params.get("hash_video_id")
            or context.metadata.get("hash_video_id")
            or video_id
        )
        youtube_url = self.params.get("youtube_url") or context.metadata.get("youtube_url") or ""
        index_name = self._resolve_index_name(video_id, context.metadata)
        video_duration = self._resolve_duration(context)

        search_provider = provider_factory.create_search_provider()
        embedding_provider = provider_factory.create_embedding_provider()

        try:
            stats = asyncio.run(
                self._index_object_collection(
                    object_collection=object_collection,
                    video_summary=video_summary,
                    hash_video_id=hash_video_id,
                    youtube_url=youtube_url,
                    video_duration=video_duration,
                    index_name=index_name,
                    search_provider=search_provider,
                    embedding_provider=embedding_provider,
                )
            )
        finally:
            self._cleanup_provider(embedding_provider)
            self._cleanup_provider(search_provider)

        produced = {
            "index_name": index_name,
            "video_id": video_id,
            "hash_video_id": hash_video_id,
            "object_count": stats.get("object_count", 0),
        }
        metrics = {
            "object_count": float(stats.get("object_count", 0)),
            "documents_indexed": float(stats.get("documents_indexed", 0)),
        }

        return StepResult(step_id=self.step_id, produced=produced, metrics=metrics)

    async def _index_object_collection(
        self,
        *,
        object_collection: List[Any],
        video_summary: str,
        hash_video_id: str,
        youtube_url: str,
        video_duration: float,
        index_name: str,
        search_provider,
        embedding_provider,
    ) -> Dict[str, Any]:
        await self._ensure_index(search_provider, index_name)

        document = await self._build_document(
            object_collection=object_collection,
            video_summary=video_summary,
            hash_video_id=hash_video_id,
            youtube_url=youtube_url,
            video_duration=video_duration,
            embedding_provider=embedding_provider,
        )

        await search_provider.upload_documents([document], index_name=index_name)
        return {
            "object_count": len(object_collection),
            "documents_indexed": 1,
        }

    async def _build_document(
        self,
        *,
        object_collection: List[Any],
        video_summary: str,
        hash_video_id: str,
        youtube_url: str,
        video_duration: float,
        embedding_provider,
    ) -> Dict[str, Any]:
        object_collection_json = self._serialize_object_collection(object_collection)
        summary_embedding: List[float] = []
        if video_summary and video_summary.strip():
            try:
                summary_embedding = await embedding_provider.embedding(video_summary)
            except Exception as exc:
                logger.warning(
                    "[{}] Video summary embedding failed for {}: {}",
                    self.step_id,
                    hash_video_id,
                    exc,
                )
                summary_embedding = []

        doc = ObjectCollectionDocument(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{hash_video_id}:object-collection")),
            video_id=hash_video_id,
            url=youtube_url or "",
            object_collection=object_collection_json,
            object_count=len(object_collection),
            video_summary=video_summary or "",
            video_summary_embedding=summary_embedding,
            video_duration=float(video_duration or 0.0),
        )

        return doc.model_dump(mode="json")

    async def _ensure_index(self, search_provider, index_name: str) -> None:
        try:
            exists = await search_provider.index_exists(index_name)
        except Exception as exc:
            logger.warning(
                "[{}] Object index existence check failed for '{}': {}",
                self.step_id,
                index_name,
                exc,
            )
            return

        if exists:
            return

        logger.info("[{}] Creating object collection index '{}'", self.step_id, index_name)
        await search_provider.create_index(index_name, "object-collection")

    def _serialize_object_collection(self, collection: List[Any]) -> str:
        normalized: List[Any] = []
        for item in collection:
            if hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            elif isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "__dict__"):
                normalized.append({k: v for k, v in vars(item).items() if not k.startswith("_")})
            else:
                normalized.append(item)

        try:
            return json.dumps(normalized, ensure_ascii=False)
        except TypeError:
            safe_normalized = json.loads(json.dumps(normalized, default=str))
            return json.dumps(safe_normalized, ensure_ascii=False)

    def _resolve_summary(self, payload: Dict[str, Any], context: StepContext) -> str:
        if self.params.get("video_summary") is not None:
            return str(self.params["video_summary"])
        summary_key = self.params.get("summary_key", "video_summary")
        if payload.get(summary_key):
            return str(payload[summary_key])
        if payload.get("global_summary"):
            return str(payload["global_summary"])
        if context.metadata.get("video_summary"):
            return str(context.metadata["video_summary"])
        if context.metadata.get("global_summary"):
            return str(context.metadata["global_summary"])
        return ""

    def _resolve_video_id(self, context: StepContext) -> str:
        if self.params.get("video_id"):
            return str(self.params["video_id"])
        if context.metadata.get("video_id"):
            return str(context.metadata["video_id"])
        return self._slugify(Path(context.video_uri).stem)

    def _resolve_index_name(self, video_id: str, metadata: Dict[str, Any]) -> str:
        if self.params.get("index_name"):
            return str(self.params["index_name"])
        if metadata.get("object_collection_index_name"):
            return str(metadata["object_collection_index_name"])
        return f"objects-{self._slugify(video_id)}"

    def _resolve_duration(self, context: StepContext) -> float:
        if self.params.get("video_duration_seconds") is not None:
            return float(self.params["video_duration_seconds"])
        if context.metadata.get("video_duration_seconds") is not None:
            return float(context.metadata["video_duration_seconds"])
        if context.video_duration_seconds is not None:
            return float(context.video_duration_seconds)
        return 0.0

    def _cleanup_provider(self, provider: Any) -> None:
        if not provider:
            return
        close_fn = getattr(provider, "close", None)
        if not close_fn:
            return
        try:
            if asyncio.iscoroutinefunction(close_fn):
                asyncio.run(close_fn())
            else:
                close_fn()
        except RuntimeError as exc:
            logger.debug("[{}] Provider cleanup skipped: {}", self.step_id, exc)
        except Exception as exc:
            logger.debug("[{}] Provider cleanup failed: {}", self.step_id, exc)

    @staticmethod
    def _slugify(value: str) -> str:
        lowered = value.lower()
        cleaned = [ch if ch.isalnum() else "-" for ch in lowered]
        slug = "".join(cleaned).strip("-")
        slug = "-".join(filter(None, slug.split("-")))
        return slug or "video"