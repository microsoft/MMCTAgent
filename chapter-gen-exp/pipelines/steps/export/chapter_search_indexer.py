"""Export step that indexes enriched chapters into Azure AI Search."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

from providers.base.search_provider import SearchProvider
from providers.factory import provider_factory
from providers.search_document_models import ChapterIndexDocument
from settings import MMCTConfig

from ..base import PipelineStep, StepContext, StepResult
from ..chapters.models import ChapterCreationResponse
from ..registry import register_step

_config = MMCTConfig()


@register_step("export.chapter-search-index")
class ChapterSearchIndexExporter(PipelineStep):
    """Generates embeddings for chapters and pushes them to Azure AI Search."""

    description = "Uploads enriched chapter documents to Azure AI Search."

    def run(self, context: StepContext) -> StepResult:
        chapters_step = self.params.get("chapters_step")
        if not chapters_step:
            raise ValueError("'chapters_step' parameter is required")

        chapters_payload = context.data_store.get(chapters_step, {}) or {}
        chapters: List[Dict[str, Any]] = chapters_payload.get("chapters", []) or []
        if not chapters:
            raise ValueError(
                f"Step '{chapters_step}' did not produce chapter payloads for export."
            )

        video_id = self._resolve_video_id(context)
        hash_video_id = video_id
        parent_id = video_id
        parent_duration = self._resolve_duration(
            key="parent_duration_seconds",
            context=context,
            fallback=context.video_duration_seconds,
        )
        video_duration = self._resolve_duration(
            key="video_duration_seconds",
            context=context,
            fallback=context.video_duration_seconds,
        )

        index_name = self._resolve_index_name(video_id, context.metadata)
        upload_batch_size = int(self.params.get("upload_batch_size", 32))
        embedding_concurrency = int(self.params.get("embedding_concurrency", 4))
        skip_if_exists = bool(self.params.get("skip_if_exists", True))

        search_provider: SearchProvider = provider_factory.create_search_provider()
        embedding_provider = provider_factory.create_embedding_provider()

        resolved_strings = self._build_string_overrides(context, chapters_payload)

        try:
            stats = asyncio.run(
                self._index_chapters(
                    chapters=chapters,
                    hash_video_id=hash_video_id,
                    parent_id=parent_id,
                    parent_duration=parent_duration,
                    video_duration=video_duration,
                    video_id=video_id,
                    index_name=index_name,
                    string_fields=resolved_strings,
                    upload_batch_size=max(1, upload_batch_size),
                    embedding_concurrency=max(1, embedding_concurrency),
                    skip_if_exists=skip_if_exists,
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
            "chapters_indexed": stats.get("documents_indexed", 0),
            "chapters_processed": stats.get("chapters_processed", len(chapters)),
            "chapters_skipped": stats.get("chapters_skipped", 0),
            "already_exists": stats.get("already_exists", False),
        }
        metrics = {
            "chapters_indexed": float(produced["chapters_indexed"]),
            "chapters_processed": float(produced["chapters_processed"]),
        }

        return StepResult(
            step_id=self.step_id,
            produced=produced,
            metrics=metrics,
        )

    async def _index_chapters(
        self,
        *,
        chapters: List[Dict[str, Any]],
        hash_video_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        video_id: str,
        index_name: str,
        string_fields: Dict[str, str],
        upload_batch_size: int,
        embedding_concurrency: int,
        skip_if_exists: bool,
        search_provider,
        embedding_provider,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "chapters_processed": len(chapters),
            "chapters_skipped": 0,
            "documents_indexed": 0,
            "already_exists": False,
        }

        if skip_if_exists:
            try:
                exists = await search_provider.check_is_document_exist(
                    hash_id=hash_video_id,
                    index_name=index_name,
                )
            except Exception as exc:
                logger.warning(
                    "[{}] Duplicate check failed for video {}: {}",
                    self.step_id,
                    hash_video_id,
                    exc,
                )
            else:
                if exists:
                    logger.info(
                        "[{}] Skipping ingestion for {} because documents already exist",
                        self.step_id,
                        hash_video_id,
                    )
                    stats["already_exists"] = True
                    return stats

        await self._ensure_index(search_provider, index_name)

        documents = await self._build_documents(
            chapters=chapters,
            hash_video_id=hash_video_id,
            parent_id=parent_id,
            parent_duration=parent_duration,
            video_duration=video_duration,
            string_fields=string_fields,
            video_id=video_id,
            embedding_concurrency=embedding_concurrency,
            embedding_provider=embedding_provider,
        )

        if not documents:
            raise RuntimeError("No chapter documents were created for indexing")

        stats["chapters_skipped"] = len(chapters) - len(documents)

        uploaded = await self._upload_documents(
            search_provider=search_provider,
            documents=documents,
            index_name=index_name,
            batch_size=upload_batch_size,
        )
        stats["documents_indexed"] = uploaded
        return stats

    async def _build_documents(
        self,
        *,
        chapters: List[Dict[str, Any]],
        hash_video_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        string_fields: Dict[str, str],
        video_id: str,
        embedding_concurrency: int,
        embedding_provider,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, embedding_concurrency))
        tasks = []

        for idx, entry in enumerate(chapters):
            tasks.append(
                asyncio.create_task(
                    self._build_single_document(
                        entry=entry,
                        chunk_fallback_index=idx,
                        hash_video_id=hash_video_id,
                        parent_id=parent_id,
                        parent_duration=parent_duration,
                        video_duration=video_duration,
                        string_fields=string_fields,
                        video_id=video_id,
                        semaphore=semaphore,
                        embedding_provider=embedding_provider,
                    )
                )
            )

        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    async def _build_single_document(
        self,
        *,
        entry: Dict[str, Any],
        chunk_fallback_index: int,
        hash_video_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        string_fields: Dict[str, str],
        video_id: str,
        semaphore: asyncio.Semaphore,
        embedding_provider,
    ) -> Optional[Dict[str, Any]]:
        chapter_data = entry.get("chapter") or entry.get("chapter_data")
        if not chapter_data:
            logger.warning(
                "[{}] Chapter entry {} missing 'chapter' payload; skipping",
                self.step_id,
                chunk_fallback_index,
            )
            return None

        try:
            if isinstance(chapter_data, ChapterCreationResponse):
                chapter = chapter_data
            else:
                chapter = ChapterCreationResponse.model_validate(chapter_data)
        except Exception as exc:
            logger.warning(
                "[{}] Failed to parse chapter payload for chunk {}: {}",
                self.step_id,
                chunk_fallback_index,
                exc,
            )
            return None

        transcript = self._extract_transcript(entry)
        embedding_text = chapter.__str__(transcript=transcript)

        async with semaphore:
            try:
                embedding_vector = await embedding_provider.embedding(embedding_text)
            except Exception as exc:
                logger.warning(
                    "[{}] Embedding failed for chunk {}: {}",
                    self.step_id,
                    chunk_fallback_index,
                    exc,
                )
                return None

        chunk_index = int(entry.get("chunk_index", chunk_fallback_index))
        start_time = float(entry.get("start", 0.0))
        end_time = float(entry.get("end", 0.0))
        document_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{hash_video_id}:{chunk_index}:{start_time:.3f}:{end_time:.3f}",
        )

        document = ChapterIndexDocument(
            id=str(document_id),
            hash_video_id=hash_video_id,
            topic_of_video=string_fields["topic_of_video"],
            action_taken=chapter.action_taken or "None",
            detailed_summary=chapter.detailed_summary or "None",
            category=string_fields["category"],
            sub_category=string_fields["sub_category"],
            text_from_scene=chapter.text_from_scene or "None",
            object_collection=self._serialize_object_collection(chapter),
            youtube_url=string_fields["youtube_url"],
            chapter_transcript=transcript,
            parent_id=parent_id,
            parent_duration=str(parent_duration),
            video_duration=str(video_duration),
            start_time=start_time,
            end_time=end_time,
            blob_audio_url=string_fields["blob_audio_url"],
            blob_video_url=string_fields["blob_video_url"],
            blob_transcript_file_url=string_fields["blob_transcript_file_url"],
            blob_frames_folder_path=string_fields["blob_frames_folder_path"],
            embeddings=embedding_vector,
            time=datetime.now(timezone.utc),
        )

        return document.model_dump(mode="json")

    async def _ensure_index(self, search_provider, index_name: str) -> None:
        try:
            exists = await search_provider.index_exists(index_name)
        except Exception as exc:
            logger.warning(
                "[{}] Index existence check failed for '{}': {}",
                self.step_id,
                index_name,
                exc,
            )
            return

        if exists:
            return

        logger.info("[{}] Creating chapter index '{}'", self.step_id, index_name)
        await search_provider.create_index(index_name, "chapter")

    async def _upload_documents(
        self,
        *,
        search_provider,
        documents: List[Dict[str, Any]],
        index_name: str,
        batch_size: int,
    ) -> int:
        uploaded = 0
        for batch in self._chunked(documents, batch_size):
            await search_provider.upload_documents(batch, index_name=index_name)
            uploaded += len(batch)
            logger.info(
                "[{}] Uploaded {} / {} chapter documents to '{}'",
                self.step_id,
                uploaded,
                len(documents),
                index_name,
            )
        return uploaded

    def _build_string_overrides(
        self,
        context: StepContext,
        chapters_payload: Dict[str, Any],
    ) -> Dict[str, str]:
        global_summary = chapters_payload.get("global_summary") or context.metadata.get("global_summary")

        return {
            "topic_of_video": self._resolve_string(
                context,
                "topic_of_video",
                fallback="None",
            ),
            "category": self._resolve_string(context, "category", fallback="None"),
            "sub_category": self._resolve_string(context, "sub_category", fallback="None"),
            "youtube_url": self._resolve_string(context, "youtube_url", fallback="None"),
            "blob_video_url": self._resolve_string(context, "blob_video_url", fallback="None"),
            "blob_audio_url": self._resolve_string(context, "blob_audio_url", fallback="None"),
            "blob_transcript_file_url": self._resolve_string(
                context,
                "blob_transcript_file_url",
                fallback="None",
            ),
            "blob_frames_folder_path": self._resolve_string(
                context,
                "blob_frames_folder_path",
                fallback=context.metadata.get("frame_blob_base_url", "None"),
            ),
        }

    def _resolve_string(self, context: StepContext, key: str, fallback: Optional[str] = None) -> str:
        if self.params.get(key) is not None:
            return str(self.params[key])
        if context.metadata.get(key) is not None:
            return str(context.metadata[key])
        if fallback is not None:
            return str(fallback)
        return "None"

    def _resolve_duration(self, *, key: str, context: StepContext, fallback: Optional[float]) -> float:
        if self.params.get(key) is not None:
            return float(self.params[key])
        if context.metadata.get(key) is not None:
            return float(context.metadata[key])
        if fallback is not None:
            return float(fallback)
        return 0.0

    def _resolve_video_id(self, context: StepContext) -> str:
        if self.params.get("video_id"):
            return str(self.params["video_id"])
        if context.metadata.get("video_id"):
            return str(context.metadata["video_id"])
        return self._slugify(Path(context.video_uri).stem)

    def _resolve_index_name(self, video_id: str, metadata: Dict[str, Any]) -> str:
        if self.params.get("index_name"):
            return str(self.params["index_name"])
        if metadata.get("chapter_index_name"):
            return str(metadata["chapter_index_name"])
        configured = _config.search.index_name
        if configured and configured != "default":
            return str(configured)
        return f"chapters-{self._slugify(video_id)}"

    def _extract_transcript(self, entry: Dict[str, Any]) -> str:
        segments = entry.get("transcript_segments") or []
        if segments:
            formatted = self._format_timestamped_transcript(segments)
            if formatted:
                return formatted

        transcript = entry.get("transcript")
        if transcript:
            return str(transcript)

        return ""

    @staticmethod
    def _format_timestamped_transcript(segments: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = ChapterSearchIndexExporter._format_timestamp(seg.get("start", 0.0))
            end = ChapterSearchIndexExporter._format_timestamp(seg.get("end", seg.get("start", 0.0)))
            lines.append(f"[{start} - {end}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        try:
            total_seconds = max(0.0, float(value))
        except (TypeError, ValueError):
            total_seconds = 0.0

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        millis = int(round((total_seconds - int(total_seconds)) * 1000))

        if millis == 1000:
            millis = 0
            seconds += 1
            if seconds == 60:
                seconds = 0
                minutes += 1
                if minutes == 60:
                    minutes = 0
                    hours += 1

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _serialize_object_collection(self, chapter: ChapterCreationResponse) -> str:
        if not chapter.object_collection:
            return "[]"
        try:
            payload = [obj.model_dump() for obj in chapter.object_collection]
            return json.dumps(payload)
        except Exception as exc:
            logger.debug("[{}] Object serialization failed: {}", self.step_id, exc)
            return "[]"

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
    def _chunked(
        sequence: List[Dict[str, Any]],
        size: int,
    ) -> Iterator[List[Dict[str, Any]]]:
        if size <= 0:
            size = 1
        chunk: List[Dict[str, Any]] = []
        for item in sequence:
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    @staticmethod
    def _slugify(value: str) -> str:
        lowered = value.lower()
        cleaned = [ch if ch.isalnum() else "-" for ch in lowered]
        slug = "".join(cleaned).strip("-")
        slug = "-".join(filter(None, slug.split("-")))
        return slug or "video"