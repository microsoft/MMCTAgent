"""Object-level enrichment utilities shared across chapter processing steps."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Optional

from loguru import logger
from pydantic import BaseModel, Field

from mmct.providers.base.llm_provider import BaseLLMProvider
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse, ObjectResponse


@dataclass
class ChapterObjectBundle:
    """Wrapper for chapter metadata and its local object collection."""

    chunk_index: int
    start: float
    end: float
    transcript: str
    chapter_summary: str
    actions: Optional[str]
    objects: List[ObjectResponse]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ObjectRosterResults:
    """Aggregated roster output produced after processing all chapters."""

    object_collection: List[ObjectResponse]
    operations: List[Dict[str, Any]]
    stats: Dict[str, Dict[str, float]]
    filtered_out: int


class ObjectDelta(BaseModel):
    """Structured change command emitted by the LLM."""

    action: str = Field(..., description="Mutation type: add / update / remove")
    name: str = Field(..., description="Canonical object name or identity key")
    appearance: Optional[List[str]] = Field(
        None, description="Appearance descriptors to add/replace"
    )
    identity: Optional[List[str]] = Field(None, description="Identity descriptors to add/replace")
    first_seen: Optional[float] = Field(
        None, description="Timestamp (seconds) when the object first appeared"
    )
    additional_details: Optional[str] = Field(
        None, description="Free-form context or behavior notes"
    )


class ObjectEnrichmentResponse(BaseModel):
    """LLM response schema for active-object updates."""

    operations: List[ObjectDelta] = Field(
        ...,
        description="Ordered list of add/update/remove commands to align the active roster with the current chapter",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional reasoning or context for auditing. Ignored by downstream logic.",
    )


class ObjectRosterManager:
    """Reusable orchestrator that maintains a deduplicated object roster."""

    def __init__(
        self,
        *,
        step_id: str,
        llm_request_options: Optional[Dict[str, Any]] = None,
        max_active_context: int = 12,
        min_screen_time_seconds: float = 8.0,
        min_chunk_occurrences: int = 2,
        llm_client: BaseLLMProvider,
    ) -> None:
        self._step_id = step_id
        self._llm = llm_client
        self._options = dict(llm_request_options or {})
        self._max_active_context = max_active_context
        self._min_screen_time = min_screen_time_seconds
        self._min_chunk_occurrences = min_chunk_occurrences
        self._active_objects: MutableMapping[str, ObjectResponse] = {}
        self._global_objects: MutableMapping[str, ObjectResponse] = {}
        self._operations_log: List[Dict[str, Any]] = []
        self._presence_stats: Dict[str, Dict[str, float]] = {}

    async def process_chapters(self, bundles: List[ChapterObjectBundle]) -> None:
        for bundle in bundles:
            await self.process_chapter(bundle)

    async def process_chapter(self, bundle: ChapterObjectBundle) -> None:
        active_snapshot = self._serialize_objects(
            self._active_objects, limit=self._max_active_context
        )
        chapter_objects = [obj.model_dump() for obj in bundle.objects]
        messages = self._build_messages(bundle, active_snapshot, chapter_objects)

        logger.info(
            "[{}] Reconciling objects for chunk {} ({} local objects, {} active)",
            self._step_id,
            bundle.chunk_index,
            len(chapter_objects),
            len(self._active_objects),
        )

        raw_response = await self._llm.chat_completion(
            messages,
            response_format=ObjectEnrichmentResponse,
            **self._options,
        )
        response = self._coerce_response(raw_response)

        summary = self._apply_operations(
            response.operations,
            self._active_objects,
            self._global_objects,
            default_first_seen=bundle.start,
        )

        self._update_presence_stats(
            self._presence_stats,
            list(self._active_objects.keys()),
            bundle.duration,
        )
        self._operations_log.append(
            {
                "chunk_index": bundle.chunk_index,
                "operations": [op.model_dump() for op in response.operations],
                "stats": summary,
            }
        )

    def finalize(self) -> ObjectRosterResults:
        combined_objects = []
        filtered_stats: Dict[str, Dict[str, float]] = {}
        filtered_out = 0

        for key, obj in self._global_objects.items():
            stats = self._presence_stats.get(key, {"chunks": 0.0, "duration": 0.0})
            if (
                stats["chunks"] >= self._min_chunk_occurrences
                and stats["duration"] >= self._min_screen_time
            ):
                combined_objects.append(obj)
                filtered_stats[key] = stats
            else:
                filtered_out += 1

        combined_objects.sort(key=lambda item: item.first_seen)

        return ObjectRosterResults(
            object_collection=combined_objects,
            operations=self._operations_log,
            stats=filtered_stats,
            filtered_out=filtered_out,
        )

    def _apply_operations(
        self,
        operations: List[ObjectDelta],
        active_objects: MutableMapping[str, ObjectResponse],
        global_objects: MutableMapping[str, ObjectResponse],
        *,
        default_first_seen: float,
    ) -> Dict[str, float]:
        stats = {"adds": 0.0, "updates": 0.0, "removals": 0.0}
        for op in operations:
            key = self._normalize_name(op.name)
            if not key:
                continue

            if op.action.lower() == "add":
                obj = self._object_from_delta(op, default_first_seen)
                merged = self._merge_objects(global_objects.get(key), obj)
                active_objects[key] = merged
                global_objects[key] = merged
                stats["adds"] += 1.0
            elif op.action.lower() == "update":
                existing = active_objects.get(key) or global_objects.get(key)
                if existing is None:
                    existing = self._object_from_delta(op, default_first_seen)
                updated = self._merge_objects(
                    existing, self._object_from_delta(op, existing.first_seen)
                )
                active_objects[key] = updated
                global_objects[key] = updated
                stats["updates"] += 1.0
            elif op.action.lower() == "remove":
                active_objects.pop(key, None)
                stats["removals"] += 1.0
        return stats

    def _object_from_delta(self, op: ObjectDelta, fallback_first_seen: float) -> ObjectResponse:
        appearance = op.appearance or []
        identity = op.identity or []
        return ObjectResponse(
            name=op.name,
            appearance=list(dict.fromkeys(appearance)),
            identity=list(dict.fromkeys(identity)),
            first_seen=op.first_seen if op.first_seen is not None else fallback_first_seen,
            additional_details=op.additional_details,
        )

    def _merge_objects(
        self,
        base: Optional[ObjectResponse],
        new_obj: ObjectResponse,
    ) -> ObjectResponse:
        if base is None:
            return new_obj

        base.first_seen = min(base.first_seen, new_obj.first_seen)
        base.appearance = list(dict.fromkeys(base.appearance + new_obj.appearance))
        base.identity = list(dict.fromkeys(base.identity + new_obj.identity))
        if new_obj.additional_details:
            if (
                base.additional_details
                and new_obj.additional_details not in base.additional_details
            ):
                base.additional_details = (
                    f"{base.additional_details}\n{new_obj.additional_details}".strip()
                )
            elif not base.additional_details:
                base.additional_details = new_obj.additional_details
        return base

    def _serialize_objects(
        self,
        active_objects: MutableMapping[str, ObjectResponse],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        items = list(active_objects.values())
        items.sort(key=lambda obj: obj.first_seen)
        if limit > 0:
            items = items[-limit:]
        return [obj.model_dump() for obj in items]

    def _update_presence_stats(
        self,
        stats: MutableMapping[str, Dict[str, float]],
        active_keys: List[str],
        duration: float,
    ) -> None:
        for key in active_keys:
            record = stats.setdefault(key, {"chunks": 0.0, "duration": 0.0})
            record["chunks"] += 1.0
            record["duration"] += duration

    def _build_messages(
        self,
        bundle: ChapterObjectBundle,
        active_objects: List[Dict[str, Any]],
        chapter_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        user_prompt = (
            f"Chunk Index: {bundle.chunk_index}\n"
            f"Start: {self._format_seconds(bundle.start)} ({bundle.start:.2f}s)\n"
            f"End: {self._format_seconds(bundle.end)} ({bundle.end:.2f}s)\n"
            f"Duration: {bundle.duration:.2f}s\n\n"
            "Current Chapter Summary:\n"
            f"{bundle.chapter_summary}\n\n"
            "Actions Reported:\n"
            f"{bundle.actions or 'None'}\n\n"
            "Transcript Snippet:\n"
            f"{bundle.transcript or 'No transcript text provided.'}\n\n"
            "Objects Observed In This Chapter (raw):\n"
            f"{json.dumps(chapter_objects, ensure_ascii=False)[:4000]}\n\n"
            "Active Objects Entering This Chapter:\n"
            f"{json.dumps(active_objects, ensure_ascii=False)[:4000]}\n\n"
            "Instructions:\n"
            "- Return JSON matching ObjectEnrichmentResponse (operations list only; notes optional).\n"
            "- Use 'add' to introduce brand-new objects, 'update' to append new appearance/identity info, and 'remove' when an object leaves the scene.\n"
            "- Prefer consistent naming. Reuse existing object names when referring to the same entity.\n"
            "- Provide appearance/identity arrays whenever you add or update an object.\n"
            "- Do NOT repeat the full active object roster; only emit the delta operations."
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are VideoObjectTrackerGPT. Maintain a consistent roster of objects across video chapters using structured add/update/remove operations."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    def _coerce_response(self, payload: Any) -> ObjectEnrichmentResponse:
        content: Any = payload
        if isinstance(payload, dict) and "content" in payload:
            content = payload["content"]

        if isinstance(content, ObjectEnrichmentResponse):
            return content

        if isinstance(content, BaseModel):
            return ObjectEnrichmentResponse.model_validate(content.model_dump())

        if isinstance(content, dict):
            return ObjectEnrichmentResponse.model_validate(content)

        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as err:
                raise ValueError("LLM provider returned non-JSON string content") from err
            return ObjectEnrichmentResponse.model_validate(parsed)

        raise TypeError(f"Unsupported object enrichment response type: {type(payload)!r}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0.0, value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
