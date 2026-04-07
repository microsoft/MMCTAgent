"""Dense chapter extraction with multimodal LLM support.

This module handles LLM calls and response parsing.
All prompt/message construction is in prompts.py.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

from pydantic import BaseModel

from mmct.providers.base import BaseLLMProvider
from mmct.video_pipeline.core.ingestion.models import (
    ExtractionPlan,
    ExtractionCircuitBreaker,
    DenseChapterResponse,
)
from .prompts import build_dense_messages, build_basic_messages

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_CHUNKS = 4
DEFAULT_MAX_FRAMES_PER_CHAPTER = 12


def parse_llm_response(response: Any) -> Optional[DenseChapterResponse]:
    """Parse LLM response into DenseChapterResponse instance.
    
    Args:
        response: Raw response from LLM provider
        
    Returns:
        DenseChapterResponse instance or None on failure
    """
    content = response
    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    
    if isinstance(content, DenseChapterResponse):
        return content
    
    if isinstance(content, BaseModel):
        return DenseChapterResponse.model_validate(content.model_dump())
    
    if isinstance(content, dict):
        return DenseChapterResponse.model_validate(content)
    
    if isinstance(content, str):
        try:
            content = content.strip('```json').strip('```')
            parsed = json.loads(content)
            return DenseChapterResponse.model_validate(parsed)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse response: {e}")
            return None
    
    return None


async def extract_chapter_dense(
    chunk: Dict[str, Any],
    keyframes: Dict[str, Any],
    llm_provider: BaseLLMProvider,
    max_frames: int = DEFAULT_MAX_FRAMES_PER_CHAPTER,
) -> Optional[DenseChapterResponse]:
    """Extract dense chapter data using multimodal LLM.
    
    Args:
        chunk: Chunk data with transcript
        keyframes: Keyframe data with frame paths
        llm_provider: LLM provider instance
        max_frames: Maximum frames to send
        
    Returns:
        DenseChapterResponse instance or None on failure
    """
    messages = build_dense_messages(chunk, keyframes, max_frames)
    
    try:
        response = await llm_provider.chat_completion(
            messages,
            response_format=DenseChapterResponse,
        )
        return parse_llm_response(response)
    except Exception as e:
        logger.error(f"Dense extraction failed: {e}")
        return None


async def extract_chapter_basic(
    chunk: Dict[str, Any],
    keyframes: Dict[str, Any],
    llm_provider: BaseLLMProvider,
) -> Optional[DenseChapterResponse]:
    """Simplified extraction with fewer frames.
    
    Args:
        chunk: Chunk data with transcript
        keyframes: Keyframe data with frame paths
        llm_provider: LLM provider instance
        
    Returns:
        DenseChapterResponse instance or None on failure
    """
    messages = build_basic_messages(chunk, keyframes, max_frames=4)
    
    try:
        response = await llm_provider.chat_completion(
            messages,
            response_format={"type": "json_object"},
        )
        
        content = response.get("content", "{}")
        if isinstance(content, str):
            parsed = json.loads(content)
        elif isinstance(content, dict):
            parsed = content
        else:
            return None
        
        return DenseChapterResponse(
            timestamped_description=parsed.get("timestamped_description", "") or parsed.get("summary", ""),
            scene_composition=None,
            ocr_data=None,
        )
    except Exception as e:
        logger.error(f"Basic extraction failed: {e}")
        return None


async def extract_chapter_with_fallback(
    chunk: Dict[str, Any],
    keyframes: Dict[str, Any],
    llm_provider: BaseLLMProvider,
    circuit_breaker: ExtractionCircuitBreaker,
    extraction_plan: Optional[ExtractionPlan],
    max_frames: int = DEFAULT_MAX_FRAMES_PER_CHAPTER,
) -> Optional[DenseChapterResponse]:
    """Extract with fallback: dense -> basic -> transcript-only.
    
    Args:
        chunk: Chunk data with transcript
        keyframes: Keyframe data
        llm_provider: LLM provider
        circuit_breaker: Circuit breaker for error handling
        extraction_plan: Optional extraction configuration
        max_frames: Maximum frames to send
        
    Returns:
        DenseChapterResponse instance or None
    """
    if not circuit_breaker.should_allow_request():
        return await extract_chapter_basic(chunk, keyframes, llm_provider)
    
    max_retries = extraction_plan.max_retries_per_chapter if extraction_plan else 3
    
    # Try dense extraction
    for attempt in range(max_retries):
        try:
            result = await extract_chapter_dense(chunk, keyframes, llm_provider, max_frames)
            if result:
                circuit_breaker.record_success()
                return result
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            circuit_breaker.record_failure()
            if circuit_breaker.is_open:
                break
    
    # Fallback to basic
    try:
        result = await extract_chapter_basic(chunk, keyframes, llm_provider)
        if result:
            circuit_breaker.record_success()
            return result
    except Exception as e:
        logger.error(f"Basic extraction failed: {e}")
        circuit_breaker.record_failure()
    
    # Last resort: transcript-only
    transcript = chunk.get("transcript", "")
    return DenseChapterResponse(
        timestamped_description=transcript[:500] if transcript else "No content extracted",
        scene_composition=None,
        ocr_data=None,
    )


async def extract_chapters_parallel(
    chunks_with_keyframes: List[Tuple[int, Dict, Dict]],
    llm_provider: BaseLLMProvider,
    circuit_breaker: ExtractionCircuitBreaker,
    extraction_plan: Optional[ExtractionPlan],
    parallel_chunks: int = DEFAULT_PARALLEL_CHUNKS,
    max_frames_per_chapter: int = DEFAULT_MAX_FRAMES_PER_CHAPTER,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Process chunks in parallel batches.
    
    Args:
        chunks_with_keyframes: List of (chunk_idx, chunk, keyframes) tuples
        llm_provider: LLM provider
        circuit_breaker: Circuit breaker
        extraction_plan: Optional extraction config
        parallel_chunks: Batch size for parallel processing
        max_frames_per_chapter: Max frames per chapter
        
    Returns:
        Tuple of (chapter dicts with metadata, failed indices list)
        Each chapter dict includes: start, end, transcript, summary, scene_composition, ocr_data
    """
    chapters: List[Dict[str, Any]] = []
    failed: List[int] = []
    
    for batch_start in range(0, len(chunks_with_keyframes), parallel_chunks):
        batch = chunks_with_keyframes[batch_start:batch_start + parallel_chunks]
        
        tasks = [
            extract_chapter_with_fallback(
                chunk=chunk,
                keyframes=kf,
                llm_provider=llm_provider,
                circuit_breaker=circuit_breaker,
                extraction_plan=extraction_plan,
                max_frames=max_frames_per_chapter,
            )
            for _, chunk, kf in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (chunk_idx, chunk, _), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Chapter {chunk_idx} failed: {result}")
                circuit_breaker.record_failure()
                failed.append(chunk_idx)
            elif result is None:
                failed.append(chunk_idx)
            else:
                # Merge chunk metadata with extraction result
                # Note: video_chunking step uses start_time/end_time, normalize to start/end
                result_data = result.model_dump()
                # Derive summary from timestamped_description
                result_data["summary"] = result.summary
                chapter_data = {
                    "start": chunk.get("start_time", chunk.get("start", 0.0)),
                    "end": chunk.get("end_time", chunk.get("end", 0.0)),
                    "transcript": chunk.get("transcript", ""),
                    "chunk_index": chunk_idx,
                    **result_data,
                }
                chapters.append(chapter_data)
    
    return chapters, failed
