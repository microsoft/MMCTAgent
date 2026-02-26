"""
Dense chapter extraction prompts and multimodal message building.

All prompt strings and message construction logic lives here.
The extractor module only handles LLM calls and response parsing.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DENSE_SYSTEM_PROMPT = (
    "You are a VideoAnalyzerGPT specialized in dense video chapter analysis. "
    "Analyze the provided sequence of frames (in chronological order) along with the transcript "
    "to produce a comprehensive DenseChapterResponse. "
    "Pay attention to visual continuity between frames - they show a progression of the scene. "
    "Extract scene composition details from the visual content and any visible text (OCR)."
)

BASIC_SYSTEM_PROMPT = "Analyze the frames and transcript to provide a concise summary."


# =============================================================================
# FRAME UTILITIES
# =============================================================================

def encode_frame(filepath: str) -> Optional[str]:
    """Encode a frame image to base64.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Frame file not found: {filepath}")
            return None
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode frame {filepath}: {e}")
        return None


def format_frame_timeline(keyframes: List[Dict[str, Any]]) -> str:
    """Format frame timeline for the text prompt.
    
    Args:
        keyframes: List of keyframe dicts with timestamp and filename
        
    Returns:
        Formatted string showing frame sequence with timestamps
    """
    if not keyframes:
        return "No frames available."
    
    lines = []
    for idx, kf in enumerate(keyframes, start=1):
        timestamp = kf.get("timestamp", 0)
        filename = kf.get("filename", f"frame_{idx}")
        
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        millis = int((timestamp - int(timestamp)) * 1000)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
        
        lines.append(f"Frame {idx}: {time_str} ({timestamp:.2f}s) - {filename}")
    
    return "\n".join(lines)


def get_sorted_frames(keyframes: Dict[str, Any], max_frames: int) -> List[Dict[str, Any]]:
    """Get frames sorted by timestamp, limited to max_frames.
    
    Args:
        keyframes: Keyframe data dict with 'keyframes' list
        max_frames: Maximum number of frames to return
        
    Returns:
        Sorted and limited list of keyframe dicts
    """
    frames_list = keyframes.get("keyframes", [])
    sorted_frames = sorted(frames_list, key=lambda f: f.get("timestamp", 0))
    
    if len(sorted_frames) <= max_frames:
        return sorted_frames
    
    # Sample evenly to maintain temporal coverage
    step = len(sorted_frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [sorted_frames[i] for i in indices]


def get_schema_string() -> str:
    """Get the DenseChapterResponse JSON schema as a string."""
    from mmct.video_pipeline.core.ingestion.models import DenseChapterResponse
    
    schema = DenseChapterResponse.model_json_schema()
    
    def clean_schema(obj):
        if isinstance(obj, dict):
            return {k: clean_schema(v) for k, v in obj.items() 
                   if k not in ('title', '$defs', 'additionalProperties')}
        elif isinstance(obj, list):
            return [clean_schema(item) for item in obj]
        return obj
    
    def resolve_refs(obj, defs):
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref_name = obj['$ref'].split('/')[-1]
                if ref_name in defs:
                    return resolve_refs(defs[ref_name], defs)
            return {k: resolve_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item, defs) for item in obj]
        return obj
    
    defs = schema.get('$defs', {})
    resolved = resolve_refs(schema, defs)
    cleaned = clean_schema(resolved)
    
    return json.dumps(cleaned, indent=2)


def get_guidelines() -> str:
    """Get extraction guidelines from the model."""
    from mmct.video_pipeline.core.ingestion.models import DenseChapterResponse
    return DenseChapterResponse.get_extraction_guidelines()


# =============================================================================
# MULTIMODAL MESSAGE BUILDERS
# =============================================================================

def build_dense_messages(
    chunk: Dict[str, Any],
    keyframes: Dict[str, Any],
    max_frames: int = 12,
) -> List[Dict[str, Any]]:
    """Build multimodal messages for dense chapter extraction.
    
    Frames are stacked in chronological order for visual continuity.
    
    Args:
        chunk: Chunk data with transcript
        keyframes: Keyframe data with list of frames
        max_frames: Maximum frames to include
        
    Returns:
        List of messages for chat_completion API
    """
    transcript = chunk.get("transcript", "")
    sorted_frames = get_sorted_frames(keyframes, max_frames)
    
    # Build user content: images first (stacked), then text
    user_content: List[Dict[str, Any]] = []
    
    # Add frames in chronological order
    encoded_count = 0
    for kf in sorted_frames:
        filepath = kf.get("filepath")
        if not filepath:
            continue
        encoded = encode_frame(filepath)
        if encoded:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
            })
            encoded_count += 1
    
    # Build text content
    frame_timeline = format_frame_timeline(sorted_frames)
    schema_str = get_schema_string()
    guidelines = get_guidelines()
    
    text_content = f"""## FRAME SEQUENCE
The {encoded_count} frames above are shown in chronological order, capturing the visual progression of this chapter.

## FRAME TIMELINE
{frame_timeline}

## TRANSCRIPT
{transcript or "No transcript available."}

## TASK
Analyze the visual content from the frame sequence AND the transcript to extract:
1. A comprehensive summary combining what you SEE in the frames with what is SAID in the transcript
2. Scene composition details (environment, lighting, camera angle, spatial layout)
3. Any visible text in the frames (OCR data)

## RESPONSE SCHEMA
Return JSON matching this schema:
```json
{schema_str}
```

## GUIDELINES
{guidelines}

Return ONLY valid JSON, no additional text."""

    user_content.append({"type": "text", "text": text_content})
    
    return [
        {"role": "system", "content": DENSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_basic_messages(
    chunk: Dict[str, Any],
    keyframes: Dict[str, Any],
    max_frames: int = 4,
) -> List[Dict[str, Any]]:
    """Build simplified multimodal messages for basic extraction.
    
    Args:
        chunk: Chunk data with transcript
        keyframes: Keyframe data with list of frames
        max_frames: Maximum frames to include
        
    Returns:
        List of messages for chat_completion API
    """
    transcript = chunk.get("transcript", "")
    sorted_frames = get_sorted_frames(keyframes, max_frames)
    
    user_content: List[Dict[str, Any]] = []
    
    # Add frames
    for kf in sorted_frames:
        filepath = kf.get("filepath")
        if filepath:
            encoded = encode_frame(filepath)
            if encoded:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                })
    
    user_content.append({
        "type": "text",
        "text": f"""Transcript: {transcript or "No transcript"}

Provide a brief summary of what happens in this video segment.

Return JSON: {{"summary": "2-3 sentence summary"}}"""
    })
    
    return [
        {"role": "system", "content": BASIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
