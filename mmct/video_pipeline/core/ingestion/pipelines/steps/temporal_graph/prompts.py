"""LLM prompts for temporal graph event and object extraction.

This module provides prompt templates and utilities for extracting:
- Events: Atomic actions/occurrences from multimodal input (frames + transcript)
- Objects: Identifiable entities from visual input only (frames)

Prompt Design Principles:
1. Structured JSON output for reliable parsing
2. Clear field descriptions matching GraphEvent/GraphObject models
3. Examples for complex fields (appearance, identity)
4. Explicit constraints on output format
5. Timestamp-aware extraction for event-object linking

Frame Processing:
- Frames are sorted chronologically and sampled uniformly if exceeding max
- Each frame includes timestamp for temporal grounding
- Base64 encoding for multimodal LLM input
"""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger


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


def build_image_content(keyframes: Dict[str, Any], max_frames: int) -> tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Build image content blocks for multimodal messages.
    
    Args:
        keyframes: Keyframe data dict with 'keyframes' list
        max_frames: Maximum number of frames to include
        
    Returns:
        Tuple of (image_content_blocks, encoded_count, sorted_frames)
    """
    sorted_frames = get_sorted_frames(keyframes, max_frames)
    image_content: List[Dict[str, Any]] = []
    encoded_count = 0
    failed_count = 0
    
    for kf in sorted_frames:
        filepath = kf.get("filepath")
        if not filepath:
            failed_count += 1
            continue
        encoded = encode_frame(filepath)
        if encoded:
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
            })
            encoded_count += 1
        else:
            failed_count += 1
    
    if failed_count > 0:
        logger.warning(f"Frame encoding: {failed_count}/{len(sorted_frames)} failed")
    
    return image_content, encoded_count, sorted_frames


# =============================================================================
# EVENT EXTRACTION PROMPTS (Multimodal: frames + transcript + summary)
# =============================================================================

EVENT_EXTRACTION_SYSTEM_PROMPT: str = """You are a precise video analysis system that extracts atomic events from video chapters.
Your task is to identify discrete, meaningful events that occur within the video content.
Each event should be a single, complete action or occurrence that can stand alone.

You will analyze BOTH visual content (frame sequence) AND audio content (transcript).

Rules:
1. Events must be atomic - one action per event
2. Events must have clear start/end times based on the transcript or visual context
3. Identify participants (people, objects) involved in each event
4. Classify events into types: action, dialogue, transition, state_change
5. Maintain chronological sequence numbering
6. Be specific and descriptive in event descriptions
7. Combine visual observations with transcript context

Always return valid JSON matching the specified schema."""


# =============================================================================
# OBJECT EXTRACTION PROMPTS (Multimodal: frames + transcript for name resolution)
# =============================================================================

OBJECT_EXTRACTION_SYSTEM_PROMPT: str = """You are a precise video analysis system that extracts identifiable objects from video frames, using transcript context to identify names.
Your task is to identify and describe objects, people, and entities that you can SEE in the provided frames.
Each object should have clear identifying characteristics that allow it to be tracked across the video.

IMPORTANT: 
- Extract objects based on what you observe visually in the frames.
- Use the transcript to identify NAMES of people/objects when they are mentioned.
- If someone visible is named in the transcript, use their actual name (e.g., "Jitendra Kumar Kushwaha" not "man in blue shirt").

Rules:
1. Objects must be visually distinct and trackable
2. Include both primary subjects and significant background objects
3. Provide detailed appearance descriptions based on visual observation
4. Include identity information (type, category, role, brand if visible)
5. Note when objects first appear based on frame timestamps
6. Classify objects by type: person, item, animal, text, vehicle, etc.
7. Cross-reference transcript to use actual names when available

Always return valid JSON matching the specified schema."""


OBJECT_EXTRACTION_USER_PROMPT: str = """Extract identifiable objects from the video frames shown above, using transcript to identify names.

## FRAME CONTEXT
Chapter Index: {chapter_index}
Chapter Start Time: {start_time} seconds
Chapter End Time: {end_time} seconds
Video ID: {video_id}
Number of frames: {frame_count}

## FRAME TIMELINE
{frame_timeline}

## TRANSCRIPT (use for name identification)
{transcript}

## TASK
Analyze the frames above and extract ALL visible objects. Use the transcript to identify actual names of people/objects when mentioned.

Return JSON in this format:

```json
{{
    "objects": [
        {{
            "name": "Jitendra Kumar Kushwaha",
            "object_type": "person",
            "appearance": [
                "blue shirt",
                "standing in agricultural field",
                "middle-aged man"
            ],
            "identity": [
                "farmer",
                "interview subject",
                "land owner"
            ],
            "first_seen": 0.0,
            "last_seen": 45.5
        }},
        {{
            "name": "iPhone 15 Pro",
            "object_type": "item",
            "appearance": [
                "silver titanium finish",
                "6.1 inch display"
            ],
            "identity": [
                "smartphone",
                "Apple product"
            ],
            "first_seen": 12.0,
            "last_seen": 15.5
        }}
    ]
}}
```

## OBJECT TYPES
- "person": Human individuals - USE ACTUAL NAME from transcript if mentioned
- "item": Physical objects, tools, products
- "animal": Animals, pets, wildlife
- "text": On-screen text, signs, labels (extract the actual text)
- "vehicle": Cars, bikes, transportation
- "location": Identifiable places, rooms, settings
- "food": Food items, ingredients, dishes

## GUIDELINES
1. Extract up to {max_objects} objects maximum
2. **NAME RESOLUTION (IMPORTANT)**:
   - If a person is NAMED in the transcript, use their actual name (e.g., "Jitendra Kumar Kushwaha")
   - If a product/brand is mentioned, use that name (e.g., "John Deere tractor")
   - Only use generic descriptors ("man in blue shirt") if NO name is provided in transcript
3. Appearance: 3-5 visual characteristics you can observe
4. Identity: Role, occupation, category from transcript context (e.g., "farmer", "CEO", "interviewer")
5. first_seen: Timestamp (relative to chapter start) when object FIRST appears
6. last_seen: Timestamp (relative to chapter start) when object LAST appears
7. Focus on objects that are:
   - Visually prominent in the frames
   - Actively involved in visible actions
   - Named or referenced in the transcript
8. Only extract objects you can actually SEE in the frames

Return ONLY valid JSON with the objects array."""


# =============================================================================
# MULTIMODAL MESSAGE BUILDERS
# =============================================================================

def build_event_messages(
    chapter_data: Dict[str, Any],
    keyframes: Dict[str, Any],
    max_frames: int = 12,
    max_events: int = 10,
    min_duration_ms: int = 500,
) -> List[Dict[str, Any]]:
    """Build multimodal messages for event extraction (frames + transcript + summary).
    
    Args:
        chapter_data: Chapter data with start, end, transcript, summary
        keyframes: Keyframe data with list of frames
        max_frames: Maximum frames to include
        max_events: Maximum events to extract
        min_duration_ms: Minimum event duration in milliseconds
        
    Returns:
        List of messages for chat_completion API
    """
    start_time = chapter_data.get("start", 0.0)
    end_time = chapter_data.get("end", 0.0)
    transcript = chapter_data.get("transcript", "")
    chapter_index = chapter_data.get("chunk_index", 0)
    duration = end_time - start_time
    
    # Build user content: images first (stacked), then text
    user_content: List[Dict[str, Any]] = []
    
    # Add frames in chronological order
    image_content, encoded_count, sorted_frames = build_image_content(keyframes, max_frames)
    user_content.extend(image_content)
    
    # Build frame timeline
    frame_timeline = format_frame_timeline(sorted_frames)
    
    # Build text content with ALL information (multimodal)
    text_content = f"""## FRAME SEQUENCE
The {encoded_count} frames above are shown in chronological order, capturing the visual progression of this chapter.

## FRAME TIMELINE
{frame_timeline}

## CHAPTER CONTEXT
Chapter Index: {chapter_index}
Start Time: {start_time} seconds
End Time: {end_time} seconds
Duration: {duration:.1f} seconds

## TRANSCRIPT
{transcript or "No transcript available."}

## TASK
Analyze BOTH the visual content (frame sequence) AND the audio content (transcript) to extract atomic events.

Extract events in the following JSON format:

```json
{{
    "events": [
        {{
            "description": "Clear, specific description of what happens",
            "timestamp": 0.0,
            "duration": 2.5,
            "event_type": "action",
            "participants": ["person in blue shirt", "wooden cutting board"],
            "sequence_number": 1
        }}
    ]
}}
```

## EVENT TYPES
- "action": Physical movement, manipulation, or activity
- "dialogue": Speech, narration, or verbal communication
- "transition": Scene change, visual transition, or context shift
- "state_change": Object or environment state modification (e.g., light turns on, door opens)

## GUIDELINES
1. Extract up to {max_events} events maximum
2. Timestamps are relative to chapter start (0.0 = chapter start)
3. Duration should be estimated based on context (minimum {min_duration_ms}ms)
4. Participants should be descriptive: "chef in white apron" not just "person"
5. Sequence numbers start at 1 and increment for each event
6. If dialogue occurs during an action, create separate events
7. Combine visual observations from frames with transcript context
8. Focus on events that are visually or narratively significant

Return ONLY valid JSON with the events array."""

    user_content.append({"type": "text", "text": text_content})
    
    return [
        {"role": "system", "content": EVENT_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_object_messages(
    chapter_data: Dict[str, Any],
    keyframes: Dict[str, Any],
    max_frames: int = 12,
    max_objects: int = 15,
) -> List[Dict[str, Any]]:
    """Build multimodal messages for object extraction (frames + transcript for name resolution).
    
    Args:
        chapter_data: Chapter data with start, end, video_id, transcript
        keyframes: Keyframe data with list of frames
        max_frames: Maximum frames to include
        max_objects: Maximum objects to extract
        
    Returns:
        List of messages for chat_completion API
    """
    start_time = chapter_data.get("start", 0.0)
    end_time = chapter_data.get("end", 0.0)
    video_id = chapter_data.get("video_id", "unknown")
    chapter_index = chapter_data.get("chunk_index", 0)
    transcript = chapter_data.get("transcript", "")
    
    # Build user content: images first (stacked), then text
    user_content: List[Dict[str, Any]] = []
    
    # Add frames in chronological order
    image_content, encoded_count, sorted_frames = build_image_content(keyframes, max_frames)
    user_content.extend(image_content)
    
    # Build frame timeline
    frame_timeline = format_frame_timeline(sorted_frames)
    
    # Format the multimodal prompt with transcript
    text_content = OBJECT_EXTRACTION_USER_PROMPT.format(
        chapter_index=chapter_index,
        start_time=start_time,
        end_time=end_time,
        video_id=video_id,
        frame_count=encoded_count,
        frame_timeline=frame_timeline,
        transcript=transcript or "No transcript available.",
        max_objects=max_objects,
    )
    
    user_content.append({"type": "text", "text": text_content})
    
    return [
        {"role": "system", "content": OBJECT_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
