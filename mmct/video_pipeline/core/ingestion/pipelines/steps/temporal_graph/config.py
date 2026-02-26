"""Configuration constants for temporal graph extraction.

Defines thresholds, limits, and model configurations for event and object extraction.
"""

from typing import Dict, Any


# =============================================================================
# SIMILARITY THRESHOLDS
# =============================================================================

# Minimum cosine similarity for matching objects across chapters
OBJECT_SIMILARITY_THRESHOLD: float = 0.80

# Minimum cosine similarity for matching events
EVENT_SIMILARITY_THRESHOLD: float = 0.75


# =============================================================================
# EXTRACTION LIMITS
# =============================================================================

# Maximum events to extract per chapter
MAX_EVENTS_PER_CHAPTER: int = 10

# Maximum objects to associate with a single event
MAX_OBJECTS_PER_EVENT: int = 5

# Minimum event duration in milliseconds to be considered valid
MIN_EVENT_DURATION_MS: int = 500

# Maximum retry attempts for LLM extraction calls
MAX_EXTRACTION_RETRIES: int = 3


# =============================================================================
# EMBEDDING MODEL CONFIGURATIONS
# =============================================================================

# Event embedding model - good for semantic similarity of event descriptions
EVENT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# Object embedding model - optimized for short entity descriptions
OBJECT_EMBEDDING_MODEL: str = "Snowflake/snowflake-arctic-embed-s"


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Temperature for event extraction (lower = more deterministic)
EVENT_EXTRACTION_TEMPERATURE: float = 0.3

# Temperature for object extraction
OBJECT_EXTRACTION_TEMPERATURE: float = 0.2

# Maximum tokens for event extraction response
EVENT_MAX_TOKENS: int = 2048

# Maximum tokens for object extraction response
OBJECT_MAX_TOKENS: int = 2048


# =============================================================================
# BATCH PROCESSING
# =============================================================================

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE: int = 32

# Concurrent LLM requests limit
MAX_CONCURRENT_EXTRACTIONS: int = 4


# =============================================================================
# DEDUPLICATION SETTINGS
# =============================================================================

# Minimum appearance count for object to be considered significant
MIN_OBJECT_APPEARANCES: int = 1

# Time window (seconds) for merging similar consecutive events
EVENT_MERGE_WINDOW_SECONDS: float = 2.0


def get_extraction_config() -> Dict[str, Any]:
    """Get complete extraction configuration as dictionary.
    
    Returns:
        Dictionary containing all extraction configuration values.
    """
    return {
        "similarity": {
            "object_threshold": OBJECT_SIMILARITY_THRESHOLD,
            "event_threshold": EVENT_SIMILARITY_THRESHOLD,
        },
        "limits": {
            "max_events_per_chapter": MAX_EVENTS_PER_CHAPTER,
            "max_objects_per_event": MAX_OBJECTS_PER_EVENT,
            "min_event_duration_ms": MIN_EVENT_DURATION_MS,
            "max_retries": MAX_EXTRACTION_RETRIES,
        },
        "models": {
            "event_embedding": EVENT_EMBEDDING_MODEL,
            "object_embedding": OBJECT_EMBEDDING_MODEL,
        },
        "llm": {
            "event_temperature": EVENT_EXTRACTION_TEMPERATURE,
            "object_temperature": OBJECT_EXTRACTION_TEMPERATURE,
            "event_max_tokens": EVENT_MAX_TOKENS,
            "object_max_tokens": OBJECT_MAX_TOKENS,
        },
        "batch": {
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "max_concurrent": MAX_CONCURRENT_EXTRACTIONS,
        },
        "dedup": {
            "min_object_appearances": MIN_OBJECT_APPEARANCES,
            "event_merge_window": EVENT_MERGE_WINDOW_SECONDS,
        },
    }
