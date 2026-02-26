"""Configuration constants for chapter grouping.

Defines thresholds and limits for chapter grouping and summarization.
"""

from typing import Dict, Any


# =============================================================================
# GROUPING THRESHOLDS
# =============================================================================

# Minimum cosine similarity threshold for chapters to be grouped together
CHAPTER_GROUPING_THRESHOLD: float = 0.7

# Maximum chapter distance (index difference) for chapters to be in the same group
CHAPTER_TEMPORAL_WINDOW: int = 5


# =============================================================================
# SUMMARIZATION SETTINGS
# =============================================================================

# Maximum tokens for group summary generation
GROUP_SUMMARY_MAX_TOKENS: int = 1024

# Temperature for LLM summary generation (lower = more deterministic)
GROUP_SUMMARY_TEMPERATURE: float = 0.3


# =============================================================================
# BATCH PROCESSING
# =============================================================================

# Maximum number of chapters to process in a single batch
MAX_CHAPTERS_PER_BATCH: int = 50

# Concurrent LLM requests limit for summarization
MAX_CONCURRENT_SUMMARIES: int = 4


# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================

# Embedding dimension (should match the model used)
EMBEDDING_DIMENSION: int = 1536


def get_grouping_config() -> Dict[str, Any]:
    """Get complete grouping configuration as dictionary.
    
    Returns:
        Dictionary containing all grouping configuration values.
    """
    return {
        "thresholds": {
            "grouping_threshold": CHAPTER_GROUPING_THRESHOLD,
            "temporal_window": CHAPTER_TEMPORAL_WINDOW,
        },
        "summarization": {
            "max_tokens": GROUP_SUMMARY_MAX_TOKENS,
            "temperature": GROUP_SUMMARY_TEMPERATURE,
        },
        "batch": {
            "max_chapters_per_batch": MAX_CHAPTERS_PER_BATCH,
            "max_concurrent_summaries": MAX_CONCURRENT_SUMMARIES,
        },
        "embedding": {
            "dimension": EMBEDDING_DIMENSION,
        },
    }
