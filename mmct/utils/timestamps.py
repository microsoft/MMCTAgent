"""Timestamp utilities for timestamped description format."""

import re


def strip_timestamps(text: str) -> str:
    """Remove [Xs] timestamp markers from timestamped description text.

    Example:
        >>> strip_timestamps("[126s] Probability of heads is 1/2.\\n[130s] Next toss...")
        'Probability of heads is 1/2. Next toss...'
    """
    return re.sub(r"\[\d+s\]\s*", "", text).strip()
