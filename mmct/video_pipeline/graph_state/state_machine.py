"""State machine — query pipeline states and context.

Defines the states, context dataclass, and constants
for the deterministic state machine pipeline.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class QueryState(Enum):
    """States in the state machine query pipeline."""

    PARSE_INPUT = auto()
    PLAN = auto()
    VALIDATE_PLAN = auto()
    DISCOVER_VIDEOS = auto()
    RETRIEVE = auto()
    CHECK_EVIDENCE = auto()
    EXPAND_CONTEXT = auto()
    REPHRASE = auto()
    ANALYZE_IMAGES = auto()
    SYNTHESIZE = auto()
    CRITIQUE = auto()
    REVISE = auto()
    SUBMIT = auto()
    ERROR = auto()


# Hard limits enforced by code
MAX_RETRIEVE_ATTEMPTS = 2
MAX_REVISE_ATTEMPTS = 1
MAX_SUB_QUERIES = 4
MIN_SUB_QUERIES = 1

VALID_TARGETS = {"ChapterGroup", "Chapter", "Transcript", "Event", "Object"}
VALID_STRATEGIES = {"SEARCH", "OVERVIEW"}


@dataclass
class QueryContext:
    """Accumulated state threaded through the pipeline.

    Each state handler reads and mutates this context.
    """

    # Input
    query: str
    video_id: Optional[str] = None
    video_ids: Optional[List[str]] = None
    use_critic: bool = True
    request_id: str = ""
    video_catalog: Optional[str] = None

    # Determined by PARSE_INPUT (code, not LLM)
    video_scope: str = ""  # "single" | "multi" | "cross"
    effective_video_ids: Optional[List[str]] = None

    # Produced by PLAN (LLM)
    plan: Optional[Dict[str, Any]] = None

    # Produced by RETRIEVE / DISCOVER
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    image_analyses: List[Dict[str, Any]] = field(default_factory=list)

    # Produced by SYNTHESIZE (LLM)
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)

    # Produced by CRITIQUE (LLM)
    critic_feedback: Optional[Dict[str, Any]] = None

    # Counters
    retrieve_attempts: int = 0
    revise_attempts: int = 0

    # Telemetry
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    state_log: List[str] = field(default_factory=list)

    def log_state(self, state: QueryState, detail: str = "") -> None:
        entry = state.name
        if detail:
            entry += f": {detail}"
        self.state_log.append(entry)

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.token_usage["prompt_tokens"] += prompt
        self.token_usage["completion_tokens"] += completion


class StateTransitionError(Exception):
    """Raised when a state transition is invalid."""

    def __init__(self, current: QueryState, message: str):
        self.current_state = current
        super().__init__(f"[{current.name}] {message}")