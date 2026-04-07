"""Causal relationship detection and linking for event graphs.

Creates CAUSES and RESULT_OF edges between causally related events using a
hybrid heuristic + LLM validation approach.

Components:
- HeuristicCausalDetector: Scores event pairs using weighted combination of:
  * Temporal proximity (exponential decay)
  * Shared participants (Jaccard similarity)
  * Event type patterns (lookup table for common cause-effect pairs)

- LLMCausalValidator: Validates uncertain candidates (score 0.5-0.8) using LLM

- CausalLinker: Orchestrates the process:
  * High confidence (≥0.8): Accept automatically
  * Uncertain (0.5-0.8): Validate with LLM (or heuristic fallback)
  * Low (<0.5): Reject
  * Creates bidirectional edges: A --CAUSES--> B and B --RESULT_OF--> A

- CausalDetectionConfig: Configuration container with factory methods
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from mmct.video_pipeline.core.ingestion.models import GraphEvent
from mmct.providers.base import BaseLLMProvider
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider


# =============================================================================
# Pydantic models for structured LLM output
# =============================================================================

class CausalValidation(BaseModel):
    """Single causal relationship validation result."""
    model_config = ConfigDict(extra='forbid')
    
    pair_index: int = Field(..., description="0-based index of the event pair")
    is_causal: bool = Field(..., description="Whether the relationship is causal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Brief explanation for the decision")


class CausalValidationResponse(BaseModel):
    """Response containing all causal validations."""
    model_config = ConfigDict(extra='forbid')
    
    validations: List[CausalValidation] = Field(
        ..., 
        description="List of validation results for each event pair"
    )


@dataclass
class CausalLinkResult:
    """Result of causal link detection and creation.
    
    Attributes:
        causes_edges_created: Number of CAUSES edges created.
        result_of_edges_created: Number of RESULT_OF edges created.
        pairs_validated_by_llm: Number of pairs validated by LLM.
        pairs_rejected: Number of candidate pairs rejected.
        high_confidence_pairs: Number of high confidence pairs found.
        errors: List of error messages encountered during processing.
    """
    causes_edges_created: int = 0
    result_of_edges_created: int = 0
    pairs_validated_by_llm: int = 0
    pairs_rejected: int = 0
    high_confidence_pairs: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class CausalCandidate:
    """A candidate causal relationship between two events.
    
    Attributes:
        source_event: The event that may cause the target event.
        target_event: The event that may be caused by the source event.
        score: Overall causal score between 0 and 1.
        component_scores: Individual component scores (temporal, participant, event_type).
        reason: Human-readable explanation for the causal relationship.
    """
    source_event: GraphEvent
    target_event: GraphEvent
    score: float
    component_scores: Dict[str, float]
    reason: str


class HeuristicCausalDetector:
    """Heuristic-based detector for causal relationships between events.
    
    Uses weighted combination of temporal proximity, shared participants,
    and event type patterns to score potential causal relationships.
    
    Attributes:
        CAUSAL_PATTERNS: Dictionary mapping (source_type, target_type) pairs
            to base causal scores.
    """
    
    CAUSAL_PATTERNS: Dict[Tuple[str, str], float] = {
        ("action", "state_change"): 0.9,
        ("dialogue", "action"): 0.7,
        ("action", "action"): 0.5,
        ("transition", "action"): 0.6,
        ("action", "dialogue"): 0.5,
        ("state_change", "action"): 0.6,
        ("state_change", "state_change"): 0.4,
        ("dialogue", "dialogue"): 0.3,
        ("transition", "state_change"): 0.5,
    }
    
    def __init__(
        self,
        temporal_weight: float = 0.2,
        participant_weight: float = 0.3,
        event_type_weight: float = 0.5,
        max_temporal_gap_seconds: float = 30.0
    ) -> None:
        """Initialize the HeuristicCausalDetector.
        
        Args:
            temporal_weight: Weight for temporal proximity score (0-1).
            participant_weight: Weight for shared participant score (0-1).
            event_type_weight: Weight for event type pattern score (0-1).
            max_temporal_gap_seconds: Maximum time gap in seconds to consider
                events as potentially causally related.
                
        Raises:
            ValueError: If weights don't sum to 1.0 or max_gap is not positive.
        """
        weight_sum = temporal_weight + participant_weight + event_type_weight
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.6f} "
                f"(temporal={temporal_weight}, participant={participant_weight}, "
                f"event_type={event_type_weight})"
            )
        
        if max_temporal_gap_seconds <= 0:
            raise ValueError(
                f"max_temporal_gap_seconds must be positive, got {max_temporal_gap_seconds}"
            )
        
        self.temporal_weight = temporal_weight
        self.participant_weight = participant_weight
        self.event_type_weight = event_type_weight
        self.max_temporal_gap_seconds = max_temporal_gap_seconds
        
        logger.debug(
            "Initialized HeuristicCausalDetector with weights: "
            f"temporal={temporal_weight}, participant={participant_weight}, "
            f"event_type={event_type_weight}, max_gap={max_temporal_gap_seconds}s"
        )
    
    def score(
        self,
        event_a: GraphEvent,
        event_b: GraphEvent
    ) -> Tuple[float, Dict[str, float], str]:
        """Score the causal relationship between two events.
        
        Computes a weighted score based on temporal proximity, shared
        participants, and event type patterns.
        
        Args:
            event_a: The potential cause event (must occur before event_b).
            event_b: The potential effect event.
            
        Returns:
            Tuple containing:
                - float: Overall causal score between 0 and 1.
                - Dict[str, float]: Component scores (temporal, participant, event_type).
                - str: Human-readable reason for the score.
        """
        temporal_score = self._temporal_score(event_a, event_b)
        participant_score = self._participant_score(event_a, event_b)
        event_type_score = self._event_type_score(event_a, event_b)
        
        component_scores = {
            "temporal": temporal_score,
            "participant": participant_score,
            "event_type": event_type_score
        }
        
        overall_score = (
            self.temporal_weight * temporal_score +
            self.participant_weight * participant_score +
            self.event_type_weight * event_type_score
        )
        
        reason = self._generate_reason(event_a, event_b, component_scores, overall_score)
        
        return overall_score, component_scores, reason
    
    def find_causal_candidates(
        self,
        events: List[GraphEvent],
        min_score: float = 0.5,
        max_candidates_per_event: int = 3,
        skip_adjacent: bool = True,
    ) -> List[CausalCandidate]:
        """Find candidate causal relationships among a list of events.
        
        Evaluates all pairs of events where the source occurs before the target
        and returns candidates exceeding the minimum score threshold.
        
        Args:
            events: List of events to analyze for causal relationships.
            min_score: Minimum score threshold for candidates (0-1).
            max_candidates_per_event: Maximum number of effect candidates
                to return per source event.
            skip_adjacent: If True, skip temporally adjacent events (NEXT already exists).
                
        Returns:
            List of CausalCandidate objects sorted by score descending.
        """
        if not events:
            logger.debug("No events provided for causal candidate detection")
            return []
        
        # Sort events by timestamp, then sequence number
        sorted_events = sorted(
            [e for e in events if e.timestamp is not None],
            key=lambda e: (e.timestamp, e.sequence_number or 0)  # type: ignore
        )
        
        if len(sorted_events) < 2:
            logger.debug("Fewer than 2 events with timestamps, no candidates possible")
            return []
        
        # Build set of adjacent pairs to skip (i, i+1 in sorted order)
        adjacent_indices: Set[Tuple[int, int]] = set()
        if skip_adjacent:
            for i in range(len(sorted_events) - 1):
                adjacent_indices.add((i, i + 1))
        
        # Track candidates per source event
        candidates_by_source: Dict[str, List[CausalCandidate]] = {}
        
        for i, source_event in enumerate(sorted_events):
            source_id = source_event.id or f"event_{i}"
            candidates_by_source[source_id] = []
            
            # Only consider events that come after the source
            for j, target_event in enumerate(sorted_events[i + 1:], start=i + 1):
                # Skip adjacent events - NEXT relationship already captures temporal sequence
                if skip_adjacent and (i, j) in adjacent_indices:
                    continue
                
                # Check temporal gap
                time_gap = (target_event.timestamp or 0) - (source_event.timestamp or 0)
                if time_gap > self.max_temporal_gap_seconds:
                    # Events too far apart, skip remaining (list is sorted)
                    break
                
                score, component_scores, reason = self.score(source_event, target_event)
                
                if score >= min_score:
                    candidate = CausalCandidate(
                        source_event=source_event,
                        target_event=target_event,
                        score=score,
                        component_scores=component_scores,
                        reason=reason
                    )
                    candidates_by_source[source_id].append(candidate)
        
        # Limit candidates per source and collect all
        all_candidates: List[CausalCandidate] = []
        for source_id, candidates in candidates_by_source.items():
            # Sort by score descending and take top N
            candidates.sort(key=lambda c: c.score, reverse=True)
            all_candidates.extend(candidates[:max_candidates_per_event])
        
        # Sort final list by score descending
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        
        logger.debug(
            f"Found {len(all_candidates)} causal candidates from {len(sorted_events)} events"
            + (f" (skipped {len(adjacent_indices)} adjacent pairs)" if skip_adjacent else "")
        )
        
        return all_candidates
    
    def _temporal_score(self, event_a: GraphEvent, event_b: GraphEvent) -> float:
        """Calculate temporal proximity score using exponential decay.
        
        Uses exponential decay with half_life = max_temporal_gap / 3.
        Events occurring at the same time get score 1.0, events at max_gap
        distance get approximately 0.125.
        
        Args:
            event_a: First event (should be earlier).
            event_b: Second event (should be later).
            
        Returns:
            Temporal score between 0 and 1.
        """
        timestamp_a = event_a.timestamp
        timestamp_b = event_b.timestamp
        
        if timestamp_a is None or timestamp_b is None:
            return 0.0
        
        time_gap = abs(timestamp_b - timestamp_a)
        
        if time_gap > self.max_temporal_gap_seconds:
            return 0.0
        
        # Exponential decay with half_life = max_gap / 3
        half_life = self.max_temporal_gap_seconds / 3.0
        decay_constant = math.log(2) / half_life
        
        score = math.exp(-decay_constant * time_gap)
        
        return score
    
    def _participant_score(self, event_a: GraphEvent, event_b: GraphEvent) -> float:
        """Calculate shared participant score using Jaccard similarity.
        
        Uses Jaccard similarity (intersection / union) between participant sets.
        If there is any overlap, the score is boosted to at least 0.4.
        
        Args:
            event_a: First event.
            event_b: Second event.
            
        Returns:
            Participant overlap score between 0 and 1.
        """
        participants_a = set(event_a.participants or [])
        participants_b = set(event_b.participants or [])
        
        if not participants_a and not participants_b:
            # No participants in either event, neutral score
            return 0.0
        
        if not participants_a or not participants_b:
            # One event has no participants
            return 0.0
        
        intersection = participants_a & participants_b
        union = participants_a | participants_b
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Boost to minimum 0.4 if there's any overlap
        if intersection:
            jaccard = max(jaccard, 0.4)
        
        return jaccard
    
    def _event_type_score(self, event_a: GraphEvent, event_b: GraphEvent) -> float:
        """Calculate event type pattern score.
        
        Looks up the event type pair in CAUSAL_PATTERNS. If not found,
        tries a fuzzy match with 0.8 penalty. Falls back to 0.2 default.
        
        Args:
            event_a: First event (cause).
            event_b: Second event (effect).
            
        Returns:
            Event type score between 0 and 1.
        """
        type_a = (event_a.event_type or "").lower().strip()
        type_b = (event_b.event_type or "").lower().strip()
        
        if not type_a or not type_b:
            return 0.2  # Default score for unknown types
        
        # Direct lookup
        pattern_key = (type_a, type_b)
        if pattern_key in self.CAUSAL_PATTERNS:
            return self.CAUSAL_PATTERNS[pattern_key]
        
        # Fuzzy match: check if event types contain known patterns
        for (known_a, known_b), base_score in self.CAUSAL_PATTERNS.items():
            if known_a in type_a and known_b in type_b:
                # Apply 0.8 penalty for fuzzy match
                return base_score * 0.8
        
        # Default score for unrecognized patterns
        return 0.2
    
    def _generate_reason(
        self,
        event_a: GraphEvent,
        event_b: GraphEvent,
        component_scores: Dict[str, float],
        overall_score: float
    ) -> str:
        """Generate a human-readable reason for the causal score.
        
        Args:
            event_a: Source event.
            event_b: Target event.
            component_scores: Dictionary of component scores.
            overall_score: Overall causal score.
            
        Returns:
            Human-readable explanation string.
        """
        reasons = []
        
        type_a = event_a.event_type or "unknown"
        type_b = event_b.event_type or "unknown"
        
        # Event type reasoning
        event_type_score = component_scores.get("event_type", 0)
        if event_type_score >= 0.7:
            reasons.append(f"strong {type_a}->{type_b} pattern")
        elif event_type_score >= 0.4:
            reasons.append(f"moderate {type_a}->{type_b} pattern")
        else:
            reasons.append(f"weak {type_a}->{type_b} pattern")
        
        # Temporal reasoning
        temporal_score = component_scores.get("temporal", 0)
        if temporal_score >= 0.8:
            reasons.append("very close in time")
        elif temporal_score >= 0.5:
            reasons.append("close in time")
        elif temporal_score >= 0.2:
            reasons.append("moderate time gap")
        else:
            reasons.append("distant in time")
        
        # Participant reasoning
        participant_score = component_scores.get("participant", 0)
        shared_participants = self._get_shared_participants(event_a, event_b)
        if shared_participants:
            if participant_score >= 0.7:
                reasons.append(f"many shared participants: {', '.join(list(shared_participants)[:3])}")
            else:
                reasons.append(f"shared participants: {', '.join(list(shared_participants)[:3])}")
        else:
            reasons.append("no shared participants")
        
        # Build final reason
        score_level = "high" if overall_score >= 0.7 else "moderate" if overall_score >= 0.5 else "low"
        reason = f"{score_level} confidence ({overall_score:.2f}): {'; '.join(reasons)}"
        
        return reason
    
    def _get_shared_participants(
        self,
        event_a: GraphEvent,
        event_b: GraphEvent
    ) -> Set[str]:
        """Get the set of shared participants between two events.
        
        Args:
            event_a: First event.
            event_b: Second event.
            
        Returns:
            Set of participant names that appear in both events.
        """
        participants_a = set(event_a.participants or [])
        participants_b = set(event_b.participants or [])
        
        return participants_a & participants_b


class CausalDetectionConfig:
    """Configuration for causal relationship detection.
    
    Provides default values and factory methods for creating
    HeuristicCausalDetector instances.
    
    Attributes:
        DEFAULT_TEMPORAL_WEIGHT: Default weight for temporal score (0.2).
        DEFAULT_PARTICIPANT_WEIGHT: Default weight for participant score (0.3).
        DEFAULT_EVENT_TYPE_WEIGHT: Default weight for event type score (0.5).
        DEFAULT_MAX_TEMPORAL_GAP: Default maximum temporal gap in seconds (30.0).
        DEFAULT_MIN_SCORE: Default minimum score threshold (0.5).
        DEFAULT_MAX_CANDIDATES_PER_EVENT: Default max candidates per event (3).
    """
    
    DEFAULT_TEMPORAL_WEIGHT: float = 0.2
    DEFAULT_PARTICIPANT_WEIGHT: float = 0.3
    DEFAULT_EVENT_TYPE_WEIGHT: float = 0.5
    DEFAULT_MAX_TEMPORAL_GAP: float = 30.0
    DEFAULT_MIN_SCORE: float = 0.5
    DEFAULT_MAX_CANDIDATES_PER_EVENT: int = 3
    
    def __init__(
        self,
        temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
        participant_weight: float = DEFAULT_PARTICIPANT_WEIGHT,
        event_type_weight: float = DEFAULT_EVENT_TYPE_WEIGHT,
        max_temporal_gap_seconds: float = DEFAULT_MAX_TEMPORAL_GAP,
        min_score: float = DEFAULT_MIN_SCORE,
        max_candidates_per_event: int = DEFAULT_MAX_CANDIDATES_PER_EVENT
    ) -> None:
        """Initialize CausalDetectionConfig.
        
        Args:
            temporal_weight: Weight for temporal proximity score.
            participant_weight: Weight for shared participant score.
            event_type_weight: Weight for event type pattern score.
            max_temporal_gap_seconds: Maximum time gap to consider.
            min_score: Minimum score threshold for candidates.
            max_candidates_per_event: Maximum candidates per source event.
        """
        self.temporal_weight = temporal_weight
        self.participant_weight = participant_weight
        self.event_type_weight = event_type_weight
        self.max_temporal_gap_seconds = max_temporal_gap_seconds
        self.min_score = min_score
        self.max_candidates_per_event = max_candidates_per_event
    
    def create_detector(self) -> HeuristicCausalDetector:
        """Create a HeuristicCausalDetector with this configuration.
        
        Returns:
            Configured HeuristicCausalDetector instance.
        """
        return HeuristicCausalDetector(
            temporal_weight=self.temporal_weight,
            participant_weight=self.participant_weight,
            event_type_weight=self.event_type_weight,
            max_temporal_gap_seconds=self.max_temporal_gap_seconds
        )
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CausalDetectionConfig":
        """Create a CausalDetectionConfig from a dictionary.
        
        Args:
            config: Dictionary with configuration values.
            
        Returns:
            CausalDetectionConfig instance.
        """
        return cls(
            temporal_weight=config.get("temporal_weight", cls.DEFAULT_TEMPORAL_WEIGHT),
            participant_weight=config.get("participant_weight", cls.DEFAULT_PARTICIPANT_WEIGHT),
            event_type_weight=config.get("event_type_weight", cls.DEFAULT_EVENT_TYPE_WEIGHT),
            max_temporal_gap_seconds=config.get("max_temporal_gap_seconds", cls.DEFAULT_MAX_TEMPORAL_GAP),
            min_score=config.get("min_score", cls.DEFAULT_MIN_SCORE),
            max_candidates_per_event=config.get("max_candidates_per_event", cls.DEFAULT_MAX_CANDIDATES_PER_EVENT)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "temporal_weight": self.temporal_weight,
            "participant_weight": self.participant_weight,
            "event_type_weight": self.event_type_weight,
            "max_temporal_gap_seconds": self.max_temporal_gap_seconds,
            "min_score": self.min_score,
            "max_candidates_per_event": self.max_candidates_per_event
        }


class LLMCausalValidator:
    """LLM-based validator for causal relationship candidates.
    
    Uses an LLM with structured output (Pydantic models) to validate
    uncertain causal relationship candidates.
    """
    
    SYSTEM_PROMPT = """You are an expert at analyzing causal relationships between events.
For each event pair, determine if there is a valid cause-and-effect relationship.

Consider:
1. Does the first event logically cause or lead to the second event?
2. Is there a plausible cause-and-effect relationship?
3. Would the second event likely not occur without the first?"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        batch_size: int = 4
    ) -> None:
        """Initialize the LLMCausalValidator.
        
        Args:
            llm_provider: LLM provider for validation requests.
            batch_size: Maximum number of candidates to validate per LLM call.
        """
        self.llm_provider = llm_provider
        self.batch_size = batch_size
    
    async def validate_batch(
        self,
        candidates: List[CausalCandidate]
    ) -> List[Tuple[str, str, bool, float, str]]:
        """Validate a batch of causal candidates using LLM.
        
        Args:
            candidates: List of CausalCandidate objects to validate.
            
        Returns:
            List of tuples: (source_id, target_id, is_valid, confidence, reason)
        """
        if not candidates:
            return []
        
        results: List[Tuple[str, str, bool, float, str]] = []
        
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_results = await self._validate_single_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _validate_single_batch(
        self,
        batch: List[CausalCandidate]
    ) -> List[Tuple[str, str, bool, float, str]]:
        """Validate a single batch of candidates with LLM."""
        # Build event pairs description
        event_pairs_text = []
        for idx, candidate in enumerate(batch):
            source = candidate.source_event
            target = candidate.target_event
            
            source_desc = f"Event A: {source.description or source.event_type or 'Unknown'}"
            if source.participants:
                source_desc += f" (participants: {', '.join(source.participants[:3])})"
            
            target_desc = f"Event B: {target.description or target.event_type or 'Unknown'}"
            if target.participants:
                target_desc += f" (participants: {', '.join(target.participants[:3])})"
            
            event_pairs_text.append(
                f"Pair {idx}:\n  {source_desc}\n  {target_desc}\n  "
                f"Heuristic score: {candidate.score:.2f}"
            )
        
        user_content = f"Analyze these {len(batch)} event pairs for causal relationships:\n\n" + "\n\n".join(event_pairs_text)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.llm_provider.chat_completion(
                messages=messages,
                response_format=CausalValidationResponse,
                temperature=0.1,
            )
            return self._parse_response(response, batch)
        except Exception as e:
            logger.exception(f"LLM validation failed, falling back to heuristic")
            return self._fallback_to_heuristic(batch)
    
    def _parse_response(
        self,
        response: Dict[str, Any],
        batch: List[CausalCandidate]
    ) -> List[Tuple[str, str, bool, float, str]]:
        """Parse LLM response into validation results."""
        results: List[Tuple[str, str, bool, float, str]] = []
        
        try:
            content = response.get("content")
            
            # Handle Pydantic model response (parsed by provider)
            if isinstance(content, CausalValidationResponse):
                validations = content.validations
            elif isinstance(content, dict) and "validations" in content:
                validations = [
                    CausalValidation(**v) if isinstance(v, dict) else v 
                    for v in content["validations"]
                ]
            else:
                logger.warning(f"Unexpected response format: {type(content)}")
                return self._fallback_to_heuristic(batch)
            
            # Map validations by pair_index
            validation_map = {v.pair_index: v for v in validations}
            
            # Build results
            for idx, candidate in enumerate(batch):
                source_id = candidate.source_event.id or f"source_{idx}"
                target_id = candidate.target_event.id or f"target_{idx}"
                
                if idx in validation_map:
                    val = validation_map[idx]
                    results.append((
                        source_id, target_id, 
                        val.is_causal, val.confidence, val.reason
                    ))
                else:
                    # Missing validation, use heuristic
                    is_valid, confidence, reason = self._heuristic_decision(candidate)
                    results.append((source_id, target_id, is_valid, confidence, reason))
            
            return results
            
        except Exception as e:
            logger.exception(f"Failed to parse LLM response")
            return self._fallback_to_heuristic(batch)
    
    def _fallback_to_heuristic(
        self,
        batch: List[CausalCandidate]
    ) -> List[Tuple[str, str, bool, float, str]]:
        """Use heuristic threshold as fallback when LLM fails."""
        results: List[Tuple[str, str, bool, float, str]] = []
        
        for idx, candidate in enumerate(batch):
            source_id = candidate.source_event.id or f"source_{idx}"
            target_id = candidate.target_event.id or f"target_{idx}"
            
            is_valid, confidence, reason = self._heuristic_decision(candidate)
            results.append((source_id, target_id, is_valid, confidence, reason))
        
        return results
    
    def _heuristic_decision(
        self,
        candidate: CausalCandidate
    ) -> Tuple[bool, float, str]:
        """Make heuristic decision for a single candidate."""
        fallback_threshold = 0.65
        is_valid = candidate.score >= fallback_threshold
        confidence = candidate.score
        reason = f"Heuristic (threshold {fallback_threshold}): {candidate.reason}"
        
        return is_valid, confidence, reason


class CausalLinker:
    """Creates causal relationship edges between events in a graph.
    
    Uses heuristic scoring with optional LLM validation to identify
    and create bidirectional causal edges (CAUSES and RESULT_OF).
    
    Attributes:
        EDGE_CAUSES: Edge type for "causes" relationships.
        EDGE_RESULT_OF: Edge type for "result of" relationships.
    """
    
    EDGE_CAUSES = "CAUSES"
    EDGE_RESULT_OF = "RESULT_OF"
    
    def __init__(
        self,
        graph_provider: BaseGraphDBProvider,
        llm_provider: Optional[BaseLLMProvider] = None,
        max_causal_links_per_event: int = 1,  # Reduced from 3 to keep only strongest
        heuristic_high_threshold: float = 0.8,
        heuristic_low_threshold: float = 0.65,  # Raised from 0.5 for meaningful causality
        batch_size: int = 50,
        llm_batch_size: int = 4,
        temporal_weight: float = 0.2,
        participant_weight: float = 0.3,
        event_type_weight: float = 0.5,
        max_temporal_gap_seconds: float = 30.0,
        skip_adjacent_events: bool = True,  # Skip CAUSES for consecutive (NEXT) events
        apply_transitive_reduction: bool = True,  # Remove transitive edges (A→C if A→B→C)
    ) -> None:
        """Initialize the CausalLinker.
        
        Args:
            graph_provider: Graph database provider for creating edges.
            llm_provider: Optional LLM provider for uncertain candidate validation.
            max_causal_links_per_event: Maximum causal edges per event.
            heuristic_high_threshold: Score threshold for automatic acceptance (>=).
            heuristic_low_threshold: Score threshold for rejection (<).
            batch_size: Batch size for edge creation operations.
            llm_batch_size: Batch size for LLM validation calls.
            temporal_weight: Weight for temporal proximity in scoring.
            participant_weight: Weight for shared participants in scoring.
            event_type_weight: Weight for event type patterns in scoring.
            max_temporal_gap_seconds: Maximum time gap between causally related events.
            skip_adjacent_events: If True, don't create CAUSES for temporally adjacent events.
            apply_transitive_reduction: If True, remove transitive edges after detection.
        """
        self.graph_provider = graph_provider
        self.llm_provider = llm_provider
        self.max_causal_links_per_event = max_causal_links_per_event
        self.heuristic_high_threshold = heuristic_high_threshold
        self.heuristic_low_threshold = heuristic_low_threshold
        self.batch_size = batch_size
        self.skip_adjacent_events = skip_adjacent_events
        self.apply_transitive_reduction = apply_transitive_reduction
        
        # Create internal detector
        self.detector = HeuristicCausalDetector(
            temporal_weight=temporal_weight,
            participant_weight=participant_weight,
            event_type_weight=event_type_weight,
            max_temporal_gap_seconds=max_temporal_gap_seconds
        )
        
        # Create optional LLM validator
        self.llm_validator: Optional[LLMCausalValidator] = None
        if llm_provider is not None:
            self.llm_validator = LLMCausalValidator(
                llm_provider=llm_provider,
                batch_size=llm_batch_size
            )
        
        logger.debug(
            f"Initialized CausalLinker: high_threshold={heuristic_high_threshold}, "
            f"low_threshold={heuristic_low_threshold}, max_links={max_causal_links_per_event}, "
            f"llm_enabled={llm_provider is not None}"
        )
    
    async def link_causal_relationships(
        self,
        events: List[GraphEvent],
        video_id: str
    ) -> CausalLinkResult:
        """Detect and create causal relationship edges between events.
        
        Process:
        1. Find candidate pairs using heuristic detector
        2. Classify by score: high (>=0.8) accept, uncertain (0.5-0.8) validate, low reject
        3. Validate uncertain candidates with LLM if available
        4. Enforce max links per event limit
        5. Create bidirectional CAUSES and RESULT_OF edges
        
        Args:
            events: List of GraphEvent objects to analyze.
            video_id: Video identifier for logging and edge properties.
            
        Returns:
            CausalLinkResult with statistics about created edges.
        """
        result = CausalLinkResult()
        
        if not events:
            logger.debug("No events provided for causal linking")
            return result
        
        if len(events) < 2:
            logger.debug("Fewer than 2 events, no causal relationships possible")
            return result
        
        logger.info(
            f"Starting causal linking for video {video_id} with {len(events)} events"
        )
        
        # Find all candidates above low threshold
        # Find candidates - skip adjacent events at generation time (NEXT already exists)
        candidates = self.detector.find_causal_candidates(
            events=events,
            min_score=self.heuristic_low_threshold,
            max_candidates_per_event=self.max_causal_links_per_event * 2,  # Get extra for filtering
            skip_adjacent=self.skip_adjacent_events,
        )
        
        if not candidates:
            logger.debug("No causal candidates found")
            return result
        
        logger.debug(f"Found {len(candidates)} causal candidates")
        
        # Classify candidates by confidence level
        high_confidence: List[CausalCandidate] = []
        uncertain: List[CausalCandidate] = []
        
        for candidate in candidates:
            if candidate.score >= self.heuristic_high_threshold:
                high_confidence.append(candidate)
            elif candidate.score >= self.heuristic_low_threshold:
                uncertain.append(candidate)
            # Below low_threshold are already filtered out by find_causal_candidates
        
        result.high_confidence_pairs = len(high_confidence)
        
        logger.debug(
            f"Classification: {len(high_confidence)} high confidence, "
            f"{len(uncertain)} uncertain"
        )
        
        # Build validated pairs list: (source_id, target_id, confidence, reason)
        validated_pairs: List[Tuple[str, str, float, str]] = []
        
        # Add high confidence pairs directly
        for candidate in high_confidence:
            source_id = candidate.source_event.id or ""
            target_id = candidate.target_event.id or ""
            if source_id and target_id:
                validated_pairs.append((
                    source_id,
                    target_id,
                    candidate.score,
                    candidate.reason
                ))
        
        # Validate uncertain candidates with LLM if available
        if uncertain and self.llm_validator is not None:
            logger.debug(f"Validating {len(uncertain)} uncertain candidates with LLM")
            
            try:
                llm_results = await self.llm_validator.validate_batch(uncertain)
                
                for source_id, target_id, is_valid, confidence, reason in llm_results:
                    if is_valid and source_id and target_id:
                        validated_pairs.append((source_id, target_id, confidence, reason))
                    else:
                        result.pairs_rejected += 1
                
                result.pairs_validated_by_llm = len(llm_results)
                
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}, using heuristic fallback")
                result.errors.append(f"LLM validation error: {str(e)}")
                
                # Fallback: accept uncertain candidates above middle threshold
                middle_threshold = (
                    self.heuristic_high_threshold + self.heuristic_low_threshold
                ) / 2
                
                for candidate in uncertain:
                    source_id = candidate.source_event.id or ""
                    target_id = candidate.target_event.id or ""
                    if source_id and target_id and candidate.score >= middle_threshold:
                        validated_pairs.append((
                            source_id,
                            target_id,
                            candidate.score,
                            f"Fallback accepted: {candidate.reason}"
                        ))
                    else:
                        result.pairs_rejected += 1
        
        elif uncertain:
            # No LLM available, use heuristic fallback for uncertain candidates
            logger.debug(f"No LLM validator, using heuristic for {len(uncertain)} uncertain")
            
            middle_threshold = (
                self.heuristic_high_threshold + self.heuristic_low_threshold
            ) / 2
            
            for candidate in uncertain:
                source_id = candidate.source_event.id or ""
                target_id = candidate.target_event.id or ""
                if source_id and target_id and candidate.score >= middle_threshold:
                    validated_pairs.append((
                        source_id,
                        target_id,
                        candidate.score,
                        f"Heuristic accepted: {candidate.reason}"
                    ))
                else:
                    result.pairs_rejected += 1
        
        if not validated_pairs:
            logger.debug("No validated causal pairs to create edges for")
            return result
        
        # Apply transitive reduction: if A→B and B→C exist, remove A→C
        if self.apply_transitive_reduction and len(validated_pairs) > 1:
            validated_pairs = self._transitive_reduction(validated_pairs)
        
        # Enforce max links per event
        validated_pairs = self._enforce_max_links(validated_pairs)
        
        logger.debug(f"Creating edges for {len(validated_pairs)} validated pairs")
        
        # Create bidirectional causal edges
        try:
            causes_created, result_of_created, edge_errors = await self._create_causal_edges(
                validated_pairs=validated_pairs,
                video_id=video_id
            )
            
            result.causes_edges_created = causes_created
            result.result_of_edges_created = result_of_created
            result.errors.extend(edge_errors)
            
        except Exception as e:
            error_msg = f"Failed to create causal edges: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        logger.info(
            f"Causal linking complete for video {video_id}: "
            f"{result.causes_edges_created} CAUSES edges, "
            f"{result.result_of_edges_created} RESULT_OF edges"
        )
        
        return result
    
    def _enforce_max_links(
        self,
        validated_pairs: List[Tuple[str, str, float, str]]
    ) -> List[Tuple[str, str, float, str]]:
        """Enforce maximum causal links per event by keeping highest confidence.
        
        Args:
            validated_pairs: List of validated pair tuples.
            
        Returns:
            Filtered list respecting max_causal_links_per_event limit.
        """
        if not validated_pairs:
            return []
        
        # Sort by confidence descending
        sorted_pairs = sorted(validated_pairs, key=lambda x: x[2], reverse=True)
        
        # Track links per event (both as source and target)
        source_links: Dict[str, int] = defaultdict(int)
        target_links: Dict[str, int] = defaultdict(int)
        
        filtered_pairs: List[Tuple[str, str, float, str]] = []
        
        for source_id, target_id, confidence, reason in sorted_pairs:
            # Check if either event has reached the limit
            if (source_links[source_id] < self.max_causal_links_per_event and
                target_links[target_id] < self.max_causal_links_per_event):
                
                filtered_pairs.append((source_id, target_id, confidence, reason))
                source_links[source_id] += 1
                target_links[target_id] += 1
        
        if len(filtered_pairs) < len(validated_pairs):
            logger.debug(
                f"Reduced from {len(validated_pairs)} to {len(filtered_pairs)} pairs "
                f"(max {self.max_causal_links_per_event} per event)"
            )
        
        return filtered_pairs
    
    def _transitive_reduction(
        self,
        validated_pairs: List[Tuple[str, str, float, str]]
    ) -> List[Tuple[str, str, float, str]]:
        """Remove transitive edges: if A→B and B→C exist, remove A→C.
        
        Keeps direct causal links only, allowing graph traversal to discover
        indirect relationships. Preserves retrieval capability while reducing
        redundant edges.
        
        Args:
            validated_pairs: List of validated pair tuples.
            
        Returns:
            Filtered list with transitive edges removed.
        """
        if len(validated_pairs) <= 1:
            return validated_pairs
        
        # Build adjacency set for quick lookup
        direct_edges: Set[Tuple[str, str]] = {(src, tgt) for src, tgt, _, _ in validated_pairs}
        
        # Build adjacency list for reachability check
        adj: Dict[str, Set[str]] = defaultdict(set)
        for src, tgt, _, _ in validated_pairs:
            adj[src].add(tgt)
        
        # Find transitive edges to remove
        # Edge A→C is transitive if there exists B such that A→B and B→C
        transitive_edges: Set[Tuple[str, str]] = set()
        
        for src, tgt, _, _ in validated_pairs:
            # Check if there's an intermediate node
            for mid in adj[src]:
                if mid != tgt and tgt in adj[mid]:
                    # Found path src → mid → tgt, so src → tgt is transitive
                    transitive_edges.add((src, tgt))
                    break
        
        if not transitive_edges:
            return validated_pairs
        
        # Filter out transitive edges
        reduced_pairs = [
            (src, tgt, conf, reason) for src, tgt, conf, reason in validated_pairs
            if (src, tgt) not in transitive_edges
        ]
        
        logger.info(
            f"Transitive reduction: removed {len(transitive_edges)} edges "
            f"({len(validated_pairs)} → {len(reduced_pairs)})"
        )
        
        return reduced_pairs
    
    async def _create_causal_edges(
        self,
        validated_pairs: List[Tuple[str, str, float, str]],
        video_id: str
    ) -> Tuple[int, int, List[str]]:
        """Create bidirectional CAUSES and RESULT_OF edges.
        
        Args:
            validated_pairs: List of (source_id, target_id, confidence, reason) tuples.
            video_id: Video identifier for edge properties.
            
        Returns:
            Tuple of (causes_count, result_of_count, errors).
        """
        causes_edges: List[Dict[str, Any]] = []
        result_of_edges: List[Dict[str, Any]] = []
        
        for source_id, target_id, confidence, reason in validated_pairs:
            # CAUSES: source -> target
            causes_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "type": self.EDGE_CAUSES,
                "properties": {
                    "video_id": video_id,
                    "confidence": confidence,
                    "reason": reason,
                    "edge_type": self.EDGE_CAUSES,
                }
            })
            
            # RESULT_OF: target -> source (bidirectional)
            result_of_edges.append({
                "source_id": target_id,
                "target_id": source_id,
                "type": self.EDGE_RESULT_OF,
                "properties": {
                    "video_id": video_id,
                    "confidence": confidence,
                    "reason": reason,
                    "edge_type": self.EDGE_RESULT_OF,
                }
            })
        
        errors: List[str] = []
        causes_created = 0
        result_of_created = 0
        
        # Create CAUSES edges in batches
        for i in range(0, len(causes_edges), self.batch_size):
            batch = causes_edges[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                causes_created += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} CAUSES edges")
            except Exception as e:
                errors.append(f"CAUSES batch error: {str(e)}")
        
        # Create RESULT_OF edges in batches
        for i in range(0, len(result_of_edges), self.batch_size):
            batch = result_of_edges[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                result_of_created += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} RESULT_OF edges")
            except Exception as e:
                errors.append(f"RESULT_OF batch error: {str(e)}")
        
        return causes_created, result_of_created, errors
