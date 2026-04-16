"""Video ingestion, graph construction, and agentic querying.

This package contains the core video intelligence features of MMCT.
It supports structured ingestion of video data into a temporal knowledge
graph and provides multiple orchestration strategies (swarm-graph and
state-machine) for querying the resulting data.

Subpackages:
    - core.ingestion: Tools for scene segmentation and feature extraction.
    - graph_agent: Swarm-based orchestration strategy.
    - graph_state: Deterministic state-machine orchestration strategy.
"""

from .core.ingestion.ingestion_pipeline import IngestionPipeline
from .core.ingestion.languages import Languages
from .core.ingestion.pipelines import (
    PipelineStep,
    StepContext,
    StepResult,
    register_step,
)

__all__ = [
    "IngestionPipeline",
    "Languages",
    "PipelineStep",
    "StepContext",
    "StepResult",
    "register_step",
]
