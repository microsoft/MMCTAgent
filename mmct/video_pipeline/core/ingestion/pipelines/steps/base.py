"""Base classes for pipeline steps.

This module defines the foundational interfaces and data structures used 
throughout the ingestion pipeline, including the execution context, step 
results, and the abstract base step class.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class StepContext:
    """Shared context passed to all pipeline steps.

    Contains user-provided parameters, provider instances, runtime state, 
    and inter-step communication via the data store.

    Attributes:
        video_path (str): Local path to the source video file.
        provider (Any): IngestionProviderConfig containing service clients.
        language (Optional[Any]): Primary language as a Languages enum.
        url (str, optional): Remote URL associated with the video.
        transcript_path (str, optional): Local path to a pre-existing transcript.
        save_local_report (bool): If True, persists execution details to disk.
        verbosity (int): Logging detail level (0, 1, or 2).
        output_dir (str): Working directory for generated files (frames, etc.).
        video_id (str): Unique hash or identifier for the video.
        video_duration (float): Length of the video in seconds.
        data_store (Any): Centralized repository for inter-step outputs.
        logger (Any): Logger instance for recording execution logs.
        user_params (Dict[str, Any]): Global runtime overrides for all steps.
    """

    # User-provided parameters
    video_path: str
    provider: Any
    language: Optional[Any] = None
    url: Optional[str] = None
    transcript_path: Optional[str] = None
    save_local_report: bool = False
    verbosity: int = 0

    # Runtime state
    output_dir: str = ""
    video_id: str = ""
    video_duration: float = 0.0

    # Step communication
    data_store: Optional[Any] = None

    # Utilities
    logger: Optional[Any] = None

    # User config overrides (keyframe_config, frame_stacking_grid_size, etc.)
    user_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result object returned by a pipeline step after execution.

    Encapsulates all outputs meant for downstream steps, execution metrics, 
    and paths to generated file artifacts.

    Attributes:
        step_id (str): The identifier of the step that produced this result.
        outputs (Dict[str, Any]): Data to be stored in the centralized data store.
        metrics (Dict[str, float]): Quantitative measurements (e.g., latency, counts).
        artifacts (List[str]): Paths to files created during step execution.
    """

    step_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensures that mutable default collections are properly initialized."""
        if self.outputs is None:
            self.outputs = {}
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = []


class PipelineStep(ABC):
    """Abstract base class for all video ingestion steps.

    Subclasses must implement the asynchronous `run` method to define their 
    specific processing logic and should register themselves in the step 
    registry.

    Attributes:
        step_type (str): Unique identifier for this category of step (e.g., 'ingestion.ocr').
        description (str): Human-readable summary of the step's purpose.
        step_id (str): The unique instance identifier within a specific pipeline.
        params (Dict[str, Any]): Static configuration parameters from YAML.
    """

    step_type: str = "base"
    description: str = ""

    def __init__(self, step_id: str, params: Optional[Dict[str, Any]] = None):
        """Initializes the PipelineStep.

        Args:
            step_id: Unique string identifying this step instance.
            params: Dictionary of configuration parameters.
        """
        self.step_id = step_id
        self.params = params or {}

    @abstractmethod
    async def run(self, context: StepContext) -> StepResult:
        """Executes the core processing logic for the step.

        Args:
            context: The shared StepContext containing state and providers.

        Returns:
            StepResult: The result containing outputs, metrics, and artifacts.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(f"Step {self.step_type} must implement run()")

    def get_param(self, key: str, context: StepContext, default: Any = None) -> Any:
        """Retrieves a configuration parameter with hierarchical priority.

        The priority order is:
        1. `context.user_params` (highest - runtime overrides)
        2. `self.params` (middle - static YAML config)
        3. `default` (lowest)

        Args:
            key: The name of the parameter to fetch.
            context: The StepContext to check for overrides.
            default: The fallback value if the key is not found elsewhere.

        Returns:
            Any: The first value found in the priority chain.
        """
        # 1. User runtime overrides (highest priority)
        if key in context.user_params:
            return context.user_params[key]

        # 2. YAML step configuration
        if key in self.params:
            return self.params[key]

        # 3. Default value
        return default
