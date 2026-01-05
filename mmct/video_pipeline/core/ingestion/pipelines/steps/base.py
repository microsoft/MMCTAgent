"""Base classes for pipeline steps."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class StepContext:
    """
    Shared context passed to all pipeline steps.

    Contains user-provided parameters, providers, runtime state, and inter-step communication.
    """

    # User-provided parameters
    video_path: str
    provider: Any  # IngestionProviderConfig
    language: Optional[Any] = None  # Languages enum
    url: Optional[str] = None
    transcript_path: Optional[str] = None
    save_local_report: bool = False
    verbosity: int = 0

    # Runtime state
    output_dir: str = ""
    video_id: str = ""
    video_duration: float = 0.0

    # Step communication
    data_store: Optional[Any] = None  # StepDataStore

    # Utilities
    logger: Optional[Any] = None

    # User config overrides (keyframe_config, frame_stacking_grid_size, etc.)
    user_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """
    Result returned by a pipeline step.

    Contains outputs for downstream steps, metrics, and artifacts.
    """

    step_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure mutable defaults are properly initialized."""
        if self.outputs is None:
            self.outputs = {}
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = []


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Subclasses must implement the run() method and register themselves
    using the @register_step decorator.

    Attributes:
        step_type: Unique identifier for this step type (e.g., "ingestion.compress")
        description: Human-readable description of what this step does
    """

    step_type: str = "base"
    description: str = ""

    def __init__(self, step_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the step.

        Args:
            step_id: Unique identifier for this step instance in the pipeline
            params: Configuration parameters from YAML
        """
        self.step_id = step_id
        self.params = params or {}

    @abstractmethod
    async def run(self, context: StepContext) -> StepResult:
        """
        Execute the step logic.

        Args:
            context: Shared context containing providers, state, and data store

        Returns:
            StepResult containing outputs, metrics, and artifacts

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(f"Step {self.step_type} must implement run()")

    def get_param(self, key: str, context: StepContext, default: Any = None) -> Any:
        """
        Get parameter with priority: user_params > step_params > default.

        Args:
            key: Parameter name
            context: Step context containing user_params
            default: Default value if not found

        Returns:
            Parameter value from highest priority source
        """
        # 1. User runtime overrides (highest priority)
        if key in context.user_params:
            return context.user_params[key]

        # 2. YAML step configuration
        if key in self.params:
            return self.params[key]

        # 3. Default value
        return default
