"""Step registration system for pipeline framework."""

from typing import Dict, Type, List

# Global registry mapping step types to their classes
_STEP_REGISTRY: Dict[str, Type] = {}


def register_step(step_type: str):
    """
    Decorator to register a step class with the framework.

    Usage:
        @register_step("ingestion.compress")
        class CompressionStep(PipelineStep):
            ...

    Args:
        step_type: Unique identifier for this step type

    Returns:
        Decorator function that registers the class

    Raises:
        ValueError: If step_type is already registered
    """

    def decorator(cls):
        if step_type in _STEP_REGISTRY:
            raise ValueError(
                f"Step type '{step_type}' is already registered to {_STEP_REGISTRY[step_type].__name__}"
            )

        # Set step_type as class attribute
        cls.step_type = step_type

        # Register the class
        _STEP_REGISTRY[step_type] = cls

        return cls

    return decorator


def get_step_class(step_type: str) -> Type:
    """
    Retrieve a registered step class by its type identifier.

    Args:
        step_type: Step type identifier (e.g., "ingestion.compress")

    Returns:
        The registered step class

    Raises:
        KeyError: If step_type is not registered
    """
    if step_type not in _STEP_REGISTRY:
        available = ", ".join(sorted(_STEP_REGISTRY.keys()))
        raise KeyError(
            f"Step type '{step_type}' not found in registry. "
            f"Available steps: {available}"
        )

    return _STEP_REGISTRY[step_type]


def available_steps() -> List[str]:
    """
    Get a sorted list of all registered step types.

    Returns:
        List of step type identifiers
    """
    return sorted(_STEP_REGISTRY.keys())
