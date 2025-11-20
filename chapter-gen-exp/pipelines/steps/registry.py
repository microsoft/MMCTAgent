"""Registry helpers for pluggable pipeline steps."""
from __future__ import annotations

from typing import Dict, Iterable, Type

from .base import PipelineStep

_STEP_REGISTRY: Dict[str, Type[PipelineStep]] = {}


def register_step(step_type: str):
    """Class decorator used by step implementations to self-register."""

    def decorator(cls: Type[PipelineStep]) -> Type[PipelineStep]:
        if step_type in _STEP_REGISTRY:
            raise ValueError(f"Step type '{step_type}' already registered")
        cls.step_type = step_type
        _STEP_REGISTRY[step_type] = cls
        return cls

    return decorator


def get_step_class(step_type: str) -> Type[PipelineStep]:
    try:
        return _STEP_REGISTRY[step_type]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Step type '{step_type}' is not registered. Available: {sorted(_STEP_REGISTRY)}"
        ) from exc


def available_steps() -> Iterable[str]:
    return sorted(_STEP_REGISTRY)
