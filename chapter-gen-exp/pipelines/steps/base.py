"""Base interfaces shared by all experimentation steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from ..utils.video import VideoMetadata


class StepDataStore:
    """Lightweight in-memory store shared across pipeline steps."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def save(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any | None = None) -> Any:
        return self._store.get(key, default)

    def namespace(self, name: str) -> "StepNamespace":
        return StepNamespace(self, name)

    def as_dict(self) -> Mapping[str, Any]:
        return dict(self._store)


class StepNamespace(MutableMapping[str, Any]):
    """Convenience view into a nested namespace inside the datastore."""

    def __init__(self, store: StepDataStore, name: str) -> None:
        self._store = store
        self._name = name
        if name not in store._store:
            store._store[name] = {}

    def _bucket(self) -> Dict[str, Any]:
        return self._store._store[self._name]

    def __getitem__(self, key: str) -> Any:
        return self._bucket()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._bucket()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._bucket()[key]

    def __iter__(self):
        return iter(self._bucket())

    def __len__(self) -> int:
        return len(self._bucket())


@dataclass
class StepContext:
    """Runtime context shared with every step execution."""

    video_uri: str
    transcript_path: str
    output_dir: Path
    video_duration_seconds: Optional[float]
    metadata: Dict[str, Any]
    data_store: StepDataStore
    video_metadata: Optional[VideoMetadata] = None


@dataclass
class StepResult:
    """Standard return payload emitted by each step."""

    step_id: str
    produced: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)


class PipelineStep:
    """Base class for swappable pipeline steps."""

    step_type: str = "base"
    description: str = ""

    def __init__(self, step_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.step_id = step_id
        self.params = params or {}

    def run(self, context: StepContext) -> StepResult:  # pragma: no cover - abstract method
        raise NotImplementedError
