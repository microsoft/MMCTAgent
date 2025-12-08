"""Data store for inter-step communication in pipelines."""

from typing import Any, Dict, Optional


class StepDataStore:
    """
    Thread-safe data store for passing data between pipeline steps.

    Each step can store and retrieve outputs by step_id and key.
    This enables loose coupling between steps while maintaining data dependencies.

    Example:
        # Step 1: Store compressed video path
        data_store.set("compress", "video_path", "/path/to/compressed.mp4")

        # Step 2: Retrieve compressed video path
        video_path = data_store.get("compress", "video_path")
    """

    def __init__(self):
        """Initialize empty data store."""
        self._data: Dict[str, Dict[str, Any]] = {}

    def set(self, step_id: str, key: str, value: Any) -> None:
        """
        Store a value for a specific step and key.

        Args:
            step_id: ID of the step storing the data
            key: Key to store the value under
            value: Value to store
        """
        if step_id not in self._data:
            self._data[step_id] = {}

        self._data[step_id][key] = value

    def get(self, step_id: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from a specific step.

        Args:
            step_id: ID of the step that stored the data
            key: Key to retrieve
            default: Default value if not found

        Returns:
            The stored value, or default if not found
        """
        if step_id not in self._data:
            return default

        return self._data[step_id].get(key, default)

    def get_all(self, step_id: str) -> Dict[str, Any]:
        """
        Get all data stored by a specific step.

        Args:
            step_id: ID of the step

        Returns:
            Dictionary of all key-value pairs for that step
        """
        return self._data.get(step_id, {}).copy()

    def has(self, step_id: str, key: Optional[str] = None) -> bool:
        """
        Check if a step has stored data.

        Args:
            step_id: ID of the step
            key: Optional specific key to check

        Returns:
            True if the data exists, False otherwise
        """
        if step_id not in self._data:
            return False

        if key is None:
            return True

        return key in self._data[step_id]

    def clear(self, step_id: Optional[str] = None) -> None:
        """
        Clear data from the store.

        Args:
            step_id: If provided, clear only that step's data.
                    If None, clear all data.
        """
        if step_id is None:
            self._data.clear()
        elif step_id in self._data:
            del self._data[step_id]

    def __repr__(self) -> str:
        """String representation of the data store."""
        return f"StepDataStore(steps={list(self._data.keys())})"
