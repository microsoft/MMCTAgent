"""Output formatter mixin for video pipeline tools and executors.

Provides TOON/JSON output formatting configurable via environment variable
or per-instance override.
"""

import os
import json
from typing import Any, Optional

from .toon_encoder import to_toon


class OutputFormatterMixin:
    """Mixin that provides format_output() for tool and executor classes.

    Outputs data in TOON (token-efficient) or JSON format.

    Configuration:
        - Default: TOON format (for token efficiency)
        - Environment: Set MMCT_OUTPUT_FORMAT=json to use JSON
        - Per-instance: tool.output_format = "json"
    """

    _output_format: Optional[str] = None

    @property
    def output_format(self) -> str:
        if self._output_format:
            return self._output_format
        return os.environ.get("MMCT_OUTPUT_FORMAT", "toon").lower()

    @output_format.setter
    def output_format(self, value: Optional[str]) -> None:
        self._output_format = value.lower() if value else None

    def format_output(self, data: Any) -> str:
        """Format data as TOON or JSON. Falls back to JSON if TOON encoding fails."""
        if self.output_format == "json":
            return json.dumps(data, indent=2)
        try:
            return to_toon(data)
        except Exception:
            return json.dumps(data, indent=2)
