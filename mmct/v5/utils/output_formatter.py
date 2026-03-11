"""Output formatter mixin for V5 tools.

Provides TOON/JSON output formatting for tool results.
"""

import os
import json
from typing import Any, Optional

from .toon_encoder import to_toon


class OutputFormatterMixin:
    """Mixin that provides format_output() for executor classes.

    Outputs data in TOON (token-efficient) or JSON format.
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
        if self.output_format == "json":
            return json.dumps(data, indent=2)
        try:
            return to_toon(data)
        except Exception:
            return json.dumps(data, indent=2)
