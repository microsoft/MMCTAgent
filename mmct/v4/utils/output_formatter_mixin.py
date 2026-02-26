"""Output formatter mixin for V4 tools.

Provides a mixin class that adds TOON/JSON output formatting
capability to tool classes without requiring inheritance changes.
"""

import os
import json
from typing import Any, Optional

from .toon_encoder import to_toon


class OutputFormatterMixin:
    """Mixin that provides format_output() method for tools.
    
    Adds the ability to output data in either TOON (token-efficient)
    or JSON format, configurable via environment or per-instance.
    
    Usage:
        class MyTool(OutputFormatterMixin):
            def my_method(self):
                data = {"results": [...]}
                return self.format_output(data)  # Returns TOON or JSON
    
    Configuration:
        - Default: TOON format (for token efficiency)
        - Environment: Set MMCT_OUTPUT_FORMAT=json to use JSON
        - Per-instance: tool.output_format = "json"
    
    Attributes:
        _output_format: Instance-level format override (None uses env/default)
    """
    
    # Instance-level format override
    _output_format: Optional[str] = None
    
    @property
    def output_format(self) -> str:
        """Get output format from instance, environment, or default.
        
        Priority:
            1. Instance-level _output_format if set
            2. MMCT_OUTPUT_FORMAT environment variable
            3. Default: "toon"
        
        Returns:
            Format string: "toon" or "json"
        """
        if self._output_format:
            return self._output_format
        return os.environ.get("MMCT_OUTPUT_FORMAT", "toon").lower()
    
    @output_format.setter
    def output_format(self, value: Optional[str]) -> None:
        """Set output format for this instance.
        
        Args:
            value: "toon", "json", or None (to use env/default)
        """
        self._output_format = value.lower() if value else None
    
    def format_output(self, data: Any) -> str:
        """Format data as TOON or JSON based on configuration.
        
        Args:
            data: Python dict/list to format
            
        Returns:
            Formatted string (TOON or JSON)
            
        Note:
            Falls back to JSON if TOON encoding fails for any reason.
        """
        if self.output_format == "json":
            return json.dumps(data, indent=2)
        
        try:
            return to_toon(data)
        except Exception:
            # Fallback to JSON on any encoding error
            return json.dumps(data, indent=2)
