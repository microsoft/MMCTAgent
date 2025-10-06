"""
Lively MCP Server

MCP (Model Context Protocol) server for visual search and video understanding.
Provides tools for searching video keyframes using natural language queries.
"""

from .tools import search_keyframes, get_visual_timestamps, VisualSearchClient

__version__ = "1.0.0"

__all__ = [
    "search_keyframes",
    "get_visual_timestamps",
    "VisualSearchClient",
]
