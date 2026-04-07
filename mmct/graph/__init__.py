"""Graph node type registry and base classes.

This package provides a modular architecture for defining graph node types.
Each node type is self-contained and registers itself with the global registry.

Usage:
    from mmct.graph import node_registry, BaseNodeType
    
    # Get all registered node types
    all_types = node_registry.names()
    
    # Get a specific node type
    chapter_type = node_registry.get("Chapter")
    
    # Infer type from node ID
    node_type = node_registry.infer_type_from_id("chapter_abc123_001")
"""

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry

# Auto-register all node types on import
from mmct.graph import node_types  # noqa: F401

__all__ = [
    "BaseNodeType",
    "EdgeDefinition",
    "TemporalEdgeDefinition",
    "node_registry",
]
