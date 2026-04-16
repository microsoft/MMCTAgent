"""Node type registry singleton.

Provides centralized registration and lookup of all graph node types.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mmct.video_pipeline.core.graph.base import BaseNodeType, EdgeDefinition


class NodeTypeRegistry:
    """Singleton registry for all graph node types.
    
    Node types register themselves on import. The registry provides:
    - Lookup by name or ID prefix
    - Validation of node type names
    - Centralized access to properties, indexes, and formatting
    
    Usage:
        from mmct.video_pipeline.core.graph import node_registry
        
        # Get all registered types
        types = node_registry.names()
        
        # Get specific type
        chapter = node_registry.get("Chapter")
        
        # Infer type from node ID
        type_name = node_registry.infer_type_from_id("chapter_xyz_001")
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._node_types: Dict[str, "BaseNodeType"] = {}
            cls._instance._initialized = False
        return cls._instance
    
    def register(self, node_type: "BaseNodeType") -> None:
        """Register a node type.
        
        Args:
            node_type: BaseNodeType instance to register.
        """
        self._node_types[node_type.name] = node_type
    
    def get(self, name: str) -> Optional["BaseNodeType"]:
        """Get node type by name.
        
        Args:
            name: Node type name (e.g., "Chapter").
            
        Returns:
            BaseNodeType instance or None if not found.
        """
        return self._node_types.get(name)
    
    def all(self) -> List["BaseNodeType"]:
        """Get all registered node types.
        
        Returns:
            List of all BaseNodeType instances.
        """
        return list(self._node_types.values())
    
    def names(self) -> List[str]:
        """Get all registered node type names.
        
        Returns:
            List of node type names.
        """
        return list(self._node_types.keys())
    
    def searchable_names(self) -> List[str]:
        """Get names of node types that support vector search.
        
        Returns:
            List of searchable node type names.
        """
        return [nt.name for nt in self._node_types.values() if nt.is_searchable]
    
    def infer_type_from_id(self, node_id: str) -> Optional[str]:
        """Infer node type from node ID prefix.
        
        Args:
            node_id: Node identifier (e.g., "chapter_abc_001").
            
        Returns:
            Node type name or None if cannot infer.
        """
        node_id_lower = node_id.lower()
        for node_type in self._node_types.values():
            if node_id_lower.startswith(node_type.id_prefix.lower()):
                return node_type.name
        return None
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def get_neo4j_properties(self, name: str) -> List[str]:
        """Get Neo4j properties for a node type.
        
        Args:
            name: Node type name.
            
        Returns:
            List of property names or empty list if type not found.
        """
        node_type = self.get(name)
        return node_type.neo4j_properties if node_type else []
    
    def get_index_name(self, name: str) -> str:
        """Get Neo4j index name for a node type.
        
        Args:
            name: Node type name.
            
        Returns:
            Index name or empty string if type not found.
        """
        node_type = self.get(name)
        return node_type.embedding_index_name if node_type else ""
    
    def get_embedding_text(self, name: str, attrs: Dict[str, Any]) -> str:
        """Get text to embed for a node.
        
        Args:
            name: Node type name.
            attrs: Node attributes dictionary.
            
        Returns:
            Text to embed or empty string if type not found.
        """
        node_type = self.get(name)
        return node_type.get_embedding_text(attrs) if node_type else ""
    
    def format_result(self, name: str, props: Dict[str, Any]) -> Dict[str, Any]:
        """Format search result for a node type.
        
        Args:
            name: Node type name.
            props: Raw node properties.
            
        Returns:
            Formatted properties or original props if type not found.
        """
        node_type = self.get(name)
        return node_type.format_search_result(props) if node_type else props
    
    # =========================================================================
    # TRAVERSAL MAPS
    # =========================================================================
    
    def build_traversal_map(self) -> Dict[tuple, tuple]:
        """Build traversal map from all registered node types.
        
        Returns:
            Dict mapping (source_type, target_type) to (edge_type, direction).
        """
        traversal_map = {}
        
        for node_type in self._node_types.values():
            # Outgoing edges (DOWN traversals)
            for target_name, edge_def in node_type.get_outgoing_edges().items():
                key = (node_type.name, target_name)
                traversal_map[key] = (edge_def.edge_type, edge_def.direction)
            
            # Incoming edges (UP traversals)
            for source_name, edge_def in node_type.get_incoming_edges().items():
                key = (node_type.name, source_name)
                # Reverse direction for incoming
                direction = "in" if edge_def.direction == "out" else "out"
                traversal_map[key] = (edge_def.edge_type, direction)
        
        return traversal_map
    
    def build_id_prefix_map(self) -> Dict[str, str]:
        """Build ID prefix to type name mapping.
        
        Returns:
            Dict mapping ID prefix to node type name.
        """
        return {
            node_type.id_prefix: node_type.name
            for node_type in self._node_types.values()
        }
    
    def build_index_map(self) -> Dict[str, str]:
        """Build node type to index name mapping.
        
        Returns:
            Dict mapping node type name to Neo4j index name.
        """
        return {
            node_type.name: node_type.embedding_index_name
            for node_type in self._node_types.values()
            if node_type.is_searchable
        }
    
    def build_properties_map(self) -> Dict[str, List[str]]:
        """Build node type to properties mapping.
        
        Returns:
            Dict mapping node type name to list of Neo4j properties.
        """
        return {
            node_type.name: node_type.neo4j_properties
            for node_type in self._node_types.values()
        }
    
    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================
    
    def generate_prompt_documentation(self) -> str:
        """Generate node type documentation for agent prompts.
        
        Returns:
            Markdown-formatted documentation string.
        """
        lines = ["# KNOWLEDGE GRAPH STRUCTURE", ""]
        lines.append("The system stores video information at multiple granularity levels:")
        
        for node_type in self._node_types.values():
            desc = node_type.description
            lines.append(f"- **{node_type.name}**: {desc}")
        
        lines.append("")
        lines.append("# QUERY TYPE TO GRANULARITY MAPPING")
        lines.append("")
        lines.append("| Query Type | Granularity Levels | Example |")
        lines.append("|------------|-------------------|---------|")
        
        for node_type in self._node_types.values():
            for use_case in node_type.use_cases:
                if "|" in use_case:
                    lines.append(f"| {use_case} |")
        
        return "\n".join(lines)


# Global singleton instance
node_registry = NodeTypeRegistry()
