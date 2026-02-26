# Graph Node Type Registry

A modular architecture for defining and managing graph node types in the MMCT video knowledge graph.

## Overview

The registry pattern allows adding new node types without modifying existing code. Each node type is self-contained in its own file and automatically registers itself on import.

```
mmct/graph/
├── __init__.py          # Package init, exports node_registry
├── base.py              # BaseNodeType ABC, EdgeDefinition classes
├── registry.py          # NodeTypeRegistry singleton
└── node_types/
    ├── __init__.py      # Imports all node types (triggers auto-registration)
    ├── chapter_group.py
    ├── chapter.py
    ├── transcript.py
    ├── event.py
    ├── object.py
    └── keyframe.py
```

## Quick Start

### Using the Registry

```python
from mmct.graph import node_registry

# List all registered node types
node_registry.names()  # ['ChapterGroup', 'Chapter', 'Transcript', 'Event', 'Object', 'Keyframe']

# Get searchable types only (excludes Keyframe which uses image embeddings)
node_registry.searchable_names()  # ['ChapterGroup', 'Chapter', 'Transcript', 'Event', 'Object']

# Get a specific node type
chapter = node_registry.get("Chapter")
chapter.neo4j_properties  # ['node_id', 'video_id', 'chunk_index', ...]
chapter.is_searchable     # True

# Infer type from node ID
node_registry.infer_type_from_id("chapter_xyz_001")  # "Chapter"
node_registry.infer_type_from_id("transcript_abc_002")  # "Transcript"

# Get embedding text for a node
attrs = {"transcript": "Hello world", "video_id": "abc"}
text = node_registry.get("Transcript").get_embedding_text(attrs)  # "Hello world"
```

---

## Adding a New Node Type

### Step 1: Create Pydantic Model (if needed)

Add your model to `mmct/video_pipeline/core/ingestion/models.py`:

```python
class GraphMyNode(BaseModel):
    """Model for MyNode in the knowledge graph."""
    id: str
    video_id: str
    content: str
    timestamp: Optional[float] = None
    # ... other fields
```

### Step 2: Create Node Type File

Create `mmct/graph/node_types/my_node.py`:

```python
"""MyNode node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry


class MyNodeNodeType(BaseNodeType):
    """MyNode - description of what this node represents.
    
    Detailed explanation of when to use this node type,
    how it differs from similar types, etc.
    """
    
    # =========================================================================
    # IDENTITY (required)
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Node label in Neo4j. Must be unique."""
        return "MyNode"
    
    @property
    def id_prefix(self) -> str:
        """Prefix for node IDs. Used for type inference."""
        return "mynode_"
    
    # =========================================================================
    # SCHEMA (required)
    # =========================================================================
    
    @property
    def model_class(self) -> Type:
        """Pydantic model class for validation."""
        from mmct.video_pipeline.core.ingestion.models import GraphMyNode
        return GraphMyNode
    
    @property
    def neo4j_properties(self) -> List[str]:
        """Properties to return in queries. NEVER include 'embedding'."""
        return [
            "node_id", "video_id", "content", "timestamp"
        ]
    
    # =========================================================================
    # EMBEDDING (required: get_embedding_text)
    # =========================================================================
    
    # Optional: Override if not using default 384-dim BGE-small
    # @property
    # def embedding_dimension(self) -> int:
    #     return 512  # For CLIP or other models
    
    # Optional: Override if using custom index name
    # @property
    # def embedding_index_name(self) -> str:
    #     return "mynode_custom_index"
    
    # Optional: Set to False if node doesn't support text vector search
    # @property
    # def is_searchable(self) -> bool:
    #     return False  # E.g., for image-only nodes like Keyframe
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        """Extract text for embedding generation."""
        return attrs.get("content", "") or ""
    
    # =========================================================================
    # GRAPH CONSTRUCTION (required)
    # =========================================================================
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        """Convert Pydantic model to Neo4j properties."""
        return {
            "video_id": instance.video_id or "",
            "content": instance.content or "",
            "timestamp": instance.timestamp or 0.0,
        }
    
    # =========================================================================
    # QUERY FORMATTING (required)
    # =========================================================================
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Format properties for search results. Truncate long text."""
        return {
            "content": (props.get("content", "") or "")[:500],  # Truncate
            "timestamp": props.get("timestamp"),
        }
    
    # =========================================================================
    # TIME FILTERING (optional)
    # =========================================================================
    
    @property
    def time_property(self) -> Optional[str]:
        """Property for time-based filtering."""
        return "timestamp"  # or "start_time" for range nodes
    
    @property
    def supports_time_range_filter(self) -> bool:
        """True if node has start_time AND end_time."""
        return False  # Set True for range nodes like Chapter, Transcript
    
    # =========================================================================
    # RELATIONSHIPS (optional)
    # =========================================================================
    
    def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
        """Define edges FROM this node TO other nodes."""
        return {
            "ChildNode": EdgeDefinition("HAS_CHILD", "ChildNode", "out"),
        }
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        """Define edges TO this node FROM other nodes (for traversal)."""
        return {
            "ParentNode": EdgeDefinition("HAS_MYNODE", "ParentNode", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        """Define NEXT/PREV edges for temporal ordering."""
        return TemporalEdgeDefinition(
            next_edge="NEXT_MYNODE",
            prev_edge="PREV_MYNODE",
            order_by="timestamp",  # Property to order by
        )
    
    # =========================================================================
    # AGENT PROMPTS (optional)
    # =========================================================================
    
    @property
    def description(self) -> str:
        """Short description for agent prompts."""
        return "Custom content nodes for specific use case"
    
    @property
    def use_cases(self) -> List[str]:
        """Example queries this node type answers."""
        return [
            "Use case 1 | MyNode | \"Example query 1\"",
            "Use case 2 | MyNode | \"Example query 2\"",
        ]


# ⚠️ CRITICAL: Auto-register on module import
node_registry.register(MyNodeNodeType())
```

### Step 3: Register in Package

Edit `mmct/graph/node_types/__init__.py`:

```python
from mmct.graph.node_types.chapter_group import ChapterGroupNodeType
from mmct.graph.node_types.chapter import ChapterNodeType
# ... existing imports ...
from mmct.graph.node_types.my_node import MyNodeNodeType  # ← ADD THIS

__all__ = [
    "ChapterGroupNodeType",
    "ChapterNodeType",
    # ... existing exports ...
    "MyNodeNodeType",  # ← ADD THIS
]
```

### Step 4: Verify Registration

```python
from mmct.graph import node_registry

assert "MyNode" in node_registry.names()
my_node = node_registry.get("MyNode")
assert my_node.id_prefix == "mynode_"
print("✓ MyNode registered successfully")
```

---

## Removing a Node Type

### Step 1: Remove from Package Init

Edit `mmct/graph/node_types/__init__.py` - remove the import and `__all__` entry.

### Step 2: Delete Node Type File

```bash
rm mmct/graph/node_types/my_node.py
```

### Step 3: Update Related Node Types (if any)

If other node types reference the removed type in their edge definitions, update them:

```python
# In parent_node.py - remove edge to deleted type
def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
    return {
        # "MyNode": EdgeDefinition(...),  ← REMOVE THIS
        "OtherNode": EdgeDefinition(...),
    }
```

### Step 4: Update Traversal Maps (if needed)

Check `mmct/v4/query/neo4j_provider.py` for hardcoded `TRAVERSAL_MAP` or `MULTI_HOP_PATHS` entries and remove them.

---

## Adding/Removing Relationships

### Adding a New Relationship

Edit the source node type's `get_outgoing_edges()` or target node type's `get_incoming_edges()`:

```python
# In chapter.py - add outgoing edge to new node type
def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
    return {
        "Event": EdgeDefinition("HAS_EVENT", "Event", "out"),
        "Keyframe": EdgeDefinition("HAS_KEYFRAME", "Keyframe", "out"),
        "Transcript": EdgeDefinition("HAS_TRANSCRIPT", "Transcript", "out"),
        "MyNewNode": EdgeDefinition("HAS_MYNEWNODE", "MyNewNode", "out"),  # ← ADD
    }
```

For bidirectional traversal, also add to the target's `get_incoming_edges()`:

```python
# In my_new_node.py
def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
    return {
        "Chapter": EdgeDefinition("HAS_MYNEWNODE", "Chapter", "in"),  # ← ADD
    }
```

### Adding Temporal Edges (NEXT/PREV)

```python
def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
    return TemporalEdgeDefinition(
        next_edge="NEXT_MYNODE",
        prev_edge="PREV_MYNODE",
        order_by="chunk_index",  # Property used to determine order
    )
```

### Removing a Relationship

Remove the entry from `get_outgoing_edges()` or `get_incoming_edges()` in the relevant node type files.

---

## BaseNodeType Reference

### Required Properties (must implement)

| Property/Method | Type | Description |
|----------------|------|-------------|
| `name` | `str` | Node label in Neo4j (e.g., "Chapter") |
| `id_prefix` | `str` | ID prefix for type inference (e.g., "chapter_") |
| `model_class` | `Type[BaseModel]` | Pydantic model for validation |
| `neo4j_properties` | `List[str]` | Properties to return in queries (NO 'embedding') |
| `get_embedding_text(attrs)` | `str` | Extract text for embedding |
| `create_node_properties(instance)` | `Dict` | Convert model to Neo4j props |
| `format_search_result(props)` | `Dict` | Format for search results |

### Optional Properties (have defaults)

| Property | Default | Description |
|----------|---------|-------------|
| `embedding_dimension` | `384` | Vector dimension (BGE-small) |
| `embedding_index_name` | `"{name}_embedding_index"` | Neo4j index name |
| `is_searchable` | `True` | Include in vector search |
| `time_property` | `None` | Property for time filtering |
| `supports_time_range_filter` | `False` | Has start_time/end_time |
| `description` | `"{name} nodes"` | For agent prompts |
| `use_cases` | `[]` | Example queries |

### Optional Methods (return empty by default)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_outgoing_edges()` | `Dict[str, EdgeDefinition]` | Edges from this node |
| `get_incoming_edges()` | `Dict[str, EdgeDefinition]` | Edges to this node |
| `get_temporal_edges()` | `Optional[TemporalEdgeDefinition]` | NEXT/PREV edges |

---

## EdgeDefinition Reference

```python
@dataclass
class EdgeDefinition:
    edge_type: str      # Relationship name (e.g., "HAS_TRANSCRIPT")
    target_type: str    # Target node type name (e.g., "Transcript")
    direction: str      # "out" (source→target) or "in" (source←target)
    properties: Dict    # Optional static edge properties
```

### Example

```python
# Chapter -[HAS_EVENT]-> Event
EdgeDefinition("HAS_EVENT", "Event", "out")

# Transcript <-[HAS_TRANSCRIPT]- Chapter (incoming)
EdgeDefinition("HAS_TRANSCRIPT", "Chapter", "in")
```

---

## TemporalEdgeDefinition Reference

```python
@dataclass
class TemporalEdgeDefinition:
    next_edge: str   # e.g., "NEXT_CHAPTER"
    prev_edge: str   # e.g., "PREV_CHAPTER"
    order_by: str    # Property to sort by (e.g., "chunk_index")
```

---

## Registry API Reference

```python
from mmct.graph import node_registry

# Registration
node_registry.register(MyNodeType())  # Register a node type

# Lookup
node_registry.get("Chapter")           # Get by name (or None)
node_registry.all()                    # All node types
node_registry.names()                  # All type names
node_registry.searchable_names()       # Types with is_searchable=True

# Type Inference
node_registry.infer_type_from_id("chapter_xyz_001")  # Returns "Chapter"

# Convenience Methods
node_registry.get_neo4j_properties("Chapter")        # Property list
node_registry.get_index_name("Chapter")              # Index name
node_registry.get_embedding_text("Chapter", attrs)   # Embedding text
node_registry.format_result("Chapter", props)        # Formatted result

# Map Building (for consumers)
node_registry.build_index_map()       # {type_name: index_name}
node_registry.build_properties_map()  # {type_name: [properties]}
node_registry.build_id_prefix_map()   # {prefix: type_name}
node_registry.build_traversal_map()   # {(src, tgt): (edge, dir)}
```

---

## Checklist for Adding a Node Type

- [ ] Create Pydantic model in `models.py` (if needed)
- [ ] Create `mmct/graph/node_types/your_node.py`
- [ ] Implement all required properties and methods
- [ ] Add `node_registry.register(YourNodeType())` at end of file
- [ ] Import in `mmct/graph/node_types/__init__.py`
- [ ] Verify with `node_registry.names()`
- [ ] Update parent node types with outgoing edges (if any)
- [ ] Update `TRAVERSAL_MAP` in `neo4j_provider.py` (if needed for traversal)
- [ ] Update `MULTI_HOP_PATHS` in `neo4j_provider.py` (if multi-hop traversal needed)
- [ ] Test embedding text extraction
- [ ] Test search result formatting

---

## Example: Current Node Types

| Node Type | ID Prefix | Searchable | Time Filter | Description |
|-----------|-----------|------------|-------------|-------------|
| ChapterGroup | `chaptergroup_` | ✓ | ✗ | Video-level topic clusters |
| Chapter | `chapter_` | ✓ | Range | Multimodal chunk summaries |
| Transcript | `transcript_` | ✓ | Range | Raw verbal content |
| Event | `event_` | ✓ | Point | Actions/happenings |
| Object | `object_` | ✓ | Point | Entities (people, items) |
| Keyframe | `kf_` | ✗ | Point | Visual frames (CLIP embeddings) |
