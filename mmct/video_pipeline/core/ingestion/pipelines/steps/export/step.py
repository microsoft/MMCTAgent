"""Export pipeline step.

Exports pipeline outputs to local files:
- Chapters as JSON (from chapters step)
- Events, objects, and graph structure as JSON (from graph_construction step)
- Interactive HTML graph visualization using pyvis

Output directory: {output_dir}/export/{run_id}_{video_id}/
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generate a unique pipeline run ID.
    
    Format: YYYYMMDD_HHMMSS_<short_uuid>
    Example: 20260215_060512_a1b2c3
    
    Returns:
        Unique run ID string
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


def _generate_graph_html(
    graph_provider: Any,
    output_path: str,
    video_id: str,
) -> bool:
    """Generate interactive HTML visualization of the hierarchical graph.
    
    Creates a standalone HTML file with:
    - Full hierarchical graph visualization using vis.js
    - Filter controls for node types (ChapterGroup, Chapter, Keyframe, Event, Object)
    - Filter controls for all edge types
    - Search functionality
    - Node details panel
    - Keyboard navigation
    
    Args:
        graph_provider: NetworkX graph provider with _graph attribute.
        output_path: Path to save the HTML file.
        video_id: Video ID for the title.
        
    Returns:
        True if successful, False otherwise.
    """
    if not hasattr(graph_provider, '_graph') or graph_provider._graph is None:
        logger.warning("Graph provider has no graph data")
        return False
    
    nx_graph = graph_provider._graph
    
    if nx_graph.number_of_nodes() == 0:
        logger.warning("Graph is empty, skipping visualization")
        return False
    
    # Node type styling configuration
    node_styles = {
        "ChapterGroup": {"color": "#9b59b6", "shape": "diamond", "size": 35, "label_field": "name"},
        "Chapter": {"color": "#3498db", "shape": "box", "size": 25, "label_field": "summary"},
        "Transcript": {"color": "#2ecc71", "shape": "box", "size": 20, "label_field": "transcript"},
        "Keyframe": {"color": "#f1c40f", "shape": "image", "size": 20, "label_field": "timestamp"},
        "Event": {"color": "#e94560", "shape": "dot", "size": 20, "label_field": "description"},
        "Object": {"color": "#4ecdc4", "shape": "triangle", "size": 15, "label_field": "name"},
    }
    
    # Prepare nodes data
    nodes_data = []
    node_counts = {"ChapterGroup": 0, "Chapter": 0, "Transcript": 0, "Keyframe": 0, "Event": 0, "Object": 0}
    
    for node_id, attrs in nx_graph.nodes(data=True):
        node_type = attrs.get("_type", "Node")
        style = node_styles.get(node_type, {"color": "#888888", "shape": "dot", "size": 15, "label_field": "id"})
        
        # Get label from appropriate field
        label_field = style["label_field"]
        raw_label = attrs.get(label_field, node_id) or node_id
        
        # Truncate label based on node type
        max_len = 50 if node_type == "ChapterGroup" else 40 if node_type in ("Chapter", "Transcript") else 35
        label = str(raw_label)[:max_len]
        if len(str(raw_label)) > max_len:
            label += "..."
        
        # Track counts
        if node_type in node_counts:
            node_counts[node_type] += 1
        
        nodes_data.append({
            "id": node_id,
            "label": label,
            "color": style["color"],
            "shape": style["shape"],
            "size": style["size"],
            "group": node_type,
            "properties": {k: v for k, v in attrs.items() if k != "_type" and not k.startswith("embedding")}
        })
    
    # Prepare edges data
    edges_data = []
    edge_colors = {
        # Hierarchy edges
        "HAS_CHAPTER": "#9b59b6",
        "IN_GROUP": "#9b59b6",
        "HAS_KEYFRAME": "#f1c40f",
        "KEYFRAME_IN_CHAPTER": "#f1c40f",
        "HAS_EVENT": "#3498db",
        "IN_CHAPTER": "#3498db",
        # Temporal edges
        "NEXT_GROUP": "#f39c12",
        "PREV_GROUP": "#f39c12",
        "NEXT_CHAPTER": "#e67e22",
        "PREV_CHAPTER": "#e67e22",
        "NEXT_EVENT": "#f9a825",
        "PREV_EVENT": "#f9a825",
        # Event-Object edges
        "CONTAINS": "#16c79a",
        "APPEARS_IN": "#1abc9c",
        # Semantic edges
        "SIMILAR_TO": "#7b68ee",
        "CAUSES": "#ff6b6b",
        "RESULT_OF": "#ff6b6b",
    }
    
    # Handle both DiGraph and MultiDiGraph
    try:
        # MultiDiGraph returns (source, target, key, data)
        edge_iter = nx_graph.edges(data=True, keys=True)
        for source, target, key, attrs in edge_iter:
            edge_type = attrs.get("_type", "RELATED")
            color = edge_colors.get(edge_type, "#888888")
            
            edges_data.append({
                "from": source,
                "to": target,
                "label": edge_type,
                "color": {"color": color, "highlight": color},
                "edgeType": edge_type,
                "properties": {k: v for k, v in attrs.items() if k != "_type"}
            })
    except TypeError:
        # DiGraph returns (source, target, data)
        for source, target, attrs in nx_graph.edges(data=True):
            edge_type = attrs.get("_type", "RELATED")
            color = edge_colors.get(edge_type, "#888888")
            
            edges_data.append({
                "from": source,
                "to": target,
                "label": edge_type,
                "color": {"color": color, "highlight": color},
                "edgeType": edge_type,
                "properties": {k: v for k, v in attrs.items() if k != "_type"}
            })
    
    # Collect edge types
    edge_types = list(set(e["edgeType"] for e in edges_data))
    
    # Generate the HTML
    html_content = _generate_neo4j_style_html(
        nodes_data, edges_data, video_id, node_counts, edge_types, edge_colors
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(
        f"Generated graph visualization: {len(nodes_data)} nodes, "
        f"{len(edges_data)} edges"
    )
    return True


def _generate_neo4j_style_html(
    nodes: list,
    edges: list,
    video_id: str,
    node_counts: dict,
    edge_types: list,
    edge_colors: dict,
) -> str:
    """Generate Neo4j-style HTML with hierarchical graph visualization."""
    
    nodes_json = json.dumps(nodes, default=str)
    edges_json = json.dumps(edges, default=str)
    
    # Calculate total stats
    total_nodes = sum(node_counts.values())
    total_edges = len(edges)
    
    # Node type colors for CSS
    node_colors = {
        "ChapterGroup": "#9b59b6",
        "Chapter": "#3498db",
        "Keyframe": "#f1c40f",
        "Event": "#e94560",
        "Object": "#4ecdc4",
    }
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph: {video_id}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
            color: white;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .header h1 {{ font-size: 18px; font-weight: 500; }}
        .header .stats {{ font-size: 12px; opacity: 0.9; }}
        .stats-items {{ display: flex; gap: 12px; flex-wrap: wrap; }}
        .stats-item {{ display: flex; align-items: center; gap: 4px; }}
        .stats-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        
        /* Main container */
        .main {{ display: flex; flex: 1; overflow: hidden; }}
        
        /* Sidebar */
        .sidebar {{
            width: 300px;
            background: white;
            border-right: 1px solid #ddd;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }}
        .sidebar-section {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .sidebar-section h3 {{
            font-size: 11px;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}
        
        /* Search */
        .search-input {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }}
        .search-input:focus {{ outline: none; border-color: #4ecdc4; }}
        
        /* Filters */
        .filter-group {{ margin-bottom: 8px; }}
        .filter-label {{
            display: flex;
            align-items: center;
            cursor: pointer;
            padding: 3px 0;
            font-size: 12px;
        }}
        .filter-label input {{ margin-right: 8px; }}
        .filter-color {{
            width: 10px;
            height: 10px;
            border-radius: 2px;
            margin-right: 6px;
        }}
        .filter-section-title {{
            font-size: 10px;
            color: #999;
            margin: 8px 0 4px 0;
            text-transform: uppercase;
        }}
        
        /* Details panel */
        .details-panel {{
            background: #fafafa;
            flex: 1;
            overflow-y: auto;
        }}
        .details-content {{ padding: 12px; }}
        .details-content h4 {{
            font-size: 13px;
            margin-bottom: 10px;
            color: #333;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }}
        .node-type-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 10px;
            color: white;
            margin-left: 8px;
        }}
        .detail-row {{
            display: flex;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
            font-size: 11px;
        }}
        .detail-key {{
            width: 90px;
            color: #666;
            flex-shrink: 0;
        }}
        .detail-value {{
            color: #333;
            word-break: break-word;
            flex: 1;
        }}
        .no-selection {{
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 20px;
            font-size: 12px;
        }}
        
        /* Graph container */
        .graph-container {{
            flex: 1;
            background: #1a1a2e;
            position: relative;
        }}
        #graph {{ width: 100%; height: 100%; }}
        
        /* Controls overlay */
        .controls {{
            position: absolute;
            bottom: 15px;
            right: 15px;
            display: flex;
            gap: 8px;
        }}
        .control-btn {{
            background: rgba(255,255,255,0.9);
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .control-btn:hover {{ background: white; }}
        
        /* Legend */
        .legend {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(26,26,46,0.95);
            padding: 12px;
            border-radius: 6px;
            color: white;
            font-size: 10px;
            max-width: 180px;
        }}
        .legend-title {{ font-weight: 600; margin-bottom: 8px; font-size: 11px; }}
        .legend-section {{ margin-top: 8px; }}
        .legend-section-title {{ color: #888; font-size: 9px; margin-bottom: 4px; }}
        .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }}
        .legend-diamond {{ width: 10px; height: 10px; transform: rotate(45deg); margin-right: 6px; flex-shrink: 0; }}
        .legend-box {{ width: 10px; height: 8px; margin-right: 6px; flex-shrink: 0; border-radius: 2px; }}
        .legend-triangle {{ width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 10px solid; margin-right: 6px; flex-shrink: 0; }}
        .legend-line {{ width: 16px; height: 3px; margin-right: 6px; border-radius: 2px; flex-shrink: 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 {video_id}</h1>
        <div class="stats">
            <div class="stats-items">
                <span class="stats-item"><span class="stats-dot" style="background:#9b59b6"></span>{node_counts.get("ChapterGroup", 0)} Groups</span>
                <span class="stats-item"><span class="stats-dot" style="background:#3498db"></span>{node_counts.get("Chapter", 0)} Chapters</span>
                <span class="stats-item"><span class="stats-dot" style="background:#f1c40f"></span>{node_counts.get("Keyframe", 0)} Keyframes</span>
                <span class="stats-item"><span class="stats-dot" style="background:#e94560"></span>{node_counts.get("Event", 0)} Events</span>
                <span class="stats-item"><span class="stats-dot" style="background:#4ecdc4"></span>{node_counts.get("Object", 0)} Objects</span>
                <span class="stats-item">• {total_edges} Edges</span>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="sidebar">
            <div class="sidebar-section">
                <h3>Search</h3>
                <input type="text" class="search-input" id="searchInput" placeholder="Search nodes...">
            </div>
            
            <div class="sidebar-section">
                <h3>Node Types</h3>
                <div class="filter-group">
                    <label class="filter-label">
                        <input type="checkbox" data-filter="node" data-value="ChapterGroup">
                        <span class="filter-color" style="background: #9b59b6;"></span>
                        ChapterGroups ({node_counts.get("ChapterGroup", 0)})
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" data-filter="node" data-value="Chapter">
                        <span class="filter-color" style="background: #3498db;"></span>
                        Chapters ({node_counts.get("Chapter", 0)})
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" data-filter="node" data-value="Keyframe">
                        <span class="filter-color" style="background: #f1c40f;"></span>
                        Keyframes ({node_counts.get("Keyframe", 0)})
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" data-filter="node" data-value="Event">
                        <span class="filter-color" style="background: #e94560;"></span>
                        Events ({node_counts.get("Event", 0)})
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" data-filter="node" data-value="Object">
                        <span class="filter-color" style="background: #4ecdc4;"></span>
                        Objects ({node_counts.get("Object", 0)})
                    </label>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Edge Types</h3>
                <div class="filter-group" id="edgeFilters"></div>
            </div>
            
            <div class="details-panel">
                <div class="sidebar-section">
                    <h3>Selected Node</h3>
                </div>
                <div class="details-content" id="detailsContent">
                    <div class="no-selection">Click a node to see details</div>
                </div>
            </div>
        </div>
        
        <div class="graph-container">
            <div id="graph"></div>
            <div class="legend">
                <div class="legend-title">Node Types</div>
                <div class="legend-item"><span class="legend-diamond" style="background: #9b59b6;"></span> ChapterGroup</div>
                <div class="legend-item"><span class="legend-box" style="background: #3498db;"></span> Chapter</div>
                <div class="legend-item"><span class="legend-box" style="background: #f1c40f;"></span> Keyframe</div>
                <div class="legend-item"><span class="legend-dot" style="background: #e94560;"></span> Event</div>
                <div class="legend-item"><span class="legend-triangle" style="border-bottom-color: #4ecdc4;"></span> Object</div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Hierarchy</div>
                    <div class="legend-item"><span class="legend-line" style="background: #9b59b6;"></span> HAS_CHAPTER</div>
                    <div class="legend-item"><span class="legend-line" style="background: #f1c40f;"></span> HAS_KEYFRAME / KEYFRAME_IN_CHAPTER</div>
                    <div class="legend-item"><span class="legend-line" style="background: #3498db;"></span> HAS_EVENT</div>
                </div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Temporal</div>
                    <div class="legend-item"><span class="legend-line" style="background: #f39c12;"></span> NEXT_GROUP</div>
                    <div class="legend-item"><span class="legend-line" style="background: #e67e22;"></span> NEXT_CHAPTER</div>
                    <div class="legend-item"><span class="legend-line" style="background: #f9a825;"></span> NEXT_EVENT</div>
                </div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Semantic</div>
                    <div class="legend-item"><span class="legend-line" style="background: #16c79a;"></span> CONTAINS</div>
                    <div class="legend-item"><span class="legend-line" style="background: #7b68ee;"></span> SIMILAR_TO</div>
                    <div class="legend-item"><span class="legend-line" style="background: #ff6b6b;"></span> CAUSES</div>
                </div>
            </div>
            <div class="controls">
                <button class="control-btn" onclick="network.fit()">Fit View</button>
                <button class="control-btn" onclick="togglePhysics()">Toggle Physics</button>
                <button class="control-btn" onclick="toggleHierarchy()">Hierarchy Layout</button>
            </div>
        </div>
    </div>

    <script>
        // Data
        const rawNodes = {nodes_json};
        const rawEdges = {edges_json};
        const edgeTypes = {json.dumps(edge_types)};
        
        // Edge colors
        const edgeColors = {json.dumps(edge_colors)};
        
        // Node colors
        const nodeColors = {json.dumps(node_colors)};
        
        // Create vis.js datasets
        const nodes = new vis.DataSet(rawNodes.map(n => ({{
            id: n.id,
            label: n.label,
            color: n.color,
            shape: n.shape,
            size: n.size,
            group: n.group,
            font: {{ color: 'white', size: 10 }},
            properties: n.properties,
            // Level for hierarchical layout
            level: n.group === 'ChapterGroup' ? 0 : n.group === 'Chapter' ? 1 : n.group === 'Keyframe' ? 2 : n.group === 'Event' ? 3 : 4
        }})));
        
        const edges = new vis.DataSet(rawEdges.map((e, i) => ({{
            id: i,
            from: e.from,
            to: e.to,
            color: e.color,
            edgeType: e.edgeType,
            arrows: 'to',
            smooth: {{ type: 'curvedCW', roundness: 0.1 }},
            properties: e.properties
        }})));
        
        // Initialize network
        const container = document.getElementById('graph');
        const data = {{ nodes, edges }};
        const options = {{
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{ iterations: 200 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: false,
                keyboard: true
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        let physicsEnabled = true;
        let hierarchyMode = false;
        
        // Generate edge type filters grouped by category
        const edgeFiltersEl = document.getElementById('edgeFilters');
        const edgeCategories = {{
            'Hierarchy': ['HAS_CHAPTER', 'IN_GROUP', 'HAS_EVENT', 'IN_CHAPTER', 'HAS_KEYFRAME', 'KEYFRAME_IN_CHAPTER'],
            'Temporal': ['NEXT_GROUP', 'PREV_GROUP', 'NEXT_CHAPTER', 'PREV_CHAPTER', 'NEXT', 'PREVIOUS'],
            'Semantic': ['CONTAINS', 'APPEARS_IN', 'SIMILAR_TO', 'CAUSES', 'RESULT_OF']
        }};
        
        Object.entries(edgeCategories).forEach(([category, types]) => {{
            const categoryTypes = types.filter(t => edgeTypes.includes(t));
            if (categoryTypes.length === 0) return;
            
            edgeFiltersEl.innerHTML += `<div class="filter-section-title">${{category}}</div>`;
            categoryTypes.forEach(type => {{
                const color = edgeColors[type] || '#888';
                const count = rawEdges.filter(e => e.edgeType === type).length;
                edgeFiltersEl.innerHTML += `
                    <label class="filter-label">
                        <input type="checkbox" data-filter="edge" data-value="${{type}}">
                        <span class="filter-color" style="background: ${{color}};"></span>
                        ${{type}} (${{count}})
                    </label>
                `;
            }});
        }});
        
        // Filter functionality - start with all types selected (visible)
        const allNodeTypes = new Set(['ChapterGroup', 'Chapter', 'Keyframe', 'Event', 'Object']);
        const allEdgeTypes = new Set(edgeTypes);
        let activeNodeTypes = new Set(allNodeTypes);
        let activeEdgeTypes = new Set(allEdgeTypes);
        
        // Check all checkboxes initially
        document.querySelectorAll('input[data-filter]').forEach(checkbox => {{
            checkbox.checked = true;
        }});
        
        document.querySelectorAll('input[data-filter]').forEach(checkbox => {{
            checkbox.addEventListener('change', (e) => {{
                const filterType = e.target.dataset.filter;
                const value = e.target.dataset.value;
                
                if (filterType === 'node') {{
                    if (e.target.checked) activeNodeTypes.add(value);
                    else activeNodeTypes.delete(value);
                }} else {{
                    if (e.target.checked) activeEdgeTypes.add(value);
                    else activeEdgeTypes.delete(value);
                }}
                
                applyFilters();
            }});
        }});
        
        function applyFilters() {{
            // Filter nodes
            const visibleNodeIds = new Set();
            rawNodes.forEach(n => {{
                const visible = activeNodeTypes.has(n.group);
                if (visible) visibleNodeIds.add(n.id);
                nodes.update({{ id: n.id, hidden: !visible }});
            }});
            
            // Filter edges
            rawEdges.forEach((e, i) => {{
                const typeVisible = activeEdgeTypes.has(e.edgeType);
                const nodesVisible = visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to);
                edges.update({{ id: i, hidden: !typeVisible || !nodesVisible }});
            }});
        }}
        
        // Apply initial filter state (show all)
        applyFilters();
        
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            if (!query) {{
                rawNodes.forEach(n => nodes.update({{ id: n.id, hidden: !activeNodeTypes.has(n.group) }}));
                return;
            }}
            
            rawNodes.forEach(n => {{
                const matchesSearch = n.label.toLowerCase().includes(query) ||
                    n.id.toLowerCase().includes(query) ||
                    JSON.stringify(n.properties).toLowerCase().includes(query);
                const matchesType = activeNodeTypes.has(n.group);
                nodes.update({{ id: n.id, hidden: !matchesSearch || !matchesType }});
            }});
        }});
        
        // Node selection - show details
        network.on('selectNode', (params) => {{
            if (params.nodes.length === 0) return;
            const nodeId = params.nodes[0];
            const node = rawNodes.find(n => n.id === nodeId);
            if (!node) return;
            
            const bgColor = nodeColors[node.group] || '#888';
            let html = `<h4>${{node.id}}<span class="node-type-badge" style="background:${{bgColor}}">${{node.group}}</span></h4>`;
            const props = node.properties || {{}};
            
            // Priority properties by node type
            const priorityByType = {{
                'ChapterGroup': ['name', 'summary', 'topics', 'order', 'start_time', 'end_time', 'chapter_indices'],
                'Chapter': ['summary', 'transcript', 'chunk_index', 'start_time', 'end_time', 'group_index'],
                'Event': ['description', 'event_type', 'timestamp', 'duration', 'participants', 'chapter_index', 'sequence_number'],
                'Object': ['name', 'object_type', 'first_seen', 'last_seen', 'appearance', 'identity']
            }};
            
            const priority = priorityByType[node.group] || ['name', 'description'];
            
            priority.forEach(key => {{
                if (props[key] !== undefined && props[key] !== null) {{
                    let val = props[key];
                    if (Array.isArray(val)) val = val.join(', ');
                    if (typeof val === 'number') val = val.toFixed(2);
                    if (typeof val === 'string' && val.length > 200) val = val.substring(0, 200) + '...';
                    html += `<div class="detail-row"><span class="detail-key">${{key}}</span><span class="detail-value">${{val}}</span></div>`;
                }}
            }});
            
            // Show remaining properties
            Object.entries(props).forEach(([key, val]) => {{
                if (priority.includes(key)) return;
                if (key.startsWith('embedding') || key === 'metadata') return;
                if (typeof val === 'object') val = JSON.stringify(val);
                if (typeof val === 'number') val = val.toFixed(2);
                if (typeof val === 'string' && val.length > 100) val = val.substring(0, 100) + '...';
                html += `<div class="detail-row"><span class="detail-key">${{key}}</span><span class="detail-value">${{val}}</span></div>`;
            }});
            
            document.getElementById('detailsContent').innerHTML = html;
        }});
        
        network.on('deselectNode', () => {{
            document.getElementById('detailsContent').innerHTML = '<div class="no-selection">Click a node to see details</div>';
        }});
        
        // Toggle physics
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        // Toggle hierarchical layout
        function toggleHierarchy() {{
            hierarchyMode = !hierarchyMode;
            if (hierarchyMode) {{
                network.setOptions({{
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 150,
                            nodeSpacing: 100
                        }}
                    }},
                    physics: {{ enabled: false }}
                }});
                physicsEnabled = false;
            }} else {{
                network.setOptions({{
                    layout: {{ hierarchical: {{ enabled: false }} }},
                    physics: {{ enabled: true }}
                }});
                physicsEnabled = true;
            }}
        }}
        
        // Stabilize then disable physics
        network.once('stabilizationIterationsDone', () => {{
            setTimeout(() => {{
                network.setOptions({{ physics: {{ enabled: false }} }});
                physicsEnabled = false;
            }}, 500);
        }});
    </script>
</body>
</html>'''


@register_step("ingestion.export")
class ExportStep(PipelineStep):
    """Export pipeline outputs to local JSON files and HTML visualization.
    
    Extracts data from the constructed graph and exports:
    - Transcript text and SRT file (from transcribe step)
    - Chapters JSON (from chapters step)
    - Chapter groups JSON (from chapter_grouping step)
    - Transcript nodes JSON (Transcript nodes from graph - verbal content per chapter)
    - Events JSON (Event nodes from graph)
    - Objects JSON (Object nodes from graph)
    - Graph JSON (full graph structure)
    - Interactive HTML visualization (using pyvis)
    - Keyframes metadata JSON and image files (from keyframes step)
    
    Output: {output_dir}/export/{video_id}/
    
    Params:
        transcribe_step: Step ID for transcription (default: "transcribe")
        chapters_step: Step ID for chapters (default: "chapters")
        keyframes_step: Step ID for keyframes (default: "keyframes")
        chapter_grouping_step: Step ID for chapter groups (default: "chapter_grouping")
        graph_step: Step ID for constructed graph (default: "graph_construction")
        output_subdir: Subdirectory under output_dir (default: "export")
        include_transcript: Export transcript text and SRT file (default: True)
        include_chapters: Export chapters JSON (default: True)
        include_chapter_groups: Export chapter groups JSON (default: True)
        include_transcript_nodes: Export transcript nodes JSON (default: True)
        include_events: Export events JSON (default: True)
        include_objects: Export objects JSON (default: True)
        include_graph_json: Export graph structure JSON (default: True)
        include_graph_html: Export HTML visualization (default: True)
        include_keyframes: Export keyframes metadata and images (default: True)
        pretty_print: Format JSON with indentation (default: True)
    """
    
    step_type = "ingestion.export"
    description = "Export pipeline outputs to JSON files and HTML visualization."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute export."""
        import shutil
        
        # Get parameters
        transcribe_step = self.get_param("transcribe_step", context, default="transcribe")
        chapters_step = self.get_param("chapters_step", context, default="chapters")
        keyframes_step = self.get_param("keyframes_step", context, default="keyframes")
        chapter_grouping_step = self.get_param("chapter_grouping_step", context, default="chapter_grouping")
        graph_step = self.get_param("graph_step", context, default="graph_construction")
        output_subdir = self.get_param("output_subdir", context, default="export")
        
        include_transcript = self.get_param("include_transcript", context, default=True)
        include_chapters = self.get_param("include_chapters", context, default=None)
        if include_chapters is None:
            include_chapters = True
        include_chapter_groups = self.get_param("include_chapter_groups", context, default=True)
        include_events = self.get_param("include_events", context, default=True)
        include_objects = self.get_param("include_objects", context, default=True)
        include_graph_json = self.get_param("include_graph_json", context, default=True)
        include_graph_html = self.get_param("include_graph_html", context, default=True)
        include_keyframes = self.get_param("include_keyframes", context, default=True)
        pretty_print = self.get_param("pretty_print", context, default=True)
        
        video_id = getattr(context, 'video_id', 'unknown')
        
        # Generate unique pipeline run ID
        run_id = _generate_run_id()
        
        # Create output directory with run_id prefix: {output_dir}/{output_subdir}/{run_id}_{video_id}/
        export_folder_name = f"{run_id}_{video_id}"
        export_dir = os.path.join(context.output_dir, output_subdir, export_folder_name)
        os.makedirs(export_dir, exist_ok=True)
        
        logger.info(f"Exporting outputs to: {export_dir} (run_id={run_id})")
        
        # Track exported files and metrics
        exported_files: List[str] = []
        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "files_exported": 0,
            "transcript_word_count": 0,
            "chapters_count": 0,
            "chapter_groups_count": 0,
            "graph_chapter_groups_count": 0,
            "graph_chapters_count": 0,
            "transcript_nodes_count": 0,
            "events_count": 0,
            "objects_count": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "keyframes_count": 0,
            "keyframe_images_exported": 0,
        }
        
        indent = 2 if pretty_print else None
        
        # Export transcript
        if include_transcript:
            transcript_text = context.data_store.get(transcribe_step, "transcript")
            transcript_path = context.data_store.get(transcribe_step, "transcript_path")
            word_count = context.data_store.get(transcribe_step, "word_count") or 0
            
            if transcript_text:
                # Export transcript as plain text
                txt_filepath = os.path.join(export_dir, "transcript.txt")
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                exported_files.append(txt_filepath)
                metrics["files_exported"] += 1
                metrics["transcript_word_count"] = word_count
                logger.info(f"Exported transcript ({word_count} words)")
                
                # Copy SRT file if available
                if transcript_path and os.path.exists(transcript_path):
                    srt_filepath = os.path.join(export_dir, "transcript.srt")
                    try:
                        shutil.copy2(transcript_path, srt_filepath)
                        exported_files.append(srt_filepath)
                        metrics["files_exported"] += 1
                        logger.info(f"Exported transcript SRT file")
                    except Exception as e:
                        logger.warning(f"Failed to copy SRT file: {e}")
        
        # Export chapters
        if include_chapters:
            chapters = (
                context.data_store.get(chapters_step, "raw_chapters")
                or context.data_store.get(chapters_step, "chapters")
                or []
            )
            if chapters:
                filepath = self._export_json(export_dir, "chapters.json", chapters, indent)
                exported_files.append(filepath)
                metrics["chapters_count"] = len(chapters)
                metrics["files_exported"] += 1
                logger.info(f"Exported {len(chapters)} chapters")
        
        # Export keyframes metadata and images
        if include_keyframes:
            keyframes_data = (
                context.data_store.get(keyframes_step, "keyframes_per_chunk")
                or []
            )
            if keyframes_data:
                # Create keyframes subdirectory
                keyframes_dir = os.path.join(export_dir, "keyframes")
                os.makedirs(keyframes_dir, exist_ok=True)
                
                # Export keyframes metadata
                keyframes_metadata = []
                total_keyframes = 0
                images_exported = 0
                
                for chunk_data in keyframes_data:
                    chunk_keyframes = chunk_data.get("keyframes", [])
                    chunk_metadata = {
                        "chunk_index": chunk_data.get("chunk_index"),
                        "chunk_id": chunk_data.get("chunk_id"),
                        "boundaries": chunk_data.get("boundaries", []),
                        "keyframes": [],
                    }
                    
                    for kf in chunk_keyframes:
                        # Copy keyframe image to export directory
                        src_path = kf.get("filepath")
                        if src_path and os.path.exists(src_path):
                            filename = os.path.basename(src_path)
                            dst_path = os.path.join(keyframes_dir, filename)
                            try:
                                shutil.copy2(src_path, dst_path)
                                images_exported += 1
                                # Update filepath in metadata to relative path
                                kf_export = {**kf, "filepath": f"keyframes/{filename}"}
                            except Exception as e:
                                logger.warning(f"Failed to copy keyframe {src_path}: {e}")
                                kf_export = {**kf}
                        else:
                            kf_export = {**kf}
                        
                        chunk_metadata["keyframes"].append(kf_export)
                        total_keyframes += 1
                    
                    keyframes_metadata.append(chunk_metadata)
                
                # Export keyframes metadata JSON
                filepath = self._export_json(export_dir, "keyframes.json", keyframes_metadata, indent)
                exported_files.append(filepath)
                metrics["keyframes_count"] = total_keyframes
                metrics["keyframe_images_exported"] = images_exported
                metrics["files_exported"] += 1
                logger.info(f"Exported {total_keyframes} keyframes metadata, {images_exported} images")
        
        # Export chapter groups
        if include_chapter_groups:
            chapter_groups = context.data_store.get(chapter_grouping_step, "chapter_groups") or []
            if chapter_groups:
                filepath = self._export_json(export_dir, "chapter_groups.json", chapter_groups, indent)
                exported_files.append(filepath)
                metrics["chapter_groups_count"] = len(chapter_groups)
                metrics["files_exported"] += 1
                logger.info(f"Exported {len(chapter_groups)} chapter groups")
        
        # Get graph provider
        graph_provider = context.data_store.get(graph_step, "graph_provider")
        
        if graph_provider and hasattr(graph_provider, '_graph') and graph_provider._graph:
            nx_graph = graph_provider._graph
            
            # Extract and export chapter groups from graph nodes
            chapter_groups_from_graph = self._extract_nodes_by_type(nx_graph, "ChapterGroup")
            if chapter_groups_from_graph:
                filepath = self._export_json(export_dir, "graph_chapter_groups.json", chapter_groups_from_graph, indent)
                exported_files.append(filepath)
                metrics["graph_chapter_groups_count"] = len(chapter_groups_from_graph)
                metrics["files_exported"] += 1
                logger.info(f"Exported {len(chapter_groups_from_graph)} chapter groups from graph")
            
            # Extract and export chapters from graph nodes
            chapters_from_graph = self._extract_nodes_by_type(nx_graph, "Chapter")
            if chapters_from_graph:
                filepath = self._export_json(export_dir, "graph_chapters.json", chapters_from_graph, indent)
                exported_files.append(filepath)
                metrics["graph_chapters_count"] = len(chapters_from_graph)
                metrics["files_exported"] += 1
                logger.info(f"Exported {len(chapters_from_graph)} chapters from graph")
            
            # Extract and export events from graph nodes
            if include_events:
                events = self._extract_nodes_by_type(nx_graph, "Event")
                if events:
                    filepath = self._export_json(export_dir, "events.json", events, indent)
                    exported_files.append(filepath)
                    metrics["events_count"] = len(events)
                    metrics["files_exported"] += 1
                    logger.info(f"Exported {len(events)} events")
            
            # Extract and export objects from graph nodes
            if include_objects:
                objects = self._extract_nodes_by_type(nx_graph, "Object")
                if objects:
                    filepath = self._export_json(export_dir, "objects.json", objects, indent)
                    exported_files.append(filepath)
                    metrics["objects_count"] = len(objects)
                    metrics["files_exported"] += 1
                    logger.info(f"Exported {len(objects)} objects")
            
            # Extract and export transcript nodes from graph
            include_transcript_nodes = self.get_param("include_transcript_nodes", context, default=True)
            if include_transcript_nodes:
                transcripts = self._extract_nodes_by_type(nx_graph, "Transcript")
                if transcripts:
                    filepath = self._export_json(export_dir, "transcript_nodes.json", transcripts, indent)
                    exported_files.append(filepath)
                    metrics["transcript_nodes_count"] = len(transcripts)
                    metrics["files_exported"] += 1
                    logger.info(f"Exported {len(transcripts)} transcript nodes")
            
            # Export full graph as JSON
            if include_graph_json:
                graph_json = self._extract_graph_json(nx_graph)
                filepath = self._export_json(export_dir, "graph.json", graph_json, indent)
                exported_files.append(filepath)
                metrics["graph_nodes"] = len(graph_json.get("nodes", []))
                metrics["graph_edges"] = len(graph_json.get("edges", []))
                metrics["files_exported"] += 1
                logger.info(f"Exported graph: {metrics['graph_nodes']} nodes, {metrics['graph_edges']} edges")
            
            # Export interactive HTML visualization
            if include_graph_html:
                html_path = os.path.join(export_dir, "graph.html")
                if _generate_graph_html(graph_provider, html_path, video_id):
                    exported_files.append(html_path)
                    metrics["files_exported"] += 1
                    logger.info(f"Exported graph visualization: {html_path}")
        else:
            logger.warning("No graph provider found, skipping graph exports")
        
        # Export metadata
        metadata = {
            "run_id": run_id,
            "video_id": video_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "files": [os.path.basename(f) for f in exported_files],
            "metrics": metrics,
        }
        self._export_json(export_dir, "metadata.json", metadata, indent)
        
        logger.info(f"Export complete: {metrics['files_exported']} files to {export_dir}")
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "run_id": run_id,
                "export_dir": export_dir,
                "exported_files": exported_files,
                "graph_html_path": os.path.join(export_dir, "graph.html") if include_graph_html else None,
                "keyframes_dir": os.path.join(export_dir, "keyframes") if include_keyframes else None,
            },
            metrics=metrics,
            artifacts=exported_files,
        )
    
    def _export_json(
        self,
        directory: str,
        filename: str,
        data: Any,
        indent: Optional[int] = 2,
    ) -> str:
        """Export data to JSON file."""
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return filepath
    
    def _extract_nodes_by_type(self, nx_graph: Any, node_type: str) -> List[Dict[str, Any]]:
        """Extract nodes of a specific type from the graph."""
        nodes = []
        for node_id, attrs in nx_graph.nodes(data=True):
            if attrs.get("_type") == node_type:
                props = dict(attrs)
                props.pop("_type", None)
                nodes.append({"id": node_id, **props})
        return nodes
    
    def _extract_graph_json(self, nx_graph: Any) -> Dict[str, Any]:
        """Extract full graph as JSON structure."""
        nodes = []
        for node_id, attrs in nx_graph.nodes(data=True):
            props = dict(attrs)
            node_type = props.pop("_type", "Node")
            nodes.append({"id": node_id, "type": node_type, "properties": props})
        
        edges = []
        for source, target, attrs in nx_graph.edges(data=True):
            props = dict(attrs)
            edge_type = props.pop("_type", "RELATED")
            edges.append({"source": source, "target": target, "type": edge_type, "properties": props})
        
        return {"nodes": nodes, "edges": edges}


