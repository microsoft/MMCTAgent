import React from 'react';
import { POSITIONS, NODES } from '../../config/agents';
import { usePlaybackStore } from '../../store/playbackStore';

// Calculate curved path between two points with custom control point option
const getCurvedPath = (
  x1: number, y1: number, 
  x2: number, y2: number,
  customCtrl?: { x: number; y: number }
): string => {
  if (customCtrl) {
    return `M ${x1} ${y1} Q ${customCtrl.x} ${customCtrl.y} ${x2} ${y2}`;
  }
  
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  
  // Calculate perpendicular offset for curve control point
  const dx = x2 - x1;
  const dy = y2 - y1;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  // Curve intensity based on distance (more curve for longer lines)
  const curveOffset = Math.min(distance * 0.15, 8);
  
  // Perpendicular direction
  const perpX = -dy / distance;
  const perpY = dx / distance;
  
  // Control point offset from midpoint
  const ctrlX = midX + perpX * curveOffset;
  const ctrlY = midY + perpY * curveOffset;
  
  return `M ${x1} ${y1} Q ${ctrlX} ${ctrlY} ${x2} ${y2}`;
};

// Custom control points for specific connections to avoid overlapping nodes
const CUSTOM_CURVES: Record<string, { x: number; y: number }> = {
  // Planner to VideoAgent: curve goes between Summary and Search tools
  'planner->VideoAgent': { x: 28, y: 38 },
  'VideoAgent->planner': { x: 28, y: 38 },
  // Planner to ImageAgent: curve goes between ViT and Recognize tools  
  'planner->ImageAgent': { x: 72, y: 38 },
  'ImageAgent->planner': { x: 72, y: 38 },
  // Planner to Critic
  'Critic->planner': { x: 62, y: 8 },
  'planner->Critic': { x: 62, y: 8 },
};

export const DynamicConnections: React.FC = () => {
  const activeConnections = usePlaybackStore((state) => state.activeConnections);
  const handoffConnections = usePlaybackStore((state) => state.handoffConnections);
  const shownConnections = usePlaybackStore((state) => state.shownConnections);
  
  // Normalize key to ensure A->B and B->A use the same line
  const normalizeKey = (from: string, to: string): string => {
    return from < to ? `${from}->${to}` : `${to}->${from}`;
  };
  
  // Get the actual direction of active connection
  const getActiveDirection = (from: string, to: string): 'forward' | 'backward' | null => {
    // Check forward direction (from -> to)
    if (activeConnections.some(c => c.from === from && c.to === to) ||
        handoffConnections.some(c => c.from === from && c.to === to)) {
      return 'forward';
    }
    // Check backward direction (to -> from)
    if (activeConnections.some(c => c.from === to && c.to === from) ||
        handoffConnections.some(c => c.from === to && c.to === from)) {
      return 'backward';
    }
    return null;
  };
  
  // Create a set of active connection keys for quick lookup (normalized)
  const activeKeys = new Set([
    ...activeConnections.map(c => normalizeKey(c.from, c.to)),
    ...handoffConnections.map(c => normalizeKey(c.from, c.to)),
  ]);
  
  // Get the color for an active connection based on the source node
  const getActiveColor = (from: string, to: string): string => {
    const direction = getActiveDirection(from, to);
    if (direction === 'forward') return NODES[from]?.color || '#94a3b8';
    if (direction === 'backward') return NODES[to]?.color || '#94a3b8';
    return '#94a3b8';
  };
  
  // Parse shown connections and deduplicate (A->B and B->A become one line)
  // Only show connections between agents (not tools)
  const seenPairs = new Set<string>();
  const uniqueConnections: Array<{ from: string; to: string; normalizedKey: string }> = [];
  
  Array.from(shownConnections).forEach(key => {
    const [from, to] = key.split('->');
    
    // Skip connections involving tools (only show agent-to-agent)
    const fromNode = NODES[from];
    const toNode = NODES[to];
    if (!fromNode || !toNode) return;
    if (fromNode.type === 'tool' || toNode.type === 'tool') return;
    
    const normalized = normalizeKey(from, to);
    
    if (!seenPairs.has(normalized)) {
      seenPairs.add(normalized);
      // Always use consistent ordering for the path
      const [first, second] = from < to ? [from, to] : [to, from];
      uniqueConnections.push({ from: first, to: second, normalizedKey: normalized });
    }
  });
  
  if (uniqueConnections.length === 0) return null;
  
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 5 }}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {uniqueConnections.map(({ from, to, normalizedKey }) => {
        const fromPos = POSITIONS[from];
        const toPos = POSITIONS[to];
        
        if (!fromPos || !toPos) return null;
        
        const isActive = activeKeys.has(normalizedKey);
        const color = isActive ? getActiveColor(from, to) : '#94a3b8';
        const direction = getActiveDirection(from, to);
        
        // Check for custom curve control point
        const customCtrl = CUSTOM_CURVES[`${from}->${to}`] || CUSTOM_CURVES[`${to}->${from}`];
        
        // Generate curved path - direction determines start/end points
        // When backward, we need to draw from 'to' to 'from' so animation flows correctly
        let pathD: string;
        if (direction === 'backward') {
          pathD = getCurvedPath(toPos.x, toPos.y, fromPos.x, fromPos.y, customCtrl);
        } else {
          pathD = getCurvedPath(fromPos.x, fromPos.y, toPos.x, toPos.y, customCtrl);
        }
        
        return (
          <path
            key={normalizedKey}
            d={pathD}
            fill="none"
            stroke={color}
            strokeWidth={isActive ? 0.4 : 0.2}
            strokeDasharray={isActive ? '1 0.6' : 'none'}
            strokeOpacity={isActive ? 1 : 0.5}
            className={isActive ? 'animate-dash-flow' : ''}
            style={{
              strokeLinecap: 'round',
              filter: isActive ? `drop-shadow(0 0 0.5px ${color}80)` : undefined,
              transition: 'stroke 0.3s, stroke-width 0.3s, stroke-opacity 0.3s',
            }}
          />
        );
      })}
    </svg>
  );
};
