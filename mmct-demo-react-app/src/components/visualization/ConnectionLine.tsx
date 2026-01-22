import React from 'react';
import { POSITIONS, NODES } from '../../config/agents';
import { usePlaybackStore } from '../../store/playbackStore';

interface ConnectionLineProps {
  from: string;
  to: string;
  isHandoff?: boolean; // true = dotted line for handoffs
}

export const ConnectionLine: React.FC<ConnectionLineProps> = ({ from, to }) => {
  const activeConnections = usePlaybackStore((state) => state.activeConnections);
  const handoffConnections = usePlaybackStore((state) => state.handoffConnections);
  
  const fromPos = POSITIONS[from];
  const toPos = POSITIONS[to];
  const fromNode = NODES[from];
  
  if (!fromPos || !toPos) return null;
  
  // Check if this connection is active
  const isActiveToolCall = activeConnections.some(
    (conn) => conn.from === from && conn.to === to
  );
  const isActiveHandoff = handoffConnections.some(
    (conn) => conn.from === from && conn.to === to
  );
  const isActive = isActiveToolCall || isActiveHandoff;
  
  // Calculate the color based on the source node
  const color = isActive ? (fromNode?.color || '#94a3b8') : '#475569';
  
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
    >
      <defs>
        <marker
          id={`arrow-${from}-${to}`}
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path
            d="M0,0 L0,6 L6,3 z"
            fill={color}
            className="transition-colors duration-300"
          />
        </marker>
      </defs>
      
      <line
        x1={`${fromPos.x}%`}
        y1={`${fromPos.y}%`}
        x2={`${toPos.x}%`}
        y2={`${toPos.y}%`}
        stroke={color}
        strokeWidth={isActive ? 2.5 : 1.5}
        strokeDasharray={isActive ? (isActiveHandoff ? '8 6' : '6 4') : 'none'}
        strokeOpacity={isActive ? 1 : 0.4}
        markerEnd={`url(#arrow-${from}-${to})`}
        className={`transition-all duration-300 ${isActive ? 'animate-dash-flow' : ''}`}
        style={{
          strokeLinecap: 'round',
        }}
      />
    </svg>
  );
};
