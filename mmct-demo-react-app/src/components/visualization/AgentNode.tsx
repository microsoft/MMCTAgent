import React from 'react';
import { NODES, POSITIONS } from '../../config/agents';
import { usePlaybackStore } from '../../store/playbackStore';

interface AgentNodeProps {
  nodeId: string;
}

export const AgentNode: React.FC<AgentNodeProps> = ({ nodeId }) => {
  const node = NODES[nodeId];
  const position = POSITIONS[nodeId];
  const activeNodes = usePlaybackStore((state) => state.activeNodes);
  const calledTools = usePlaybackStore((state) => state.calledTools);
  
  if (!node || !position) return null;
  
  // Check for active state with case-insensitive matching
  const isActive = activeNodes.has(nodeId) || 
                   activeNodes.has(nodeId.toLowerCase()) || 
                   activeNodes.has(nodeId.toUpperCase());
  
  // Check if this tool was called (subtle highlight)
  const wasCalled = !isActive && node.type === 'tool' && calledTools.has(nodeId);
  
  const Icon = node.icon;
  const isAgent = node.type === 'agent';
  
  // Size based on type
  const size = isAgent ? 'w-20 h-20' : 'w-14 h-14';
  const iconSize = isAgent ? 28 : 20;
  
  return (
    <div
      className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300"
      style={{
        left: `${position.x}%`,
        top: `${position.y}%`,
      }}
    >
      {/* Glow ring behind node when active */}
      {isActive && (
        <div
          className="absolute inset-0 rounded-full animate-ping"
          style={{
            backgroundColor: node.color,
            opacity: 0.3,
            transform: 'scale(1.5)',
          }}
        />
      )}
      
      {/* Outer glow ring for active nodes */}
      {isActive && (
        <div
          className={`absolute ${isAgent ? '-inset-3' : '-inset-2'} ${isAgent ? 'rounded-full' : 'rounded-2xl'}`}
          style={{
            background: `radial-gradient(circle, ${node.color}40 0%, transparent 70%)`,
            animation: 'pulse 2s infinite',
          }}
        />
      )}
      
      {/* Glow for called tools - brighter than subtle */}
      {wasCalled && (
        <div
          className="absolute -inset-2 rounded-xl"
          style={{
            background: `radial-gradient(circle, ${node.color}50 0%, ${node.color}20 50%, transparent 80%)`,
          }}
        />
      )}
      
      {/* Node circle/rounded rect */}
      <div
        className={`
          ${size} 
          ${isAgent ? 'rounded-full' : 'rounded-xl'}
          flex items-center justify-center
          border-2 relative
          transition-all duration-300 ease-out
          ${isActive 
            ? 'scale-125 shadow-2xl bg-white' 
            : wasCalled
              ? 'scale-105 shadow-md bg-white'
              : 'scale-100 shadow-sm hover:shadow-md bg-white/90'
          }
        `}
        style={{
          borderColor: node.color,
          borderWidth: isActive ? '3px' : wasCalled ? '2.5px' : '2px',
          boxShadow: isActive 
            ? `0 0 40px ${node.color}80, 0 0 80px ${node.color}40, 0 8px 32px rgba(0,0,0,0.3), inset 0 0 20px ${node.color}20` 
            : wasCalled
              ? `0 0 25px ${node.color}70, 0 0 50px ${node.color}40, 0 4px 16px rgba(0,0,0,0.15)`
              : undefined,
          outline: isActive ? `4px solid ${node.color}40` : wasCalled ? `2px solid ${node.color}30` : undefined,
          outlineOffset: isActive ? '4px' : wasCalled ? '2px' : undefined,
        }}
      >
        <Icon 
          size={iconSize} 
          style={{ color: node.color }}
          strokeWidth={isActive ? 2.5 : wasCalled ? 2.2 : 2}
        />
      </div>
      
      {/* Label */}
      <div className="absolute left-1/2 -translate-x-1/2 mt-2 whitespace-nowrap">
        <span 
          className={`
            text-xs font-semibold px-3 py-1 rounded-full
            transition-all duration-300
            ${isActive 
              ? 'bg-white text-slate-800 shadow-lg scale-110' 
              : wasCalled
                ? 'bg-white text-slate-700 shadow-md scale-105'
                : 'bg-white/80 text-slate-600 border border-slate-200'
            }
          `}
          style={{
            boxShadow: isActive ? `0 0 20px ${node.color}60` : wasCalled ? `0 0 18px ${node.color}50` : undefined,
          }}
        >
          {node.label}
        </span>
      </div>
    </div>
  );
};
