import React from 'react';
import { NODES } from '../../config/agents';
import { AgentNode } from './AgentNode';
import { DynamicConnections } from './DynamicConnections';
import { ThoughtBubble } from './ThoughtBubble';

export const AgentCanvas: React.FC = () => {
  const nodeIds = Object.keys(NODES);
  
  return (
    <div className="relative w-full h-full bg-transparent rounded-2xl overflow-hidden">
      {/* Dot grid background - subtle on light */}
      <div 
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `radial-gradient(circle, #94a3b8 1px, transparent 1px)`,
          backgroundSize: '24px 24px',
        }}
      />
      
      {/* Thought bubble */}
      <ThoughtBubble />
      
      {/* Dynamic connection lines - only show active connections */}
      <DynamicConnections />
      
      {/* Agent and tool nodes */}
      {nodeIds.map((nodeId) => (
        <AgentNode key={nodeId} nodeId={nodeId} />
      ))}
    </div>
  );
};
