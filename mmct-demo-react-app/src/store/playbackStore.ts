import { create } from 'zustand';
import { AgentEvent } from '../types';
import { TOOL_NAME_MAP, AGENT_NAME_MAP, TRANSFER_TOOLS, NODES } from '../config/agents';

// Helper to map any name to our node ID
function mapToNodeId(name: string): string {
  return AGENT_NAME_MAP[name] || TOOL_NAME_MAP[name] || name;
}

// Check if a tool is a transfer/handoff tool
function isTransferTool(toolName: string): boolean {
  return TRANSFER_TOOLS.has(toolName) || toolName.startsWith('transfer_to');
}

interface PlaybackStore {
  // All recorded events
  events: AgentEvent[];
  // Current playback position
  currentIndex: number;
  // Playback state
  isPlaying: boolean;
  isPaused: boolean;
  isComplete: boolean;
  // Currently active nodes (for visualization)
  activeNodes: Set<string>;
  // Tools that have been called (for subtle highlight)
  calledTools: Set<string>;
  // Active connections (tool calls - solid lines)
  activeConnections: Array<{ from: string; to: string }>;
  // Handoff connections (dotted lines)
  handoffConnections: Array<{ from: string; to: string }>;
  // All connections that have been shown (for persistence)
  shownConnections: Set<string>;
  // Current thought/message bubble
  currentMessage: { source: string; content: string } | null;
  // Actions
  addEvent: (event: AgentEvent) => void;
  loadEvents: (events: AgentEvent[]) => void;
  setCurrentIndex: (index: number) => void;
  setIsPlaying: (playing: boolean) => void;
  pause: () => void;
  resume: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  replay: () => void;
  reset: () => void;
  processEventAtIndex: (index: number) => void;
}

export const usePlaybackStore = create<PlaybackStore>((set, get) => ({
  events: [],
  currentIndex: -1,
  isPlaying: false,
  isPaused: false,
  isComplete: false,
  activeNodes: new Set<string>(),
  calledTools: new Set<string>(),
  handoffConnections: [],
  activeConnections: [],
  shownConnections: new Set<string>(),
  currentMessage: null,

  addEvent: (event) => {
    const { events, isPlaying, isPaused } = get();
    const newEvents = [...events, event];
    const newIndex = newEvents.length - 1;
    
    set({ events: newEvents });
    
    // If playing and not paused, process the new event immediately
    if (isPlaying && !isPaused) {
      get().processEventAtIndex(newIndex);
      set({ currentIndex: newIndex });
    }
    
    // Check for result event - keep the planner->user line visible for 2 seconds
    if (event.type === 'result') {
      setTimeout(() => {
        // Clear active connections but keep shown connections
        set({ 
          activeNodes: new Set<string>(),
          activeConnections: [],
          handoffConnections: [],
          currentMessage: { source: 'System', content: 'Query complete' }
        });
      }, 2000);
    }
    
    // Check for completion
    if (event.type === 'complete') {
      set({ isComplete: true, isPlaying: false });
    }
  },

  loadEvents: (newEvents) => {
    // Load events from history for replay
    set({
      events: newEvents,
      currentIndex: 0,
      isPlaying: true,
      isPaused: false,
      isComplete: false,
      activeNodes: new Set<string>(),
      calledTools: new Set<string>(),
      activeConnections: [],
      handoffConnections: [],
      shownConnections: new Set<string>(),
      currentMessage: null,
    });
    // Process first event
    if (newEvents.length > 0) {
      get().processEventAtIndex(0);
    }
  },

  setCurrentIndex: (index) => {
    set({ currentIndex: index });
    get().processEventAtIndex(index);
  },

  setIsPlaying: (playing) => set({ isPlaying: playing, isPaused: false }),

  pause: () => set({ isPaused: true }),

  resume: () => set({ isPaused: false }),

  stepForward: () => {
    const { currentIndex, events } = get();
    if (currentIndex < events.length - 1) {
      const newIndex = currentIndex + 1;
      set({ currentIndex: newIndex });
      get().processEventAtIndex(newIndex);
    }
  },

  stepBackward: () => {
    const { currentIndex } = get();
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1;
      set({ currentIndex: newIndex });
      get().processEventAtIndex(newIndex);
    }
  },

  replay: () => {
    const { events } = get();
    if (events.length > 0) {
      set({
        currentIndex: 0,
        isPlaying: true,
        isPaused: false,
        isComplete: false,
        activeNodes: new Set<string>(),
        calledTools: new Set<string>(),
        activeConnections: [],
        handoffConnections: [],
        shownConnections: new Set<string>(),
        currentMessage: null,
      });
      get().processEventAtIndex(0);
    }
  },

  reset: () => set({
    events: [],
    currentIndex: -1,
    isPlaying: false,
    isPaused: false,
    isComplete: false,
    activeNodes: new Set<string>(),
    calledTools: new Set<string>(),
    activeConnections: [],
    handoffConnections: [],
    shownConnections: new Set<string>(),
    currentMessage: null,
  }),

  processEventAtIndex: (index) => {
    const { events } = get();
    if (index < 0 || index >= events.length) return;

    const event = events[index];
    const activeNodes = new Set<string>();
    const activeConnections: Array<{ from: string; to: string }> = [];
    const handoffConnections: Array<{ from: string; to: string }> = [];
    let currentMessage: { source: string; content: string } | null = null;

    switch (event.type) {
      case 'connected':
        activeNodes.add('user');
        currentMessage = { source: 'System', content: 'Query received' };
        break;

      case 'message': {
        const mappedSource = mapToNodeId(event.source);
        activeNodes.add(mappedSource);
        if (event.content) {
          currentMessage = { 
            source: event.source, 
            content: event.content.slice(0, 150) + (event.content.length > 150 ? '...' : '')
          };
        }
        break;
      }

      case 'handoff': {
        const mappedFrom = mapToNodeId(event.source);
        const mappedTo = mapToNodeId(event.target);
        activeNodes.add(mappedFrom);
        activeNodes.add(mappedTo);
        // Use handoffConnections for dotted lines
        handoffConnections.push({ from: mappedFrom, to: mappedTo });
        currentMessage = { 
          source: event.source, 
          content: `Handing off to ${event.target}` 
        };
        break;
      }

      case 'tool_call': {
        const mappedSource = mapToNodeId(event.source);
        activeNodes.add(mappedSource);
        
        // Filter out transfer tools and only show actual visualization tools
        const visualizationTools = event.tool_names.filter(name => !isTransferTool(name));
        
        // Add tools to calledTools set (they get subtle highlight, no connection lines)
        visualizationTools.forEach(toolName => {
          const mappedToolName = mapToNodeId(toolName);
          activeNodes.add(mappedToolName);
          // Don't add connections to tools - they just get highlighted
        });
        
        // Show message based on what tools are being called
        if (visualizationTools.length > 0) {
          currentMessage = { 
            source: event.source, 
            content: `Calling: ${visualizationTools.join(', ')}` 
          };
        } else {
          // If only transfer tools, show transfer message
          currentMessage = {
            source: event.source,
            content: `Processing...`
          };
        }
        break;
      }

      case 'tool_result': {
        const mappedSource = mapToNodeId(event.source);
        activeNodes.add(mappedSource);
        currentMessage = { 
          source: event.source, 
          content: 'Processing tool results...' 
        };
        break;
      }

      case 'result':
        // Show connection from planner back to user for final response
        activeNodes.add('planner');
        activeNodes.add('user');
        handoffConnections.push({ from: 'planner', to: 'user' });
        currentMessage = { 
          source: 'System', 
          content: 'Response generated!' 
        };
        break;

      case 'complete':
        currentMessage = { 
          source: 'System', 
          content: 'Query complete' 
        };
        break;

      case 'error':
        currentMessage = { 
          source: 'Error', 
          content: event.message 
        };
        break;
    }

    // Add all current connections to shownConnections for persistence
    const { shownConnections, calledTools } = get();
    const newShownConnections = new Set(shownConnections);
    const newCalledTools = new Set(calledTools);
    
    // Track which tools were called (from activeNodes that are tools)
    activeNodes.forEach(nodeId => {
      const node = NODES[nodeId];
      if (node && node.type === 'tool') {
        newCalledTools.add(nodeId);
      }
    });
    
    activeConnections.forEach(conn => newShownConnections.add(`${conn.from}->${conn.to}`));
    handoffConnections.forEach(conn => newShownConnections.add(`${conn.from}->${conn.to}`));

    set({ activeNodes, activeConnections, handoffConnections, shownConnections: newShownConnections, calledTools: newCalledTools, currentMessage });
  },
}));
