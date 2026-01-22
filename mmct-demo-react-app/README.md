# MMCT Agent - React Visualization App

A React + TypeScript frontend for the MMCT (Multi-Modal Content Transformer) V2 API, featuring real-time agent flow visualization during query processing.

## Tech Stack

- **Vite 5.0** - Build tool with dev server proxy to `localhost:8000`
- **React 18.2** + **TypeScript 5.3**
- **Tailwind CSS 3.4** - Light cream/beige theme
- **Zustand 4.5** - State management
- **Lucide React** - Icons
- **React Markdown** - Response rendering

## Architecture Overview

```
src/
├── App.tsx                    # Main app with chat + visualizer overlay
├── components/
│   ├── chat/
│   │   └── ChatWindow.tsx     # Main chat interface (light theme)
│   ├── visualization/
│   │   ├── AgentCanvas.tsx    # Container for nodes + connections
│   │   ├── AgentNode.tsx      # Individual node rendering (agents/tools)
│   │   ├── DynamicConnections.tsx  # Curved SVG lines between agents
│   │   └── ThoughtBubble.tsx  # Current message display (top-left)
│   └── visualizer/
│       └── VisualizerOverlay.tsx   # Full-screen overlay with controls
├── config/
│   └── agents.ts              # Node configs, positions, mappings
├── hooks/
│   └── useV2Query.ts          # SSE streaming hook + history
├── store/
│   ├── playbackStore.ts       # Visualization state (events, nodes, connections)
│   └── queryStore.ts          # Query state (loading, result, error)
├── styles/
│   └── globals.css            # Tailwind + custom animations
└── types/
    └── index.ts               # TypeScript interfaces
```

## Key Features

### 1. Light Theme
- **Background**: Warm cream gradient (`from-amber-50 via-orange-50 to-rose-50`)
- **Chat**: White cards with slate borders
- **Visualizer**: Same light gradient with subtle dot grid

### 2. Agent Flow Visualization

**Node Layout** (positions in `config/agents.ts`):
```
Top Row (y: 15%):     User(25%) → Planner(50%) → Critic(75%)

Left Side (y: 55%):   VideoAgent(28%) with 4 tools around it
                      - Summary(16,38), Search(40,38)
                      - Frames(16,72), Objects(40,72)

Right Side (y: 55%):  ImageAgent(72%) with 4 tools around it
                      - ViT(60,38), Recognize(84,38)
                      - OCR(60,72), Objects(84,72)
```

**Connection Lines**:
- Only between agents (no lines to tools)
- Curved using quadratic Bezier paths
- Custom control points for Planner↔VideoAgent/ImageAgent (passes between tool nodes)
- Direction-aware animation (flows from source to target)
- Persist as grey lines after shown, colored when active

**Node States**:
- **Active**: Large glow, ping animation, 1.25x scale, thick border
- **Called Tool**: Subtle glow, 1.05x scale, persists after tool was used
- **Default**: Normal appearance with hover effects

### 3. SSE Streaming

**Endpoint**: `POST /v2/query/stream`
- Uses `URLSearchParams` (not FormData) for FastAPI compatibility
- Parses `event: type\ndata: json\n\n` format

**Event Types**:
- `connected` - Query received
- `message` - Agent thinking/processing
- `handoff` - Agent delegation (dotted line)
- `tool_call` - Tool invocation (highlights tool)
- `tool_result` - Tool response
- `result` - Final answer (shows Planner→User line for 2s)
- `complete` - Query finished
- `error` - Error occurred

### 4. State Management

**playbackStore.ts**:
```typescript
{
  events: AgentEvent[]           // All SSE events
  currentIndex: number           // Playback position
  isPlaying: boolean
  isPaused: boolean
  activeNodes: Set<string>       // Currently highlighted nodes
  calledTools: Set<string>       // Tools that were invoked (subtle highlight)
  activeConnections: []          // Tool call connections (not used for lines now)
  handoffConnections: []         // Agent handoffs (dotted lines)
  shownConnections: Set<string>  // All connections ever shown (persistent grey)
  currentMessage: {source, content}  // ThoughtBubble content
}
```

**queryStore.ts**:
```typescript
{
  isLoading: boolean
  result: { response: string, sources: [...] }
  error: string | null
}
```

### 5. Name Mappings

The API returns different names than our node IDs. Mappings in `config/agents.ts`:

```typescript
AGENT_NAME_MAP = {
  'videoagent': 'VideoAgent',
  'imageagent': 'ImageAgent',
  'critic': 'Critic',
  // ... case variations
}

TOOL_NAME_MAP = {
  'get_context': 'search_video_context',
  'get_object_collection': 'detect_video_objects',
  // ... API name → node ID
}

TRANSFER_TOOLS = Set(['transfer_to_video_agent', ...])  // Filtered out
```

## Custom Animations (globals.css)

```css
@keyframes dash-flow    /* Animated dashed lines */
@keyframes pulse-glow   /* Node glow pulsing */
@keyframes fade-in-up   /* Entry animation */
@keyframes slide-in-right  /* History panel */
```

## Playback Controls

- **Replay** - Restart from beginning
- **Step Back/Forward** - Navigate events
- **Play/Pause** - Auto-advance through events
- **History Panel** - Saved queries with replay

## API Integration

**Proxy Config** (vite.config.ts):
```typescript
proxy: {
  '/v2': {
    target: 'http://localhost:8000',
    changeOrigin: true
  }
}
```

**Query Flow**:
1. User submits query in ChatWindow
2. `useV2Query.streamQuery()` POSTs to `/v2/query/stream`
3. SSE events arrive, parsed and added to playbackStore
4. `processEventAtIndex()` updates activeNodes, connections, message
5. Components react to state changes via Zustand selectors

## File Highlights

### agents.ts
- `NODES`: Node configs (id, label, icon, color, type: 'agent'|'tool')
- `POSITIONS`: x,y percentages for each node
- `CUSTOM_CURVES`: Control points for specific curved paths
- Name mappings for API compatibility

### DynamicConnections.tsx
- Filters to agent-only connections
- Normalizes bidirectional connections (A→B and B→A = same line)
- Determines animation direction based on active connection source
- Custom Bezier curves for specific paths

### AgentNode.tsx
- Three visual states: active, wasCalled (tools), default
- Responsive sizing: agents=80px circles, tools=56px rounded squares
- Glow effects and scaling based on state

### VisualizerOverlay.tsx
- Full-screen overlay with light theme
- History panel (right side drawer)
- Centered playback controls
- Event counter in header

## Running the App

```bash
cd mmct-demo-react-app
npm install
npm run dev  # Starts on port 3000 (or 3001 if busy)
```

Requires backend API running on `localhost:8000`.

## Recent Changes Summary

1. **Light Theme**: Converted from dark slate to cream/beige gradient
2. **Curved Lines**: Bezier curves instead of straight lines
3. **Agent-Only Lines**: Removed tool connection lines
4. **Called Tools Highlight**: Subtle persistent glow for used tools
5. **Direction-Aware Animation**: Dotted line flows from source to target
6. **Custom Path Routing**: Lines between Planner↔Agents go through tool gaps
