import React, { useState, useEffect, useRef } from 'react';
import { 
  User, 
  BrainCircuit, 
  Video, 
  ShieldCheck, 
  Image as ImageIcon,
  Play, 
  Pause, 
  RotateCcw, 
  ChevronRight, 
  MessageSquare, 
  Cpu, 
  Search,
  FileText,
  ScanEye,
  Type,
  Frame,
  Database,
  Layers,
  BoxSelect,
  Eye,
  Wifi,
  WifiOff
} from 'lucide-react';

// --- CONFIGURATION ---
const POS = {
  // Main Agents
  User: { x: 50, y: 10 },
  Planner: { x: 50, y: 40 },
  Critic: { x: 50, y: 80 },
  VideoAgent: { x: 20, y: 50 },
  ImageAgent: { x: 80, y: 50 },

  // Video Tools (Clustered Left)
  video_summary: { x: 5, y: 40 },
  video_context: { x: 5, y: 50 },
  video_frames: { x: 5, y: 60 },
  video_objects: { x: 15, y: 70 },

  // Image Tools (Clustered Right)
  img_vit: { x: 95, y: 40 },
  img_recog: { x: 95, y: 50 },
  img_obj: { x: 95, y: 60 },
  img_ocr: { x: 85, y: 70 },
};

const NODES = {
  // Agents
  User: { label: "User", role: "End User", icon: User, color: "#64748b", type: 'agent' },
  Planner: { label: "Planner", role: "Orchestrator", icon: BrainCircuit, color: "#3b82f6", type: 'agent' },
  VideoAgent: { label: "VideoAgent", role: "Visual Tools", icon: Video, color: "#f97316", type: 'agent' },
  ImageAgent: { label: "ImageAgent", role: "Vision Tools", icon: ImageIcon, color: "#8b5cf6", type: 'agent' },
  Critic: { label: "Critic", role: "Reviewer", icon: ShieldCheck, color: "#22c55e", type: 'agent' },

  // Video Tools
  video_summary: { label: "GetSummary", role: "Summarization", icon: FileText, color: "#fdba74", type: 'tool', parent: 'VideoAgent' },
  video_context: { label: "GetContext", role: "Context Retr.", icon: Database, color: "#fdba74", type: 'tool', parent: 'VideoAgent' },
  video_frames: { label: "GetFrames", role: "Frame Extr.", icon: Layers, color: "#fdba74", type: 'tool', parent: 'VideoAgent' },
  video_objects: { label: "GetObjects", role: "Obj Collection", icon: BoxSelect, color: "#fdba74", type: 'tool', parent: 'VideoAgent' },

  // Image Tools
  img_vit: { label: "ViT", role: "Transformer", icon: ScanEye, color: "#c4b5fd", type: 'tool', parent: 'ImageAgent' },
  img_recog: { label: "Recog", role: "Recognition", icon: Eye, color: "#c4b5fd", type: 'tool', parent: 'ImageAgent' },
  img_obj: { label: "ObjDetect", role: "Detection", icon: Frame, color: "#c4b5fd", type: 'tool', parent: 'ImageAgent' },
  img_ocr: { label: "OCR", role: "Text Extr.", icon: Type, color: "#c4b5fd", type: 'tool', parent: 'ImageAgent' },
};

// --- COMPONENTS ---

const ConnectionLine = ({ start, end, isActive, color, isDashed }) => {
  return (
    <svg className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-visible">
      <defs>
        <marker
          id={`arrow-${color.replace('#', '')}`}
          markerWidth="6"
          markerHeight="6"
          refX="5"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L6,3 z" fill={color} />
        </marker>
      </defs>
      
      {/* Base Line */}
      <line
        x1={`${start.x}%`}
        y1={`${start.y}%`}
        x2={`${end.x}%`}
        y2={`${end.y}%`}
        stroke="#e2e8f0"
        strokeWidth="1.5"
        strokeDasharray={isDashed ? "4,4" : "0"}
      />

      {/* Active Animated Line */}
      <line
        x1={`${start.x}%`}
        y1={`${start.y}%`}
        x2={`${end.x}%`}
        y2={`${end.y}%`}
        stroke={isActive ? color : "transparent"}
        strokeWidth={isActive ? "2.5" : "0"}
        strokeDasharray="8, 8"
        className={isActive ? "animate-dash-flow" : ""}
        markerEnd={isActive ? `url(#arrow-${color.replace('#', '')})` : undefined}
        style={{ opacity: isActive ? 1 : 0, transition: "opacity 0.2s ease" }}
      />
    </svg>
  );
};

const Node = ({ id, config, isActive, isSecondaryActive }) => {
  const pos = POS[id];
  const Icon = config.icon;
  const isTool = config.type === 'tool';

  // Determine size based on type
  const sizeClass = isTool ? "w-16 h-16 border-2" : "w-24 h-24 border-4";
  const iconSize = isTool ? 20 : 32;
  const textSize = isTool ? "text-[9px]" : "text-xs";
  const shapeClass = isTool ? "rounded-xl" : "rounded-full"; // Shape differentiation

  return (
    <div
      className="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-10 transition-all duration-300"
      style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
    >
      <div
        className={`
          relative ${sizeClass} ${shapeClass} bg-white shadow-lg flex flex-col items-center justify-center
          transition-all duration-300
          ${isActive ? `scale-110 shadow-lg` : 'scale-100'}
          ${isSecondaryActive && !isActive ? 'scale-105 ring-2 ring-offset-2 ring-slate-200' : ''}
        `}
        style={{ 
          borderColor: isActive ? config.color : isSecondaryActive ? config.color : '#f1f5f9',
          boxShadow: isActive ? `0 0 20px ${config.color}50` : undefined 
        }}
      >
        <Icon 
          size={iconSize} 
          className={`mb-1 transition-colors duration-300 ${isActive ? 'text-slate-800' : 'text-slate-400'}`} 
          style={{ color: isActive ? config.color : undefined }}
        />
        <span className={`${textSize} font-bold text-slate-700 text-center leading-tight px-1`}>{config.label}</span>
        {!isTool && (
          <span className="text-[9px] text-slate-400 uppercase tracking-wider">{config.role}</span>
        )}
      </div>
      
      {/* Floating Badge for Active Status */}
      {isActive && (
        <div 
          className="absolute -bottom-3 px-2 py-0.5 rounded-full text-[9px] font-bold text-white uppercase tracking-wider shadow-sm animate-fade-in-up"
          style={{ backgroundColor: config.color }}
        >
          Active
        </div>
      )}
    </div>
  );
};

// --- MAIN COMPONENT ---

export default function MMCTAgentViz() {
  // Use state for events so we can add them dynamically from WebSocket
  const [events, setEvents] = useState([
    // Optional: Keep initial demo data or start empty
    // { id: 0, time: "Ready", source: "User", target: "User", type: "info", content: "Waiting for simulation...", duration: 1000 }
  ]);
  
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [isConnected, setIsConnected] = useState(false);
  const [wsError, setWsError] = useState(null);
  
  const scrollRef = useRef(null);
  const wsRef = useRef(null);

  // --- WEBSOCKET LOGIC ---
  const runSimulation = () => {
    setWsError(null);
    setIsPlaying(false); // Pause auto-player, let WS drive
    
    // Connect to Python Backend
    try {
      const ws = new WebSocket("ws://localhost:8000/ws");
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setEvents([]); // Clear logs for fresh run
        setCurrentStep(0);
        // Trigger the agent on the backend with a query
        ws.send(JSON.stringify({ query: "Why do we decorate tree in christmas?" }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'complete') {
          ws.close();
          return;
        }

        // Add incoming event to state
        setEvents(prev => {
          const newEvent = {
            ...data,
            // Ensure duration exists for playback later
            duration: 2000 
          };
          const updatedEvents = [...prev, newEvent];
          // Auto-advance step to show latest action immediately during live stream
          setCurrentStep(updatedEvents.length - 1);
          return updatedEvents;
        });
      };

      ws.onerror = (err) => {
        setWsError("Connection failed. Is app.py running?");
        console.error("WS Error:", err);
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

    } catch (err) {
      setWsError("Failed to create WebSocket.");
    }
  };

  // --- AUTO-SCROLL ---
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events, currentStep]);

  // --- PLAYBACK LOGIC (For replaying history) ---
  useEffect(() => {
    let timer;
    // Only run timer if playing AND we are not currently receiving live data (optional preference)
    // Or allow playback of history even while connected, but usually we pause during live.
    if (isPlaying && currentStep < events.length - 1) {
      const duration = (events[currentStep]?.duration || 1000) / speed;
      timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, duration);
    } else if (currentStep >= events.length - 1) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, speed, events]);

  const currentEvent = events[currentStep];

  // Helper: Is connection active?
  const getLineStatus = (startId, endId) => {
    if (!currentEvent) return false;
    // Direct match
    if (currentEvent.source === startId && currentEvent.target === endId) return true;
    if (currentEvent.source === endId && currentEvent.target === startId) return true;
    return false;
  };

  // Helper: Is node active?
  const isNodeActive = (nodeId) => {
    if (!currentEvent) return false;
    // Primary active: the source or the target of the current action
    return currentEvent.source === nodeId || currentEvent.target === nodeId;
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 font-sans text-slate-900 overflow-hidden">
      <style>{`
        @keyframes dash-flow { to { stroke-dashoffset: -16; } }
        .animate-dash-flow { animation: dash-flow 0.5s linear infinite; }
        @keyframes fade-in-up { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in-up { animation: fade-in-up 0.3s ease-out forwards; }
      `}</style>

      {/* LEFT: Visualization Canvas */}
      <div className="flex-1 relative bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] [background-size:20px_20px] flex flex-col">
        
        {/* Header */}
        <div className="absolute top-6 left-6 z-20 pointer-events-none">
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Cpu className="text-blue-600" />
            MMCTAgent <span className="text-slate-400 font-light">Architecture</span>
          </h1>
          <p className="text-slate-500 text-sm mt-1">Multi-Modal Critical Thinking Agent Execution Flow</p>
        </div>

        {/* Legend */}
        <div className="absolute top-6 right-6 z-20 flex flex-col gap-2 items-end">
          <div className="flex gap-3 text-xs font-medium bg-white/80 backdrop-blur px-4 py-2 rounded-full shadow-sm border border-slate-200">
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500"></span> Planner</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-orange-500"></span> Video</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-violet-500"></span> Image</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-green-500"></span> Critic</div>
          </div>
          
          {/* Connection Status */}
          {isConnected ? (
             <div className="flex items-center gap-1.5 px-3 py-1 bg-green-100 text-green-700 rounded-full text-[10px] font-bold uppercase tracking-wide border border-green-200 shadow-sm animate-pulse">
               <Wifi size={12} /> Live Connected
             </div>
          ) : wsError ? (
             <div className="flex items-center gap-1.5 px-3 py-1 bg-red-100 text-red-700 rounded-full text-[10px] font-bold uppercase tracking-wide border border-red-200 shadow-sm">
               <WifiOff size={12} /> {wsError}
             </div>
          ) : null}
        </div>

        {/* Graph Layer */}
        <div className="relative w-full h-full mx-auto my-auto max-w-6xl">
          
          {/* 1. Static Connections (Agent <-> Tools) */}
          {/* Video Tools */}
          {['video_summary', 'video_context', 'video_frames', 'video_objects'].map(toolId => (
            <ConnectionLine 
              key={`conn-video-${toolId}`}
              start={POS.VideoAgent} 
              end={POS[toolId]} 
              isActive={getLineStatus('VideoAgent', toolId)}
              color={NODES.VideoAgent.color}
              isDashed={true}
            />
          ))}
          
          {/* Image Tools */}
          {['img_vit', 'img_recog', 'img_obj', 'img_ocr'].map(toolId => (
            <ConnectionLine 
              key={`conn-img-${toolId}`}
              start={POS.ImageAgent} 
              end={POS[toolId]} 
              isActive={getLineStatus('ImageAgent', toolId)} // Check just in case logic expands
              color={NODES.ImageAgent.color}
              isDashed={true}
            />
          ))}

          {/* 2. Dynamic Connections (Agent <-> Agent) */}
          <ConnectionLine start={POS.User} end={POS.Planner} isActive={getLineStatus('User', 'Planner')} color={NODES.Planner.color} />
          <ConnectionLine start={POS.Planner} end={POS.VideoAgent} isActive={getLineStatus('Planner', 'VideoAgent')} color={NODES.VideoAgent.color} />
          <ConnectionLine start={POS.Planner} end={POS.ImageAgent} isActive={getLineStatus('Planner', 'ImageAgent')} color={NODES.ImageAgent.color} />
          <ConnectionLine start={POS.Planner} end={POS.Critic} isActive={getLineStatus('Planner', 'Critic')} color={NODES.Critic.color} />

          {/* 3. Render Nodes */}
          {Object.entries(NODES).map(([id, config]) => (
            <Node 
              key={id}
              id={id}
              config={config}
              isActive={isNodeActive(id)}
              isSecondaryActive={config.parent && isNodeActive(config.parent)}
            />
          ))}

          {/* 4. Context Bubble Overlay */}
          {currentEvent && (['query', 'thought', 'final'].includes(currentEvent.type)) && (
            <div 
              className="absolute transform -translate-x-1/2 -translate-y-1/2 bg-white/95 backdrop-blur shadow-xl border border-slate-100 p-4 rounded-xl max-w-sm w-full z-30 transition-all duration-300 animate-fade-in-up"
              style={{ 
                left: '50%', 
                top: currentEvent.source === 'User' ? '25%' : '60%' 
              }}
            >
              <div className="flex items-start gap-3">
                <div className={`mt-1 p-1.5 rounded-md`} style={{ backgroundColor: `${NODES[currentEvent.source]?.color || '#cbd5e1'}20` }}>
                  {currentEvent.type === 'query' ? <MessageSquare size={14} className="text-slate-600"/> : <BrainCircuit size={14} className="text-slate-600"/>}
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-500 uppercase mb-1">
                    {currentEvent.type === 'query' ? 'Incoming Query' : `${currentEvent.source} Thinking`}
                  </div>
                  <div className="text-sm text-slate-700 leading-relaxed font-medium">
                    {currentEvent.content.length > 120 ? currentEvent.content.substring(0, 120) + '...' : currentEvent.content}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Empty State / Welcome */}
          {events.length === 0 && !isConnected && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center text-slate-400">
               <Cpu size={48} className="mx-auto mb-4 opacity-50" />
               <p className="text-lg font-medium">Ready to Visualize</p>
               <p className="text-sm">Connect backend or start simulation</p>
            </div>
          )}

        </div>

        {/* Playback Controls */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center gap-4 bg-white/90 backdrop-blur px-6 py-3 rounded-2xl shadow-xl border border-slate-100 z-40">
           
           {/* Live Button */}
           {!isConnected && (
             <button 
               onClick={runSimulation}
               className="px-4 py-2 bg-blue-600 text-white text-xs font-bold rounded-full hover:bg-blue-700 transition-colors flex items-center gap-2 shadow-blue-200 shadow-md"
             >
               <Cpu size={14} /> Run Live Agent
             </button>
           )}

           {/* Playback Controls (Visible if we have data) */}
           {events.length > 0 && (
             <>
                <div className="w-px h-6 bg-slate-200 mx-1"></div>
                <button onClick={() => { setCurrentStep(0); setIsPlaying(true); }} className="p-2 hover:bg-slate-100 rounded-full transition-colors"><RotateCcw size={20} className="text-slate-500" /></button>
                <button onClick={() => setIsPlaying(!isPlaying)} className={`w-12 h-12 flex items-center justify-center rounded-full transition-all ${isPlaying ? 'bg-amber-100 text-amber-600' : 'bg-slate-100 text-slate-600 hover:scale-105'}`}>
                  {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" className="ml-1"/>}
                </button>
                <button onClick={() => setCurrentStep(Math.min(events.length - 1, currentStep + 1))} className="p-2 hover:bg-slate-100 rounded-full transition-colors"><ChevronRight size={20} className="text-slate-500" /></button>
             </>
           )}
        </div>
      </div>

      {/* RIGHT: Live Logs */}
      <div className="w-[380px] border-l border-slate-200 bg-white flex flex-col shadow-2xl z-30">
        <div className="h-14 border-b border-slate-100 flex items-center px-6 bg-slate-50/50">
          <h2 className="font-semibold text-slate-700 text-sm">System Logs</h2>
          <div className="ml-auto text-xs font-mono text-slate-400">
             {events.length > 0 ? `${currentStep + 1} / ${events.length}` : "Waiting"}
          </div>
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3 scroll-smooth">
          {events.length === 0 ? (
            <div className="text-center mt-10 text-slate-400 text-xs italic">
              No logs generated yet.
            </div>
          ) : (
            events.map((event, index) => {
              if (index > currentStep) return null;
              const isTool = event.type.includes('tool');
              const color = NODES[event.source]?.color || NODES[event.target]?.color || '#94a3b8';
              const cleanSource = event.source.replace('video_', '').replace('img_', '');
              
              return (
                <div key={event.id || index} className={`p-3 rounded-lg text-xs border transition-all duration-300 ${index === currentStep ? 'bg-blue-50 border-blue-200 shadow-sm opacity-100' : 'bg-white border-slate-100 opacity-60'}`}>
                  <div className="flex justify-between items-center mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold px-1.5 py-0.5 rounded text-[10px] text-white" style={{ backgroundColor: color }}>
                        {cleanSource}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400">{event.time}</span>
                    </div>
                  </div>
                  <div className={`font-mono leading-relaxed whitespace-pre-wrap ${isTool ? 'text-orange-600' : 'text-slate-600'}`}>
                    {event.content}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}