import { useState, useEffect } from 'react';
import { History, Trash2, Play, ChevronDown, ChevronRight, X, Clock } from 'lucide-react';
import { getSavedQueries, clearSavedQueries } from '../../hooks/useV2Query';
import { usePlaybackStore } from '../../store/playbackStore';
import { useQueryStore } from '../../store/queryStore';
import { AgentEvent } from '../../types';

interface SavedQuery {
  id: string;
  query: string;
  videoId?: string;
  timestamp: string;
  events: AgentEvent[];
}

interface HistoryPaneProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HistoryPane: React.FC<HistoryPaneProps> = ({ isOpen, onClose }) => {
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  
  const { reset: resetPlayback, addEvent, setIsPlaying } = usePlaybackStore();
  const { setResult, setTokenUsage, reset: resetQuery } = useQueryStore();

  // Load saved queries on mount and when pane opens
  useEffect(() => {
    if (isOpen) {
      setSavedQueries(getSavedQueries());
    }
  }, [isOpen]);

  const handleClearHistory = () => {
    if (confirm('Clear all saved queries?')) {
      clearSavedQueries();
      setSavedQueries([]);
    }
  };

  const handleReplayQuery = (query: SavedQuery) => {
    // Reset current state
    resetPlayback();
    resetQuery();
    
    // Replay events one by one
    query.events.forEach((event, index) => {
      setTimeout(() => {
        addEvent(event);
        
        // Handle result event
        if (event.type === 'result') {
          const resultEvent = event as any;
          if (resultEvent.content) {
            setResult(resultEvent.content);
          }
          if (resultEvent.token_usage) {
            setTokenUsage(resultEvent.token_usage);
          }
        }
      }, index * 300); // 300ms delay between events for visualization
    });
    
    setIsPlaying(true);
    onClose();
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getEventSummary = (events: AgentEvent[]) => {
    const types = events.reduce((acc, e) => {
      acc[e.type] = (acc[e.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return Object.entries(types)
      .map(([type, count]) => `${count} ${type}`)
      .join(', ');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Pane */}
      <div className="relative w-full max-w-md bg-white shadow-2xl flex flex-col animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-100 flex items-center justify-center">
              <History size={18} className="text-blue-600" />
            </div>
            <div>
              <h2 className="font-semibold text-slate-800">Query History</h2>
              <p className="text-xs text-slate-500">{savedQueries.length} saved queries</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {savedQueries.length > 0 && (
              <button
                onClick={handleClearHistory}
                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                title="Clear history"
              >
                <Trash2 size={18} />
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Query list */}
        <div className="flex-1 overflow-y-auto">
          {savedQueries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <History size={40} strokeWidth={1.5} />
              <p className="mt-3 text-sm">No saved queries yet</p>
              <p className="text-xs">Queries will appear here after you ask them</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {savedQueries.map((query) => (
                <div key={query.id} className="bg-white hover:bg-slate-50/50 transition-colors">
                  {/* Query header */}
                  <div 
                    className="px-5 py-4 cursor-pointer"
                    onClick={() => setExpandedId(expandedId === query.id ? null : query.id)}
                  >
                    <div className="flex items-start gap-3">
                      <button className="mt-0.5 text-slate-400">
                        {expandedId === query.id ? (
                          <ChevronDown size={16} />
                        ) : (
                          <ChevronRight size={16} />
                        )}
                      </button>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-800 line-clamp-2">
                          {query.query}
                        </p>
                        <div className="flex items-center gap-3 mt-1.5 text-xs text-slate-500">
                          <span className="flex items-center gap-1">
                            <Clock size={12} />
                            {formatTime(query.timestamp)}
                          </span>
                          {query.videoId && (
                            <span className="px-1.5 py-0.5 bg-orange-100 text-orange-700 rounded">
                              {query.videoId.slice(0, 8)}...
                            </span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReplayQuery(query);
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-lg transition-colors"
                      >
                        <Play size={12} />
                        Replay
                      </button>
                    </div>
                  </div>

                  {/* Expanded events */}
                  {expandedId === query.id && (
                    <div className="px-5 pb-4">
                      <div className="ml-6 pl-4 border-l-2 border-slate-200">
                        <p className="text-xs text-slate-500 mb-2">
                          {query.events.length} events: {getEventSummary(query.events)}
                        </p>
                        <div className="space-y-1.5 max-h-48 overflow-y-auto">
                          {query.events.map((event, idx) => (
                            <div 
                              key={idx}
                              className="flex items-center gap-2 text-xs"
                            >
                              <span className={`
                                px-1.5 py-0.5 rounded font-mono
                                ${event.type === 'message' ? 'bg-blue-100 text-blue-700' : ''}
                                ${event.type === 'handoff' ? 'bg-purple-100 text-purple-700' : ''}
                                ${event.type === 'tool_call' ? 'bg-orange-100 text-orange-700' : ''}
                                ${event.type === 'tool_result' ? 'bg-green-100 text-green-700' : ''}
                                ${event.type === 'result' ? 'bg-emerald-100 text-emerald-700' : ''}
                                ${event.type === 'error' ? 'bg-red-100 text-red-700' : ''}
                                ${!['message', 'handoff', 'tool_call', 'tool_result', 'result', 'error'].includes(event.type) ? 'bg-slate-100 text-slate-600' : ''}
                              `}>
                                {event.type}
                              </span>
                              {'source' in event && (
                                <span className="text-slate-600">{(event as any).source}</span>
                              )}
                              {'target' in event && (
                                <span className="text-slate-400">→ {(event as any).target}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
