import { useState, useEffect } from 'react';
import { X, Play, Pause, SkipBack, SkipForward, RotateCcw, History, Trash2 } from 'lucide-react';
import { AgentCanvas } from '../visualization/AgentCanvas';
import { usePlaybackStore } from '../../store/playbackStore';
import { getSavedQueries, clearSavedQueries } from '../../hooks/useV2Query';
import { AgentEvent } from '../../types';

interface SavedQuery {
  id: string;
  query: string;
  videoId?: string;
  timestamp: string;
  events: AgentEvent[];
}

interface VisualizerOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

export const VisualizerOverlay: React.FC<VisualizerOverlayProps> = ({ isOpen, onClose }) => {
  const [showHistory, setShowHistory] = useState(false);
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  
  const {
    events,
    currentIndex,
    isPlaying,
    isPaused,
    pause,
    resume,
    stepForward,
    stepBackward,
    replay,
    loadEvents,
  } = usePlaybackStore();

  useEffect(() => {
    if (showHistory) {
      setSavedQueries(getSavedQueries() as SavedQuery[]);
    }
  }, [showHistory]);

  const handleReplayQuery = (query: SavedQuery) => {
    loadEvents(query.events);
    setShowHistory(false);
  };

  const handleClearHistory = () => {
    clearSavedQueries();
    setSavedQueries([]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 animate-fade-in">
      {/* Animated background - light cream theme */}
      <div className="absolute inset-0 bg-gradient-to-br from-amber-50 via-orange-50 to-rose-50">
        <div className="absolute inset-0 bg-gradient-to-tl from-sky-50/50 via-transparent to-violet-50/50" />
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-orange-200/30 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-rose-200/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-violet-200/20 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-20 p-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              <span className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-500 rounded-lg flex items-center justify-center">
                <div className="w-3 h-3 border-2 border-white rounded-sm" />
              </span>
              Agent Flow Visualizer
            </h2>
            <span className="px-3 py-1 bg-white/70 border border-slate-200 rounded-full text-sm text-slate-600 shadow-sm">
              {currentIndex + 1} / {events.length} events
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                showHistory 
                  ? 'bg-violet-500 text-white shadow-lg' 
                  : 'bg-white/70 text-slate-600 hover:bg-white shadow-sm border border-slate-200'
              }`}
            >
              <History size={18} />
              History
            </button>
            <button
              onClick={onClose}
              className="p-2 bg-white/70 hover:bg-red-500 text-slate-600 hover:text-white rounded-xl transition-all duration-200 shadow-sm border border-slate-200"
            >
              <X size={24} />
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="absolute inset-0 pt-20 pb-28 px-4 z-10">
        <div className="relative w-full h-full max-w-7xl mx-auto">
          <AgentCanvas />
          
          {/* History panel */}
          {showHistory && (
            <div className="absolute right-0 top-0 bottom-0 w-80 bg-white/95 backdrop-blur-xl border-l border-slate-200 shadow-2xl animate-slide-in-right overflow-hidden flex flex-col">
              <div className="flex items-center justify-between p-4 border-b border-slate-200">
                <h3 className="font-semibold text-slate-800">Query History</h3>
                {savedQueries.length > 0 && (
                  <button
                    onClick={handleClearHistory}
                    className="p-1.5 text-slate-400 hover:text-red-500 transition-colors"
                    title="Clear history"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
              
              <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {savedQueries.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">
                    <History size={32} className="mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No saved queries yet</p>
                  </div>
                ) : (
                  savedQueries.map((query) => (
                    <button
                      key={query.id}
                      onClick={() => handleReplayQuery(query)}
                      className="w-full text-left p-3 bg-slate-50 hover:bg-violet-50 border border-slate-200 hover:border-violet-300 rounded-xl transition-all duration-200 group"
                    >
                      <p className="text-sm text-slate-700 font-medium truncate group-hover:text-violet-600">
                        {query.query}
                      </p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-slate-400">
                          {new Date(query.timestamp).toLocaleDateString()}
                        </span>
                        <span className="text-xs text-violet-500 opacity-0 group-hover:opacity-100 transition-opacity">
                          ▶ Replay
                        </span>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Playback controls */}
      <div className="absolute bottom-0 left-0 right-0 z-20 p-6">
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={replay}
            className="p-3 bg-white/80 hover:bg-white text-slate-600 hover:text-slate-800 rounded-xl transition-all duration-200 hover:scale-110 shadow-sm border border-slate-200"
            title="Replay"
          >
            <RotateCcw size={22} />
          </button>
          
          <button
            onClick={stepBackward}
            disabled={currentIndex <= 0}
            className="p-3 bg-white/80 hover:bg-white disabled:opacity-30 disabled:hover:bg-white/80 text-slate-600 hover:text-slate-800 rounded-xl transition-all duration-200 hover:scale-110 disabled:hover:scale-100 shadow-sm border border-slate-200"
            title="Previous"
          >
            <SkipBack size={22} />
          </button>
          
          <button
            onClick={isPaused ? resume : pause}
            disabled={!isPlaying && currentIndex >= events.length - 1}
            className="p-5 bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600 text-white rounded-2xl shadow-lg shadow-violet-300 transition-all duration-200 hover:scale-110 hover:shadow-violet-400 disabled:opacity-50 disabled:hover:scale-100"
            title={isPaused ? 'Play' : 'Pause'}
          >
            {isPaused || (!isPlaying && currentIndex < events.length - 1) ? (
              <Play size={28} className="ml-0.5" />
            ) : (
              <Pause size={28} />
            )}
          </button>
          
          <button
            onClick={stepForward}
            disabled={currentIndex >= events.length - 1}
            className="p-3 bg-white/80 hover:bg-white disabled:opacity-30 disabled:hover:bg-white/80 text-slate-600 hover:text-slate-800 rounded-xl transition-all duration-200 hover:scale-110 disabled:hover:scale-100 shadow-sm border border-slate-200"
            title="Next"
          >
            <SkipForward size={22} />
          </button>
        </div>
      </div>
    </div>
  );
};
