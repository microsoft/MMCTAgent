import React from 'react';
import { 
  Play, 
  Pause, 
  SkipForward, 
  SkipBack, 
  RotateCcw,
  Circle
} from 'lucide-react';
import { usePlaybackStore } from '../../store/playbackStore';

export const PlaybackControls: React.FC = () => {
  const { 
    events, 
    currentIndex, 
    isPlaying, 
    isPaused, 
    isComplete,
    pause, 
    resume, 
    stepForward, 
    stepBackward, 
    replay 
  } = usePlaybackStore();
  
  const hasEvents = events.length > 0;
  const canStepBack = currentIndex > 0;
  const canStepForward = currentIndex < events.length - 1;
  
  // Calculate progress percentage
  const progress = hasEvents ? ((currentIndex + 1) / events.length) * 100 : 0;
  
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
          <span>Event {Math.max(0, currentIndex + 1)} of {events.length}</span>
          <span className="flex items-center gap-1.5">
            {isPlaying && !isPaused && (
              <>
                <Circle size={8} className="fill-green-500 text-green-500 animate-pulse" />
                <span className="text-green-600">Live</span>
              </>
            )}
            {isPaused && (
              <>
                <Circle size={8} className="fill-amber-500 text-amber-500" />
                <span className="text-amber-600">Paused</span>
              </>
            )}
            {isComplete && !isPlaying && (
              <>
                <Circle size={8} className="fill-blue-500 text-blue-500" />
                <span className="text-blue-600">Complete</span>
              </>
            )}
            {!isPlaying && !isComplete && !hasEvents && (
              <span className="text-slate-400">Ready</span>
            )}
          </span>
        </div>
        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      
      {/* Control buttons */}
      <div className="flex items-center justify-center gap-2">
        {/* Step backward */}
        <button
          onClick={stepBackward}
          disabled={!canStepBack}
          className="p-2 rounded-xl text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Previous event"
        >
          <SkipBack size={20} />
        </button>
        
        {/* Play/Pause */}
        {isPlaying && !isPaused ? (
          <button
            onClick={pause}
            className="p-3 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors"
            title="Pause"
          >
            <Pause size={24} />
          </button>
        ) : (
          <button
            onClick={resume}
            disabled={!hasEvents || isComplete}
            className="p-3 rounded-xl bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Play"
          >
            <Play size={24} />
          </button>
        )}
        
        {/* Step forward */}
        <button
          onClick={stepForward}
          disabled={!canStepForward}
          className="p-2 rounded-xl text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Next event"
        >
          <SkipForward size={20} />
        </button>
        
        {/* Replay */}
        <button
          onClick={replay}
          disabled={!hasEvents}
          className="p-2 rounded-xl text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors ml-2"
          title="Replay from start"
        >
          <RotateCcw size={20} />
        </button>
      </div>
    </div>
  );
};
