import React from 'react';
import { usePlaybackStore } from '../../store/playbackStore';
import { MessageCircle } from 'lucide-react';

export const ThoughtBubble: React.FC = () => {
  const currentMessage = usePlaybackStore((state) => state.currentMessage);
  
  if (!currentMessage) return null;
  
  return (
    <div className="absolute top-4 left-4 z-20 animate-fade-in-up">
      <div className="bg-white/95 backdrop-blur-sm border border-slate-200 rounded-2xl px-5 py-3 shadow-lg max-w-md">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center">
            <MessageCircle size={16} className="text-violet-600" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-0.5">
              {currentMessage.source}
            </p>
            <p className="text-sm text-slate-700 leading-relaxed">
              {currentMessage.content}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
