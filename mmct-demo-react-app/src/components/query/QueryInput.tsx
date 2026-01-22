import React, { useState } from 'react';
import { Send, Video, Loader2 } from 'lucide-react';
import { useQueryStore } from '../../store/queryStore';
import { useV2Query } from '../../hooks/useV2Query';

export const QueryInput: React.FC = () => {
  const [localQuery, setLocalQuery] = useState('');
  const [localVideoId, setLocalVideoId] = useState('');
  
  const { isLoading } = useQueryStore();
  const { streamQuery, abort } = useV2Query();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!localQuery.trim() || isLoading) return;
    
    streamQuery(localQuery.trim(), localVideoId.trim() || undefined);
  };
  
  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
        {/* Query input */}
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={localQuery}
            onChange={(e) => setLocalQuery(e.target.value)}
            placeholder="Ask a question about your videos..."
            className="flex-1 text-base bg-transparent outline-none text-slate-800 placeholder:text-slate-400"
            disabled={isLoading}
          />
          
          {isLoading ? (
            <button
              type="button"
              onClick={abort}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-colors"
            >
              <Loader2 size={18} className="animate-spin" />
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!localQuery.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors"
            >
              <Send size={18} />
              Ask
            </button>
          )}
        </div>
        
        {/* Video ID input (optional) */}
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-slate-100">
          <Video size={16} className="text-slate-400" />
          <input
            type="text"
            value={localVideoId}
            onChange={(e) => setLocalVideoId(e.target.value)}
            placeholder="Video ID (optional)"
            className="flex-1 text-sm bg-transparent outline-none text-slate-600 placeholder:text-slate-400"
            disabled={isLoading}
          />
        </div>
      </div>
    </form>
  );
};
