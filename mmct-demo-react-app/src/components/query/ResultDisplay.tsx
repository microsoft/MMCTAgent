import React from 'react';
import ReactMarkdown from 'react-markdown';
import { CheckCircle, XCircle, Clock, Coins } from 'lucide-react';
import { useQueryStore } from '../../store/queryStore';

export const ResultDisplay: React.FC = () => {
  const { result, tokenUsage, error, isLoading } = useQueryStore();
  
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-2xl p-5">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <XCircle size={20} />
          <span className="font-semibold">Error</span>
        </div>
        <p className="text-red-700 text-sm">{error}</p>
      </div>
    );
  }
  
  if (!result && !isLoading) {
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 text-center">
        <p className="text-slate-500">
          Ask a question to see the response here
        </p>
      </div>
    );
  }
  
  if (isLoading && !result) {
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 text-center">
        <div className="animate-pulse flex flex-col items-center gap-3">
          <div className="w-8 h-8 bg-slate-300 rounded-full" />
          <div className="h-4 bg-slate-300 rounded w-48" />
          <div className="h-3 bg-slate-200 rounded w-64" />
        </div>
      </div>
    );
  }
  
  if (!result) return null;
  
  return (
    <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
      {/* Header with status */}
      <div className="flex items-center justify-between px-5 py-3 bg-slate-50 border-b border-slate-200">
        <div className="flex items-center gap-2">
          {result.answer_found ? (
            <>
              <CheckCircle size={18} className="text-green-500" />
              <span className="text-sm font-medium text-green-700">Answer Found</span>
            </>
          ) : (
            <>
              <XCircle size={18} className="text-amber-500" />
              <span className="text-sm font-medium text-amber-700">No Answer Found</span>
            </>
          )}
        </div>
        
        {tokenUsage && (
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <Coins size={14} />
              {tokenUsage.prompt_tokens.toLocaleString()} input
            </span>
            <span className="flex items-center gap-1">
              {tokenUsage.completion_tokens.toLocaleString()} output
            </span>
          </div>
        )}
      </div>
      
      {/* Response content */}
      <div className="p-5">
        <div className="prose prose-slate prose-sm max-w-none">
          <ReactMarkdown>{result.response}</ReactMarkdown>
        </div>
      </div>
      
      {/* Citations */}
      {result.sources && result.sources.length > 0 && (
        <div className="px-5 pb-5">
          <div className="border-t border-slate-100 pt-4">
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
              Sources
            </h4>
            <div className="flex flex-wrap gap-2">
              {result.sources.map((source) => (
                <a
                  key={source.citation_id}
                  href={`${source.url}${source.start_time ? `&t=${timeToSeconds(source.start_time)}` : ''}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium text-slate-700 transition-colors"
                >
                  <span className="w-5 h-5 bg-blue-500 text-white rounded-full flex items-center justify-center text-[10px] font-bold">
                    {source.citation_id}
                  </span>
                  <Clock size={12} />
                  <span>{source.start_time} - {source.end_time}</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper to convert HH:MM:SS to seconds for YouTube URL
function timeToSeconds(time: string): number {
  const parts = time.split(':').map(Number);
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  } else if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }
  return parts[0] || 0;
}
