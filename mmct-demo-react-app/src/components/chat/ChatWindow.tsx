import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, Sparkles, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { useQueryStore } from '../../store/queryStore';
import { useV2Query } from '../../hooks/useV2Query';
import { usePlaybackStore } from '../../store/playbackStore';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
  sources?: Array<{
    citation_id: number;
    url: string;
    start_time: string;
    end_time: string;
  }>;
}

interface ChatWindowProps {
  onOpenVisualizer: () => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ onOpenVisualizer }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [videoId, setVideoId] = useState('');
  const [showVideoInput, setShowVideoInput] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { isLoading, result, error } = useQueryStore();
  const { streamQuery } = useV2Query();
  const { events } = usePlaybackStore();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (result && !isLoading) {
      setMessages(prev => {
        const newMessages = [...prev];
        const loadingIndex = newMessages.findIndex(m => m.isLoading);
        if (loadingIndex !== -1) {
          newMessages[loadingIndex] = {
            ...newMessages[loadingIndex],
            content: result.response,
            isLoading: false,
            sources: result.sources,
          };
        }
        return newMessages;
      });
    }
  }, [result, isLoading]);

  useEffect(() => {
    if (error && !isLoading) {
      setMessages(prev => {
        const newMessages = [...prev];
        const loadingIndex = newMessages.findIndex(m => m.isLoading);
        if (loadingIndex !== -1) {
          newMessages[loadingIndex] = {
            ...newMessages[loadingIndex],
            content: `Error: ${error}`,
            isLoading: false,
          };
        }
        return newMessages;
      });
    }
  }, [error, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    const query = input.trim();
    setInput('');
    await streamQuery(query, videoId || undefined);
  };

  const timeToSeconds = (time: string): number => {
    const parts = time.split(':').map(Number);
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return parts[0] || 0;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-slate-200 bg-white/80 backdrop-blur-xl shadow-sm">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 via-purple-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-purple-500/30">
                <Sparkles size={24} className="text-white" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 bg-emerald-400 rounded-full border-2 border-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800">MMCT Agent</h1>
              <p className="text-xs text-slate-500">Multi-Agent Video Intelligence System</p>
            </div>
          </div>
          
          {events.length > 0 && (
            <button
              onClick={onOpenVisualizer}
              className="group flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white text-sm font-semibold rounded-xl shadow-lg shadow-purple-500/25 transition-all duration-300 hover:shadow-purple-500/40 hover:scale-105 hover:-translate-y-0.5"
            >
              <Zap size={18} className="group-hover:animate-pulse" />
              View Agent Flow
              <span className="ml-1 px-2 py-0.5 bg-white/20 rounded-md text-xs">{events.length}</span>
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <div className="relative mb-8">
                <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-violet-100 to-purple-100 flex items-center justify-center border border-violet-200 backdrop-blur-sm shadow-lg">
                  <Bot size={44} className="text-purple-500" />
                </div>
                <div className="absolute -inset-4 bg-purple-200/50 rounded-full blur-2xl -z-10" />
              </div>
              <h2 className="text-3xl font-bold text-slate-800 mb-3">Ask anything about your videos</h2>
              <p className="text-slate-500 max-w-lg text-lg">
                I orchestrate multiple AI agents to analyze video content and provide accurate answers with timestamps.
              </p>
              <div className="flex flex-wrap gap-3 mt-8 justify-center">
                {[
                  'Why do we decorate tree in Christmas?',
                  'What is the main topic of this video?',
                  'Summarize the key points',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="px-5 py-2.5 bg-white hover:bg-violet-50 border border-slate-200 hover:border-violet-300 rounded-xl text-sm text-slate-600 hover:text-violet-700 transition-all duration-200 hover:scale-105 shadow-sm"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''} animate-fade-in-up`}>
              <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-lg ${
                message.role === 'user' 
                  ? 'bg-gradient-to-br from-blue-500 to-cyan-500 shadow-cyan-500/20' 
                  : 'bg-gradient-to-br from-violet-500 to-purple-500 shadow-purple-500/20'
              }`}>
                {message.role === 'user' ? <User size={20} className="text-white" /> : <Bot size={20} className="text-white" />}
              </div>

              <div className={`flex-1 max-w-[75%] ${message.role === 'user' ? 'text-right' : ''}`}>
                <div className={`inline-block px-5 py-4 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-cyan-500/20'
                    : 'bg-white border border-slate-200 text-slate-700 shadow-sm'
                }`}>
                  {message.isLoading ? (
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-slate-500">Agents analyzing...</span>
                    </div>
                  ) : (
                    <div className={`prose prose-sm max-w-none ${message.role === 'user' ? 'prose-invert' : 'prose-slate'}`}>
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                </div>

                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {message.sources.map((source) => (
                      <a
                        key={source.citation_id}
                        href={`${source.url}&t=${timeToSeconds(source.start_time)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 px-3 py-1.5 bg-violet-50 hover:bg-violet-100 border border-violet-200 hover:border-violet-300 rounded-lg text-xs text-slate-600 transition-all duration-200"
                      >
                        <span className="w-5 h-5 bg-gradient-to-br from-violet-500 to-purple-500 text-white rounded-md flex items-center justify-center text-[10px] font-bold">
                          {source.citation_id}
                        </span>
                        {source.start_time} - {source.end_time}
                      </a>
                    ))}
                  </div>
                )}
                <p className="text-xs text-slate-400 mt-2">{message.timestamp.toLocaleTimeString()}</p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="flex-shrink-0 p-4 border-t border-slate-200 bg-white/80 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto">
          {showVideoInput && (
            <div className="mb-3 animate-fade-in-up">
              <input
                type="text"
                value={videoId}
                onChange={(e) => setVideoId(e.target.value)}
                placeholder="Enter Video ID (optional)"
                className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none focus:border-violet-400 focus:ring-2 focus:ring-violet-200 transition-all shadow-sm"
              />
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask about your videos..."
                rows={1}
                className="w-full px-5 py-4 pr-14 bg-white border border-slate-200 rounded-2xl text-slate-700 placeholder:text-slate-400 focus:outline-none focus:border-violet-400 focus:ring-2 focus:ring-violet-200 resize-none transition-all shadow-sm"
                style={{ minHeight: '60px' }}
              />
              <button
                type="button"
                onClick={() => setShowVideoInput(!showVideoInput)}
                className={`absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-all ${
                  showVideoInput || videoId ? 'text-violet-600 bg-violet-100' : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100'
                }`}
                title="Add Video ID"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="2" y="4" width="20" height="16" rx="2" />
                  <path d="m10 9 5 3-5 3z" />
                </svg>
              </button>
            </div>
            
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 w-14 h-14 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 disabled:from-slate-300 disabled:to-slate-300 text-white rounded-2xl flex items-center justify-center shadow-lg shadow-purple-500/25 transition-all duration-300 hover:shadow-purple-500/40 hover:scale-105 hover:-translate-y-0.5 disabled:shadow-none disabled:hover:scale-100 disabled:hover:translate-y-0"
            >
              {isLoading ? <Loader2 size={24} className="animate-spin" /> : <Send size={24} />}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
