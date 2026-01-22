import { useCallback, useRef } from 'react';
import { AgentEvent } from '../types';
import { usePlaybackStore } from '../store/playbackStore';
import { useQueryStore } from '../store/queryStore';

// LocalStorage key for saved queries
const STORAGE_KEY = 'mmct_query_events';

interface SavedQuery {
  id: string;
  query: string;
  videoId?: string;
  timestamp: string;
  events: AgentEvent[];
}

// Helper to save events to localStorage
function saveQueryToStorage(query: string, videoId: string | undefined, events: AgentEvent[]) {
  try {
    const existing = localStorage.getItem(STORAGE_KEY);
    const savedQueries: SavedQuery[] = existing ? JSON.parse(existing) : [];
    
    const newEntry: SavedQuery = {
      id: `query_${Date.now()}`,
      query,
      videoId,
      timestamp: new Date().toISOString(),
      events,
    };
    
    // Add new query at the beginning
    savedQueries.unshift(newEntry);
    
    // Keep only last 20 queries to prevent storage overflow
    const trimmed = savedQueries.slice(0, 20);
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    console.log(`Saved query with ${events.length} events to localStorage`);
  } catch (err) {
    console.warn('Failed to save events to localStorage:', err);
  }
}

// Export helper to retrieve saved queries
export function getSavedQueries(): SavedQuery[] {
  try {
    const existing = localStorage.getItem(STORAGE_KEY);
    return existing ? JSON.parse(existing) : [];
  } catch {
    return [];
  }
}

// Export helper to clear saved queries
export function clearSavedQueries() {
  localStorage.removeItem(STORAGE_KEY);
}

export function useV2Query() {
  const abortControllerRef = useRef<AbortController | null>(null);
  const eventsCollectorRef = useRef<AgentEvent[]>([]);
  const queryInfoRef = useRef<{ query: string; videoId?: string }>({ query: '' });
  
  const { 
    setLoading, 
    setError, 
    setResult, 
    setTokenUsage,
    reset: resetQuery 
  } = useQueryStore();
  
  const { 
    addEvent, 
    reset: resetPlayback,
    setIsPlaying 
  } = usePlaybackStore();

  const streamQuery = useCallback(async (query: string, videoId?: string) => {
    // Abort any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    // Reset state
    resetQuery();
    resetPlayback();
    setLoading(true);
    setIsPlaying(true);
    
    // Reset events collector and store query info
    eventsCollectorRef.current = [];
    queryInfoRef.current = { query, videoId };

    // Build URL with query parameters (FastAPI Depends() expects query params)
    const params = new URLSearchParams();
    params.append('query', query);
    if (videoId && videoId.trim()) {
      params.append('video_id', videoId.trim());
    }
    params.append('use_critic_agent', 'true');
    params.append('stream', 'true');
    params.append('cache', 'false');

    const url = `/v2/query/stream?${params.toString()}`;

    // Debug: log what we're sending
    console.log('Sending request to:', url);

    try {
      const response = await fetch(url, {
        method: 'POST',
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        // Try to get detailed error from response body
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          console.error('Server error response:', errorData);
          errorMessage = errorData.detail || JSON.stringify(errorData);
        } catch {
          // Response wasn't JSON
          const text = await response.text();
          console.error('Server error text:', text);
          if (text) errorMessage = text;
        }
        throw new Error(errorMessage);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop()!;

        for (const eventStr of events) {
          if (!eventStr.trim()) continue;

          const eventMatch = eventStr.match(/event: (\w+)/);
          const dataMatch = eventStr.match(/data: (.+)/s);

          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1];
            try {
              const data = JSON.parse(dataMatch[1]);
              const event: AgentEvent = { type: eventType as AgentEvent['type'], ...data };
              
              // Add event to playback store
              addEvent(event);
              
              // Collect event for localStorage
              eventsCollectorRef.current.push(event);

              // Handle specific event types
              if (eventType === 'result') {
                setResult(data.content);
                setTokenUsage(data.token_usage);
              } else if (eventType === 'error') {
                setError(data.message);
              }
            } catch (parseError) {
              console.warn('Failed to parse event data:', dataMatch[1]);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      // Save collected events to localStorage
      if (eventsCollectorRef.current.length > 0) {
        saveQueryToStorage(
          queryInfoRef.current.query,
          queryInfoRef.current.videoId,
          eventsCollectorRef.current
        );
      }
      
      setLoading(false);
      setIsPlaying(false);
    }
  }, [addEvent, resetPlayback, resetQuery, setError, setIsPlaying, setLoading, setResult, setTokenUsage]);

  const abort = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    setIsPlaying(false);
  }, [setIsPlaying, setLoading]);

  return { streamQuery, abort };
}
