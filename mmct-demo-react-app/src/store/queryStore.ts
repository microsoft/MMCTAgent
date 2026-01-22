import { create } from 'zustand';
import { QueryResult, TokenUsage } from '../types';

interface QueryStore {
  query: string;
  videoId: string;
  isLoading: boolean;
  error: string | null;
  result: QueryResult | null;
  tokenUsage: TokenUsage | null;
  // Actions
  setQuery: (query: string) => void;
  setVideoId: (videoId: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setResult: (result: QueryResult | null) => void;
  setTokenUsage: (usage: TokenUsage | null) => void;
  reset: () => void;
}

export const useQueryStore = create<QueryStore>((set) => ({
  query: '',
  videoId: '',
  isLoading: false,
  error: null,
  result: null,
  tokenUsage: null,

  setQuery: (query) => set({ query }),
  setVideoId: (videoId) => set({ videoId }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setResult: (result) => set({ result }),
  setTokenUsage: (usage) => set({ tokenUsage: usage }),
  
  reset: () => set({
    isLoading: false,
    error: null,
    result: null,
    tokenUsage: null,
  }),
}));
