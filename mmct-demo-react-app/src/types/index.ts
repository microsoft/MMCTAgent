import { LucideIcon } from 'lucide-react';

// SSE Event Types
export type EventType = 
  | 'connected' 
  | 'message' 
  | 'tool_call' 
  | 'tool_result' 
  | 'handoff' 
  | 'result' 
  | 'complete' 
  | 'error';

export interface ToolCall {
  name: string;
  arguments: string;
}

export interface ToolResult {
  call_id: string;
  content: string;
}

export interface Citation {
  citation_id: number;
  video_id: string;
  url: string;
  start_time: string;
  end_time: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface QueryResult {
  response: string;
  answer_found: boolean;
  sources: Citation[];
}

// Base event interface
export interface BaseEvent {
  type: EventType;
  timestamp: string;
}

export interface ConnectedEvent extends BaseEvent {
  type: 'connected';
  message: string;
  query: string;
}

export interface MessageEvent extends BaseEvent {
  type: 'message';
  source: string;
  content: string;
}

export interface ToolCallEvent extends BaseEvent {
  type: 'tool_call';
  source: string;
  tool_names: string[];
  tools: ToolCall[];
}

export interface ToolResultEvent extends BaseEvent {
  type: 'tool_result';
  source: string;
  results: ToolResult[];
}

export interface HandoffEvent extends BaseEvent {
  type: 'handoff';
  source: string;
  target: string;
  content: string;
}

export interface ResultEvent extends BaseEvent {
  type: 'result';
  source: string;
  content: QueryResult;
  message_count: number;
  stop_reason: string;
  duration_seconds: number;
  token_usage: TokenUsage;
}

export interface CompleteEvent extends BaseEvent {
  type: 'complete';
  message: string;
}

export interface ErrorEvent extends BaseEvent {
  type: 'error';
  message: string;
}

export type AgentEvent = 
  | ConnectedEvent 
  | MessageEvent 
  | ToolCallEvent 
  | ToolResultEvent 
  | HandoffEvent 
  | ResultEvent 
  | CompleteEvent 
  | ErrorEvent;

// Agent & Node Types
export type NodeType = 'agent' | 'tool';

export interface AgentNodeConfig {
  id: string;
  label: string;
  role: string;
  icon: LucideIcon;
  color: string;
  type: NodeType;
  parent?: string;
}

export interface Position {
  x: number;
  y: number;
}

export interface Connection {
  from: string;
  to: string;
  active: boolean;
}

// Playback State
export interface PlaybackState {
  isPlaying: boolean;
  isPaused: boolean;
  currentIndex: number;
  events: AgentEvent[];
  speed: number;
}

// Query State
export interface QueryState {
  query: string;
  videoId: string;
  isLoading: boolean;
  error: string | null;
  result: QueryResult | null;
  tokenUsage: TokenUsage | null;
}
