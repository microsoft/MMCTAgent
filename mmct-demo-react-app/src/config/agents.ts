import {
  User,
  Brain,
  Video,
  Image,
  CheckCircle,
  FileText,
  Search,
  Frame,
  Box,
  Eye,
  ScanLine,
  Type,
  Shapes,
} from 'lucide-react';
import { AgentNodeConfig, Position } from '../types';

// Agent and Tool Node configurations
export const NODES: Record<string, AgentNodeConfig> = {
  // User node
  user: {
    id: 'user',
    label: 'User',
    role: 'Sends query',
    icon: User,
    color: '#64748b',
    type: 'agent',
  },
  // Main agents
  planner: {
    id: 'planner',
    label: 'Planner',
    role: 'Orchestrates agents',
    icon: Brain,
    color: '#3b82f6',
    type: 'agent',
  },
  VideoAgent: {
    id: 'VideoAgent',
    label: 'Video Agent',
    role: 'Analyzes video content',
    icon: Video,
    color: '#f97316',
    type: 'agent',
  },
  ImageAgent: {
    id: 'ImageAgent',
    label: 'Image Agent',
    role: 'Analyzes images',
    icon: Image,
    color: '#8b5cf6',
    type: 'agent',
  },
  Critic: {
    id: 'Critic',
    label: 'Critic',
    role: 'Validates responses',
    icon: CheckCircle,
    color: '#22c55e',
    type: 'agent',
  },
  // Video Agent Tools
  get_video_summary: {
    id: 'get_video_summary',
    label: 'Summary',
    role: 'Get video summary',
    icon: FileText,
    color: '#fb923c',
    type: 'tool',
    parent: 'VideoAgent',
  },
  search_video_context: {
    id: 'search_video_context',
    label: 'Search',
    role: 'Search video context',
    icon: Search,
    color: '#fb923c',
    type: 'tool',
    parent: 'VideoAgent',
  },
  get_video_frames: {
    id: 'get_video_frames',
    label: 'Frames',
    role: 'Extract video frames',
    icon: Frame,
    color: '#fb923c',
    type: 'tool',
    parent: 'VideoAgent',
  },
  detect_video_objects: {
    id: 'detect_video_objects',
    label: 'Objects',
    role: 'Detect objects in video',
    icon: Box,
    color: '#fb923c',
    type: 'tool',
    parent: 'VideoAgent',
  },
  // Image Agent Tools
  analyze_image_vit: {
    id: 'analyze_image_vit',
    label: 'ViT',
    role: 'Vision transformer analysis',
    icon: Eye,
    color: '#a78bfa',
    type: 'tool',
    parent: 'ImageAgent',
  },
  recognize_image: {
    id: 'recognize_image',
    label: 'Recognize',
    role: 'Image recognition',
    icon: ScanLine,
    color: '#a78bfa',
    type: 'tool',
    parent: 'ImageAgent',
  },
  detect_image_objects: {
    id: 'detect_image_objects',
    label: 'Objects',
    role: 'Detect objects in image',
    icon: Shapes,
    color: '#a78bfa',
    type: 'tool',
    parent: 'ImageAgent',
  },
  extract_image_text: {
    id: 'extract_image_text',
    label: 'OCR',
    role: 'Extract text from image',
    icon: Type,
    color: '#a78bfa',
    type: 'tool',
    parent: 'ImageAgent',
  },
};

// Node positions (percentages of container)
export const POSITIONS: Record<string, Position> = {
  // Top row: User -> Planner -> Critic
  user: { x: 25, y: 15 },
  planner: { x: 50, y: 15 },
  Critic: { x: 75, y: 15 },
  
  // VideoAgent on the left with tools in a circle around it
  VideoAgent: { x: 28, y: 55 },
  get_video_summary: { x: 16, y: 38 },
  search_video_context: { x: 40, y: 38 },
  get_video_frames: { x: 16, y: 72 },
  detect_video_objects: { x: 40, y: 72 },
  
  // ImageAgent on the right with tools in a circle around it
  ImageAgent: { x: 72, y: 55 },
  analyze_image_vit: { x: 60, y: 38 },
  recognize_image: { x: 84, y: 38 },
  extract_image_text: { x: 60, y: 72 },
  detect_image_objects: { x: 84, y: 72 },
};

// Define connection paths between nodes
export const CONNECTIONS = [
  { from: 'user', to: 'planner' },
  { from: 'planner', to: 'VideoAgent' },
  { from: 'planner', to: 'ImageAgent' },
  { from: 'planner', to: 'Critic' },
  // Video tools
  { from: 'VideoAgent', to: 'get_video_summary' },
  { from: 'VideoAgent', to: 'search_video_context' },
  { from: 'VideoAgent', to: 'get_video_frames' },
  { from: 'VideoAgent', to: 'detect_video_objects' },
  // Image tools
  { from: 'ImageAgent', to: 'analyze_image_vit' },
  { from: 'ImageAgent', to: 'recognize_image' },
  { from: 'ImageAgent', to: 'detect_image_objects' },
  { from: 'ImageAgent', to: 'extract_image_text' },
];

// Map tool names from API to our node IDs
export const TOOL_NAME_MAP: Record<string, string> = {
  // Direct mappings (our node IDs)
  'get_video_summary': 'get_video_summary',
  'search_video_context': 'search_video_context',
  'get_video_frames': 'get_video_frames',
  'detect_video_objects': 'detect_video_objects',
  'analyze_image_vit': 'analyze_image_vit',
  'recognize_image': 'recognize_image',
  'detect_image_objects': 'detect_image_objects',
  'extract_image_text': 'extract_image_text',
  
  // API tool names (from actual API responses)
  'get_context': 'search_video_context',
  'get_object_collection': 'detect_video_objects',
  'get_frames': 'get_video_frames',
  'get_summary': 'get_video_summary',
  'analyze_vit': 'analyze_image_vit',
  'image_vit': 'analyze_image_vit',
  'image_recognize': 'recognize_image',
  'image_objects': 'detect_image_objects',
  'image_ocr': 'extract_image_text',
  
  // Short/alternate names
  'video_summary': 'get_video_summary',
  'video_context': 'search_video_context',
  'video_frames': 'get_video_frames',
  'video_objects': 'detect_video_objects',
  'img_vit': 'analyze_image_vit',
  'img_recog': 'recognize_image',
  'img_obj': 'detect_image_objects',
  'img_ocr': 'extract_image_text',
  
  // Label-based names
  'Summary': 'get_video_summary',
  'Search': 'search_video_context',
  'Frames': 'get_video_frames',
  'Objects': 'detect_video_objects',
  'ViT': 'analyze_image_vit',
  'Recognize': 'recognize_image',
  'OCR': 'extract_image_text',
  
  // Lowercase versions
  'summary': 'get_video_summary',
  'search': 'search_video_context',
  'frames': 'get_video_frames',
  'objects': 'detect_video_objects',
  'vit': 'analyze_image_vit',
  'recognize': 'recognize_image',
  'ocr': 'extract_image_text',
  'context': 'search_video_context',
  'object_collection': 'detect_video_objects',
};

// Tools that are transfers/handoffs (not visualization tools)
export const TRANSFER_TOOLS = new Set([
  'transfer_to_videoagent',
  'transfer_to_imageagent', 
  'transfer_to_planner',
  'transfer_to_critic',
  'transfer_to_video_agent',
  'transfer_to_image_agent',
]);

// Map agent names from API to our node IDs (handles case variations)
export const AGENT_NAME_MAP: Record<string, string> = {
  // Direct mappings
  'planner': 'planner',
  'Planner': 'planner',
  'PLANNER': 'planner',
  'VideoAgent': 'VideoAgent',
  'videoagent': 'VideoAgent',
  'VIDEOAGENT': 'VideoAgent',
  'video_agent': 'VideoAgent',
  'Video Agent': 'VideoAgent',
  'ImageAgent': 'ImageAgent',
  'imageagent': 'ImageAgent',
  'IMAGEAGENT': 'ImageAgent',
  'image_agent': 'ImageAgent',
  'Image Agent': 'ImageAgent',
  'Critic': 'Critic',
  'critic': 'Critic',
  'CRITIC': 'Critic',
  'CriticAgent': 'Critic',
  'critic_agent': 'Critic',
  'user': 'user',
  'User': 'user',
  'USER': 'user',
};

// Get all agent IDs (excluding tools)
export const AGENT_IDS = Object.values(NODES)
  .filter(node => node.type === 'agent')
  .map(node => node.id);

// Get all tool IDs
export const TOOL_IDS = Object.values(NODES)
  .filter(node => node.type === 'tool')
  .map(node => node.id);
