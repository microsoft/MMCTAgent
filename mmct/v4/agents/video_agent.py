"""V4 Video Agent - Graph retrieval and keyframe decisions.

The VideoAgent handles Neo4j graph interactions:
1. Executes multi-granularity vector searches via tools
2. Fetches complete video overviews (no vector search)
3. Traverses graph relationships (up/down the hierarchy)
4. Decides when keyframes are needed for visual analysis
5. Returns evidence to Planner or hands off to ImageAgent

Uses AutoGen's handoff mechanism for agent communication.
"""

from typing import Optional, List
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext


# System prompt for V4 VideoAgent
V4_VIDEO_AGENT_SYSTEM_PROMPT = """
You are the **VideoAgent** in a Video QA system. You execute the Planner's retrieval plan against the Neo4j knowledge graph.

# TOOLS

1. **search_graph** — Vector similarity search. YOUR PRIMARY TOOL.
   - `targets`: node types from ["Chapter", "Event", "Transcript", "Object", "ChapterGroup"]
   - `query`, `video_ids` (list), `time_start`, `time_end`, `limit` (default 5), `sort_by_time`

2. **find_relevant_videos** — Discover relevant videos for cross-video queries.
   - `query`, `limit` (default 3)

3. **get_video_overview** — Fetch ALL nodes for a video (no vector search).
   - `video_id` (required), `level`: "ChapterGroup" | "Chapter" | "Transcript", `limit` (default 50)
   - Use ONLY when the Planner's strategy says "OVERVIEW". NEVER use as a fallback or second-pass retrieval.

4. **traverse_graph** — Navigate graph relationships (parent/child/sibling nodes).
   - `node_ids`, `target`, `video_id`, `time_start`, `time_end`, `limit` (default 20)

5. **search_keyframes** — Image embedding search for visual content.
   - `query`, `video_ids`, `time_start`, `time_end`, `limit` (default 10)

# EXECUTION RULES

1. **Follow the Planner's plan exactly** — use the targets, query text, and scope specified.
2. For cross-video queries: call `find_relevant_videos` first, then ONE `search_graph` with all returned video_ids.
3. Batch all video_ids in ONE `search_graph` call — never make separate calls per video.
4. **After getting results, hand off immediately.** Do NOT summarize, comment, or ask questions.
   - If the plan has **Visual flag: true** → First call `search_graph` to find relevant chapters. Then call `traverse_graph` with the top chapter IDs and `target="Keyframe"` to get their actual keyframes. If `search_graph` returns no relevant chapters, fall back to `search_keyframes` instead. Then hand off to **ImageAgent** (NOT planner) with the keyframe blob_urls.
   - Otherwise → hand off to **planner**.
5. If a tool returns an error or empty results, hand off to `transfer_to_planner` anyway — let the Planner decide next steps.
6. **Do NOT call `get_video_overview` unless the Planner's plan explicitly uses the OVERVIEW strategy.**
"""


class V4VideoAgent:
    """V4 Video Agent using AutoGen handoffs.
    
    Handles Neo4j graph interactions:
    - Video overview fetching (no vector search)
    - Multi-granularity vector searches
    - Unified graph traversal (up/down hierarchy)
    - Cross-video discovery
    - Keyframe decision logic
    
    Tools (5 total):
    - get_video_overview: Fetch all nodes for video overview
    - search_graph: Multi-granularity vector search
    - traverse_graph: Unified relationship traversal with filters
    - search_keyframes: Image-based keyframe search
    - find_relevant_videos: Cross-video discovery
    """
    
    def __init__(
        self,
        model_client,
        neo4j_provider,
        embedding_provider,
        image_embedding_provider=None,
        model_context: Optional[ChatCompletionContext] = None,
    ):
        """Initialize the Video agent.
        
        Args:
            model_client: AutoGen model client.
            neo4j_provider: Neo4jQueryProvider instance.
            embedding_provider: Text embedding provider for search queries.
            image_embedding_provider: Optional image embedding provider for keyframe search.
            model_context: Optional shared model context for KV cache.
        """
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.model_context = model_context
        
        self._initialize_tools()
        self.agent = self._create_agent()
    
    def _initialize_tools(self):
        """Initialize tools that wrap Neo4jQueryProvider methods."""
        from mmct.v4.tools.graph_search_tool import GraphSearchTool
        from mmct.v4.tools.keyframe_tool import KeyframeRetrievalTool
        from mmct.v4.tools.video_discovery_tool import VideoDiscoveryTool
        from mmct.v4.tools.graph_traversal_tool import GraphTraversalTool
        from mmct.v4.tools.video_overview_tool import VideoOverviewTool
        
        # Initialize tool instances
        graph_search = GraphSearchTool(
            neo4j_provider=self.neo4j_provider,
            embedding_provider=self.embedding_provider,
        )
        
        video_overview = VideoOverviewTool(
            neo4j_provider=self.neo4j_provider,
        )
        
        keyframe_tool = KeyframeRetrievalTool(
            neo4j_provider=self.neo4j_provider,
            image_embedding_provider=self.image_embedding_provider,
        )
        
        video_discovery = VideoDiscoveryTool(
            neo4j_provider=self.neo4j_provider,
            embedding_provider=self.embedding_provider,
        )
        
        graph_traversal = GraphTraversalTool(
            neo4j_provider=self.neo4j_provider,
        )
        
        # Register 5 callable functions for AutoGen
        self.tools = [
            video_overview.get_video_overview,    # Overview (no vector search)
            graph_search.search_graph,            # Multi-granularity vector search
            graph_traversal.traverse_graph,       # Unified hierarchy traversal
            keyframe_tool.search_keyframes,       # Image-based keyframe search
            video_discovery.find_relevant_videos, # Cross-video discovery
        ]
    
    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen AssistantAgent."""
        return AssistantAgent(
            name="VideoAgent",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Agent that searches and traverses the Neo4j video knowledge graph.",
            system_message=V4_VIDEO_AGENT_SYSTEM_PROMPT,
            tools=self.tools,
            reflect_on_tool_use=False,
            handoffs=["planner", "ImageAgent"],
        )
