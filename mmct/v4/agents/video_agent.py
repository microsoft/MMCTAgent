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

2. **find_relevant_videos** — Discover relevant videos when NO specific video_ids are given.
   - `query`, `video_ids` (optional list to constrain discovery), `limit` (default 3)
   - ONLY use when the Planner's scope is "Cross-video (all videos)". NEVER use when specific video_ids are provided.

3. **get_video_overview** — Fetch ALL nodes for a video (no vector search).
   - `video_id` (required), `level`: "ChapterGroup" | "Chapter" | "Transcript", `limit` (default 50)
   - Use ONLY when the Planner's strategy says "OVERVIEW". NEVER use as a fallback or second-pass retrieval.

4. **traverse_graph** — Navigate graph relationships (parent/child/sibling nodes).
   - `node_ids`, `target`, `video_id`, `time_start`, `time_end`, `limit` (default 20)

5. **search_keyframes** — Image embedding search for visual content.
   - `query`, `video_ids`, `time_start`, `time_end`, `limit` (default 10)

# EXECUTION RULES

1. **Follow the Planner's plan exactly** — use the targets, query text, and scope specified.
2. **Video scope handling:**
   - **Single video or Multi-video (specific list):** Use the provided video_ids directly in `search_graph`. Do NOT call `find_relevant_videos`.
   - **Cross-video (all videos):** Call `find_relevant_videos` first to discover relevant videos, then ONE `search_graph` with the returned video_ids.
3. Batch all video_ids in ONE `search_graph` call — never make separate calls per video.
4. **Multiple sub-queries:** If the Planner provides multiple sub-queries, call `search_graph` for ALL sub-queries **in parallel** (i.e., issue all tool calls in a single response), all using the same video_ids and targets. This is expected for broad multi-video queries.
5. **After getting search results, decide if context expansion is needed** before handing off:
   - If the plan has **Visual flag: true** → First call `search_graph` to find relevant chapters. Then call `traverse_graph` with the top chapter IDs and `target="Keyframe"` to get their actual keyframes. If `search_graph` returns no relevant chapters, fall back to `search_keyframes` instead. Then hand off to **ImageAgent** (NOT planner) with the keyframe blob_urls.
   - Otherwise → check if **preceding context** is needed (see CONTEXT EXPANSION below), then hand off to **planner**.
5. **When handing off, call the transfer tool IMMEDIATELY.** Do NOT generate any text, summary, commentary, or questions before or alongside the handoff call. The tool results already contain all the evidence the Planner needs.
6. If a tool returns an error or empty results, hand off to `transfer_to_planner` anyway — let the Planner decide next steps.
7. **Do NOT call `get_video_overview` unless the Planner's plan explicitly uses the OVERVIEW strategy.**

# CONTEXT EXPANSION (when to traverse for neighboring chapters)

After `search_graph` returns Chapter results, check whether the query needs the **beginning** of a topic. If it does, and the earliest matched chapter for a video is NOT the first chapter (chunk_index > 0), use `traverse_graph` to fetch preceding chapter(s) so the Planner can set accurate start timestamps.

**When to expand:** The query asks to "define", "introduce", "explain from the start", "what is", "take me to where … begins", or any phrasing that implies the viewer wants to watch a topic from its introduction.

**When NOT to expand:** The query asks about a specific detail, example, step, comparison, or moment — the viewer doesn't need the topic introduction.

**How to expand:** Call `traverse_graph` with the earliest matched chapter ID per video, `target="ChapterGroup"`, to get the parent topic group (which contains the true topic start_time). Include these results alongside the original search results when handing off to the Planner.

**Keep it lightweight** — only expand for the earliest chapter per video, not every result. Do NOT summarize or comment on the extra context; just include it in the handoff.
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
