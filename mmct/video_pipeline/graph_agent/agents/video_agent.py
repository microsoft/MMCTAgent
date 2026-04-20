"""Video agent for the graph (swarm-based) pipeline.

Handles Neo4j graph interactions:
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

from mmct.video_pipeline.graph_agent.prompts.video_agent import VIDEO_AGENT_SYSTEM_PROMPT
from mmct.video_pipeline.graph_agent.middleware import ToolMiddleware, apply_middleware


class VideoAgent:
    """Video agent using AutoGen handoffs.

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
        model_context: Optional[ChatCompletionContext] = None,
        tool_middleware: Optional[List[ToolMiddleware]] = None,
    ):
        """Initialize the Video agent.

        Args:
            model_client: AutoGen model client.
            neo4j_provider: Neo4jQueryProvider instance.
            model_context: Optional shared model context for KV cache.
            tool_middleware: Optional list of ToolMiddleware instances to
                wrap tool callables with before/after hooks.
        """
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.model_context = model_context

        self._initialize_tools()
        if tool_middleware:
            self.tools = [apply_middleware(t, tool_middleware) for t in self.tools]
        self.agent = self._create_agent()

    def _initialize_tools(self):
        """Initialize tools that wrap Neo4jQueryProvider methods."""
        from mmct.video_pipeline.graph_agent.tools.graph_search_tool import GraphSearchTool
        from mmct.video_pipeline.graph_agent.tools.keyframe_tool import KeyframeRetrievalTool
        from mmct.video_pipeline.graph_agent.tools.video_discovery_tool import VideoDiscoveryTool
        from mmct.video_pipeline.graph_agent.tools.graph_traversal_tool import GraphTraversalTool
        from mmct.video_pipeline.graph_agent.tools.video_overview_tool import VideoOverviewTool

        graph_search = GraphSearchTool(
            neo4j_provider=self.neo4j_provider,
        )
        video_overview = VideoOverviewTool(
            neo4j_provider=self.neo4j_provider,
        )
        keyframe_tool = KeyframeRetrievalTool(
            neo4j_provider=self.neo4j_provider,
        )
        video_discovery = VideoDiscoveryTool(
            neo4j_provider=self.neo4j_provider,
        )
        graph_traversal = GraphTraversalTool(
            neo4j_provider=self.neo4j_provider,
        )

        self.tools = [
            video_overview.get_video_overview,     # Overview (no vector search)
            graph_search.search_graph,             # Multi-granularity vector search
            graph_traversal.traverse_graph,        # Unified hierarchy traversal
            keyframe_tool.search_keyframes,        # Image-based keyframe search
            video_discovery.find_relevant_videos,  # Cross-video discovery
        ]

    def _create_agent(self) -> AssistantAgent:
        """Create the AutoGen AssistantAgent."""
        return AssistantAgent(
            name="VideoAgent",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Agent that searches and traverses the Neo4j video knowledge graph.",
            system_message=VIDEO_AGENT_SYSTEM_PROMPT,
            tools=self.tools,
            reflect_on_tool_use=False,
            handoffs=["planner", "ImageAgent"],
        )
