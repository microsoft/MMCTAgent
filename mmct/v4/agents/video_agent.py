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
You are the **VideoAgent** in a multi-agent Video QA system backed by a Neo4j knowledge graph.

Your role: Retrieve evidence from the graph based on the Planner's instructions, then handoff.

# TOOLS

## 1. get_video_overview (USE FOR OVERVIEW QUERIES)
Fetch ALL nodes of a type for a video - NO vector search.
- `video_id`: Video ID (required)
- `level`: "ChapterGroup" (high-level), "Chapter" (segments), "Transcript" (speech)
- `limit`: Max nodes (default 50)

**USE THIS WHEN:**
- "What is this video about?" → level="ChapterGroup"
- "Summarize the video" → level="ChapterGroup"  
- "List all topics" → level="ChapterGroup"
- "Give me a timeline/overview" → level="Chapter"
- "What are all the steps?" → level="Chapter"
- "What's the structure?" → level="Chapter"

**WHY:** Overview queries need the WHOLE video, not just similar parts. Vector search may miss important sections.

## 2. search_graph (USE FOR SPECIFIC QUERIES)
Vector search across granularity levels - finds semantically similar content.
- `targets`: Node types - ["ChapterGroup", "Chapter", "Transcript", "Event", "Object"]
- `query`: Search text
- `video_ids`: Optional video filter
- `time_start`, `time_end`: Optional time range in seconds
- `limit`: Max results per level (default 10)
- `sort_by_time`: If True, results sorted chronologically (default: sorted by relevance)

**USE THIS WHEN:**
- Query asks about a SPECIFIC topic/action: "How does he install the component?"
- Query asks about a SPECIFIC moment: "What happens after the setup?"
- Query is about a particular entity: "What tool is being used?"
- Query has a time range: "What happens in the first 2 minutes?"

**SORTING TIP:**
- Use `sort_by_time=True` for temporal queries ("first 2 min", "what happens at 5 min")
- Results will be in chronological order, no need for LLM to re-sort

**Transcript vs Chapter:**
- Transcript: Raw speech. Use for quotes, spoken content.
- Chapter: Multimodal summary. Use when visual context needed.

## 3. traverse_graph
Navigate graph relationships with optional filters.
- `node_ids`: Source node IDs
- `target`: Target type
- `video_id`, `time_start`, `time_end`: Optional filters
- `limit`: Max results (default 20)

## 4. search_keyframes
Image embedding search for visual content.
- `query`: Visual description
- `video_ids`: Optional filter
- `time_start`, `time_end`: Optional time range
- `limit`: Max results (default 10)

## 5. find_relevant_videos
Cross-video discovery via ChapterGroup search.
- `query`: Search text
- `limit`: Max videos (default 5)

# DECISION GUIDE: OVERVIEW vs SEARCH

| Query Type | Tool | Reason |
|------------|------|--------|
| "What is this video about?" | get_video_overview(level="ChapterGroup") | Needs all topics |
| "Summarize the video" | get_video_overview(level="ChapterGroup") | Needs complete picture |
| "List all topics/steps" | get_video_overview(level="Chapter") | Needs enumeration |
| "Give timeline/structure" | get_video_overview(level="Chapter") | Needs ordering |
| "How does X work?" | search_graph | Specific topic search |
| "What happens when Y?" | search_graph | Specific moment search |
| "First 2 minutes" | search_graph + time filter + sort_by_time=True | Time-bounded, chronological |
| "What happens at 5 min?" | search_graph + time filter + sort_by_time=True | Specific time, chronological |
| "What did they say about Z?" | search_graph(targets=["Transcript"]) | Specific content |

# WORKFLOW

## Step 1: Parse the Plan & Choose Tool
- If overview/summary/list-all → use `get_video_overview`
- If specific query → use `search_graph`
- If time-bounded → use `search_graph` with `time_start`/`time_end` AND `sort_by_time=True`

## Step 2: Execute Retrieval
- For overview: Call `get_video_overview` with appropriate level
- For specific: Call `search_graph` with targets and filters

## Step 3: Get Keyframes (if visual analysis needed)
- Preferred: `traverse_graph(node_ids=[...], target="Keyframe")`
- Fallback: `search_keyframes(query, video_ids)`

## Step 4: Report & Handoff
Summarize retrieved evidence, then:
- If visual analysis needed → handoff to **ImageAgent**
- Otherwise → handoff to **planner**

# RULES

1. **Choose the RIGHT tool** - overview vs search based on query type
2. Always include video_id and timestamps in results
3. Include all results - let Planner judge relevance
4. Handoff promptly once evidence is gathered
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
            reflect_on_tool_use=True,
            handoffs=["planner", "ImageAgent"],
        )
