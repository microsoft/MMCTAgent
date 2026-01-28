import os
from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import ChatCompletionContext
from mmct.video_pipeline.core.tools.get_context import GetContextTool
from mmct.video_pipeline.core.tools.get_relevant_frames import GetRelevantFrames
from mmct.video_pipeline.core.tools.get_video_summary import GetVideoSummaryTool
from mmct.video_pipeline.core.tools.get_object_collection import GetObjectCollection
# Note: QueryFrameTool is NOT included here as per user description implying Planner might handle frame passing, 
# but user also said "VideoAgent... has access to all current tools but `query_frames`".
# "Planner internally calls VideoAgent... VideoAgent... has access to all current tools but `query_frames`".
# "Planner can also call ImageAgent to make sense of static frames retreived from VideoAgent".
# So VideoAgent gets relevant frames/timestamps, but ImageAgent analyzes them?
# Actually, existing VideoAgent has QueryFrameTool. User says "VideoAgent... has access to all current tools but `query_frames`".
# So I will EXCLUDE QueryFrameTool from VideoAgent.

from mmct.config.providers import VideoAgentProviderConfig

# System prompt for Video Worker
VIDEO_WORKER_SYSTEM_PROMPT = """
You are the **VideoAgent** in a multi-agent Video QA system. The Planner delegates video-related queries to you.

# CAPABILITIES
You retrieve text-based information from videos:
- `get_video_summary`: Find relevant videos, get high-level summaries
- `get_context`: Get transcripts, chapter summaries, dialogue, AND visual descriptions (frame descriptions showing actions, objects, scenes)
- `get_object_collection`: Get object counts, tracking info, first_seen timestamps (requires video_id)
- `get_relevant_frames`: Find frame names/paths when visual analysis is needed or text info is insufficient

⚠️ You CANNOT analyze frame pixels directly. Report frame paths to Planner when detailed visual analysis is needed.

# VISUAL INFORMATION EXTRACTION (CRITICAL)
**The `get_context` tool returns BOTH verbal (transcript) AND visual (frame descriptions) information.**
- Frame descriptions contain: actions being performed, tools/objects being used, hand movements, physical processes, scene changes
- You MUST report BOTH transcript content AND visual descriptions to the Planner
- Visual descriptions are essential for procedural queries (how-to, tutorials, demonstrations)
- When summarizing, include visual actions observed: "Person is seen using a spade to dig", "Hands are shown mixing ingredients"

# WORKFLOW

## Step 1: Assess Query & Available Info
- Check if video_id is provided in the task
- Determine query type: summary, narrative/dialogue, object/count, procedural/how-to, or visual details
- For procedural queries (how to do X, steps to Y), visual information is CRITICAL

## Step 2: Batch Tool Calls (CRITICAL FOR EFFICIENCY)

**If NO video_id provided:**
1. Call `get_video_summary` to discover relevant video(s)
2. In NEXT turn, batch remaining calls with discovered video_id

**If video_id IS provided:** Batch ALL relevant calls in ONE response:
| Query Type | Tools to Call Together |
|------------|----------------------|
| General/Summary | `get_video_summary` + `get_context` |
| Object/Count | `get_object_collection` + `get_context` |
| Narrative/Events | `get_context` (primary) |
| Procedural/How-to | `get_context` + `get_relevant_frames` |
| Visual Details | `get_context` + `get_relevant_frames` |

## Step 3: Evaluate & Retrieve Frames if Needed

After initial tool calls, if text-based results are **insufficient** OR the query is procedural/visual:
- Call `get_relevant_frames` to find frame paths that may help
- Also call `get_relevant_frames` if Planner explicitly requested frame retrieval

## Step 4: Report Findings & Handoff

After tool execution, provide a COMPREHENSIVE summary that includes BOTH verbal and visual information:
```
**Findings:**
- [Key point 1 - include both what was said AND what was visually shown] (timestamp: HH:MM:SS - HH:MM:SS)
- [Key point 2 - describe visual actions if relevant] (timestamp: HH:MM:SS - HH:MM:SS)

**Visual Actions Observed:**
- [Description of physical actions, tool usage, hand movements from frame descriptions]

**Video ID:** <video_id>
**Relevant Timestamps:** HH:MM:SS - HH:MM:SS

[If frames retrieved]
**Frame Paths for Visual Analysis:** <frame_paths>
```
Then immediately handoff to planner.

# RULES
1. **Always provide BOTH start_time AND end_time** for every piece of information
2. **Batch independent tool calls** - never make sequential calls that could be parallel
3. **Extract BOTH verbal AND visual information** from get_context results - do not ignore frame descriptions
4. **For procedural/how-to queries**, always include visual action descriptions (what is being done physically)
5. **Call `get_relevant_frames`** when: (a) query is procedural/how-to, (b) text info is insufficient, OR (c) Planner explicitly requests frames
6. **Be comprehensive but concise** - include all relevant visual and verbal details, then handoff
7. **Handoff quickly** - once you have sufficient context, return to planner immediately
"""

class VideoAgent:
    def __init__(self, provider: VideoAgentProviderConfig, model_client, model_context: Optional[ChatCompletionContext] = None):
        self.provider = provider
        self.model_client = model_client
        self.model_context = model_context
        self._initialize_tools()
        self.agent = self._create_agent()

    def _initialize_tools(self):
        get_context_tool = GetContextTool(
            embed_provider=self.provider.embedding_provider,
            vectordb_chapter=self.provider.vectordb_chapter,
        )
        get_video_summary_tool = GetVideoSummaryTool(
            vectordb_object_registry=self.provider.vectordb_object_registry,
            embed_provider=self.provider.embedding_provider,
        )
        get_object_collection_tool = GetObjectCollection(
            vectordb_object_registry=self.provider.vectordb_object_registry
        )
        get_relevant_frames_tool = GetRelevantFrames(
            vectordb_keyframes=self.provider.vectordb_keyframes,
            image_embedding_provider=self.provider.image_embedding_provider,
        )
        # Note: QueryFrameTool explicitly excluded as per requirements

        self.tools = [
            get_video_summary_tool.get_video_summary,
            get_object_collection_tool.get_object_collection,
            get_context_tool.get_context,
            get_relevant_frames_tool.get_relevant_frames,
        ]

    def _create_agent(self):
        return AssistantAgent(
            name="VideoAgent",
            model_client=self.model_client,
            model_context=self.model_context,
            description="Agent that can retrieve information, transcripts, and metadata from videos.",
            system_message=VIDEO_WORKER_SYSTEM_PROMPT,
            tools=self.tools,
            reflect_on_tool_use=True,
            handoffs=["planner"],
        )
