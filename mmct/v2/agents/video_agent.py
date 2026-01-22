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
You are a Video Analysis Agent. Your goal is to efficiently retrieve and analyze information from videos to answer the Planner's queries.
You have access to tools to get video summaries, object collections, transcripts/context, and relevant frames.
You do NOT have the ability to analyze the visual content of specific frames directly (no query_frame).
If you find relevant frames that need visual analysis, report their timestamps and frame names to the Planner so it can delegate to the ImageAgent.

Tools available:
- `get_video_summary`: Get high-level summary and find the most relevant video(s) for a query.
- `get_object_collection`: Get object counts/tracking info (requires video_id).
- `get_context`: Get transcripts and chapter summaries. Can be called with or without video_id.
- `get_relevant_frames`: Find relevant frame names based on visual search. **ONLY use when explicitly needed for visual analysis.**

# EFFICIENCY RULES - CRITICAL
1. **Batch independent tool calls**: When you need multiple types of information, call all relevant tools in a SINGLE response.
   - If video_id is already provided in the task: Call `get_video_summary`, `get_context`, and `get_object_collection` together.
   - If video_id is NOT provided: Call `get_video_summary` first to find the relevant video, then in the NEXT response batch `get_context` and `get_object_collection` with the discovered video_id.
   - `get_context` can also be called WITHOUT video_id for a broader search across all videos.

2. **Avoid unnecessary tool calls**: 
   - Do NOT call `get_relevant_frames` unless the query explicitly requires visual frame analysis (e.g., "what color", "show me", "what does X look like").
   - For factual/historical/explanatory queries (e.g., "why", "how", "what is"), text-based tools are sufficient.

3. **Be concise**: After gathering information, provide a brief summary and handoff to planner immediately. Do NOT generate verbose intermediate responses.

4. **Handoff quickly**: Once you have gathered sufficient information to answer the query, handoff to planner immediately. Do not make additional tool calls "just in case".

When answering:
1. Assess what information is needed and batch your tool calls.
2. **Always provide EXPLICIT start_time AND end_time** for each piece of information (e.g., "32.0s to 73.0s", NOT "48.68s onward").
3. ONLY mention frame analysis if the query genuinely requires visual inspection.
4. Keep your response concise - summarize findings and handoff to planner.
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
