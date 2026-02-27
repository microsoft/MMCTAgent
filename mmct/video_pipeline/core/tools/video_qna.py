# Importing modules
import asyncio
import json
import os
import re
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from loguru import logger

# Suppress internal logging

from typing import Annotated
from agent_framework import Agent, WorkflowBuilder, Workflow, AgentResponse
from agent_framework._tools import FunctionTool
from mmct.video_pipeline.core.tools.custom_middleware import LoggingAgentMiddleware, LoggingChatMiddleware, LoggingFunctionMiddleware, TerminationMiddleware

from mmct.video_pipeline.core.tools.get_context import GetContextTool
from mmct.video_pipeline.core.tools.get_relevant_frames import GetRelevantFrames
from mmct.video_pipeline.core.tools.query_frame import QueryFrameTool
from mmct.video_pipeline.core.tools.get_video_summary import GetVideoSummaryTool
from mmct.video_pipeline.core.tools.get_object_collection import GetObjectCollection
from mmct.video_pipeline.core.tools.critic import CriticTool
from mmct.video_pipeline.prompts_and_description import (
    get_planner_system_prompt,
    CRITIC_AGENT_SYSTEM_PROMPT,
    PLANNER_DESCRIPTION,
    CRITIC_DESCRIPTION,
)

from mmct.config.providers import VideoAgentProviderConfig

load_dotenv(override=True)


def parse_response_to_dict(content: str) -> Dict[str, Any]:
    """
    Fast JSON extractor with minimal scanning.
    Supports code-fenced JSON blocks and raw JSON objects.
    """

    def try_parse_json(s: str):
        try:
            data = json.loads(s)
            if all(k in data for k in ("answer", "source", "videos")):
                return data
        except Exception:
            return None

    try:
        clean = content.replace("TERMINATE", "").strip()

        # 1. Fast path: JSON inside code block
        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if block:
            parsed = try_parse_json(block.group(1))
            if parsed:
                return parsed

        # 2. Fast JSON extraction without brace matching
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            candidate = clean[start : end + 1]
            parsed = try_parse_json(candidate)
            if parsed:
                return parsed

        # Fallback
        logger.warning("No valid JSON found, fallback used.")
        return {"answer": clean, "source": ["TEXTUAL", "VISUAL"], "videos": []}

    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return {"answer": "Error parsing response", "source": [], "videos": []}


def _extract_last_text(output_events) -> str:
    """Pull the final text content from workflow output events."""
    if not output_events:
        return ""
    if isinstance(output_events, list):
        last = output_events[-1]
    else:
        last = output_events
    return getattr(last, "text", str(last))


def _is_terminated(conversation) -> bool:
    """Termination condition: last message contains TERMINATE."""
    return len(conversation) > 0 and "TERMINATE" in getattr(
        conversation[-1], "text", str(conversation[-1])
    )


class VideoQnA:
    """
    VideoQnA using Microsoft agent_framework WorkflowBuilder-based orchestration.

    Replaces autogen Swarm / RoundRobinGroupChat with WorkflowBuilder + add_edge.
    The Planner orchestrates tool calls; the optional Critic reviews the draft
    before the Planner finalises.

    Topology (with critic):
        planner ──► critic  (condition: "ready for criticism" in last message)
        critic  ──► planner (always, after evaluating)
        planner ──► END     (condition: "TERMINATE" in last message)

    Topology (without critic):
        planner ──► END     (condition: "TERMINATE" in last message)

    Provider dependencies (injected via VideoAgentProviderConfig):
    - llm_provider            : LLM client for agent reasoning
    - vectordb_chapter        : Video context / transcript retrieval
    - vectordb_object_registry: Video summaries and object collections
    - vectordb_keyframes      : Keyframe-level semantic search
    - embedding_provider      : Text embedding for semantic search
    - image_embedding_provider: Image embedding for frame search
    - storage_provider        : Blob storage access for frames

    Tools available to the Planner:
    1. get_video_summary    – High-level summaries; video discovery
    2. get_object_collection– Object counts, tracking, appearances
    3. get_context          – Transcript chunks + chapter visual summaries
    4. get_relevant_frames  – Frame discovery by visual query
    5. query_frame          – Vision-model analysis of frames / timestamps

    Args:
        query (str): Natural-language question about the video.
        provider (VideoAgentProviderConfig): Injected provider configuration.
        video_id (Optional[str]): Hash video ID to scope the search.
        url (Optional[str]): Video URL to scope the search.
        use_critic_agent (bool): Enable Critic review loop. Default True.
        verbose (bool): Whether to enable console logs. Default True.
        cache (bool): Whether to enable caching for model responses. Default False.
    """

    @staticmethod
    def _wrap(fn) -> FunctionTool:
        """Wrap a callable into a FunctionTool for agent_framework JSON schema."""
        return FunctionTool(name=fn.__name__, description=fn.__doc__ or "", func=fn)

    def __init__(
        self,
        query: str,
        provider: VideoAgentProviderConfig,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        use_critic_agent: bool = True,
        verbose: bool = True,
        cache: bool = False,
    ):
        self.query = query
        self.video_id = video_id
        self.url = url
        self.use_critic_agent = use_critic_agent
        self.verbose = verbose
        self.cache = cache
        self.provider = provider
        self.model_client = self.provider.llm_provider.get_agent_framework_client()

        # ── Tool instantiation ────────────────────────────────────────────
        get_context_tool_object = GetContextTool(
            embed_provider=self.provider.embedding_provider,
            vectordb_chapter=self.provider.vectordb_chapter,
        )
        get_video_summary_object = GetVideoSummaryTool(
            vectordb_object_registry=self.provider.vectordb_object_registry,
            embed_provider=self.provider.embedding_provider,
        )
        get_object_collection_object = GetObjectCollection(
            vectordb_object_registry=self.provider.vectordb_object_registry
        )
        get_relevant_frames_object = GetRelevantFrames(
            vectordb_keyframes=self.provider.vectordb_keyframes,
            image_embedding_provider=self.provider.image_embedding_provider,
        )
        query_frame_object = QueryFrameTool(
            llm_provider=self.provider.llm_provider,
            storage_provider=self.provider.storage_provider,
            vectordb_keyframes=self.provider.vectordb_keyframes,
            image_embedding_provider=self.provider.image_embedding_provider,
        )

        self.tools = [
            self._wrap(get_video_summary_object.get_video_summary),
            self._wrap(get_object_collection_object.get_object_collection),
            self._wrap(get_context_tool_object.get_context),
            self._wrap(get_relevant_frames_object.get_relevant_frames),
            self._wrap(query_frame_object.query_frame),
        ]

        # Task string passed to the workflow
        self.task = (
            f"query:{self.query}."
            + (f"\nInstruction:video id:{self.video_id}" if self.video_id is not None else "")
            + (f"\nurl:{self.url}" if self.url is not None else "")
        )

        self.workflow: Optional[Workflow] = None

    # ── Workflow construction ─────────────────────────────────────────────

    async def _build_workflow(self) -> None:
        """Build the WorkflowBuilder graph with planner (+ optional critic)."""
        planner_system_prompt = await get_planner_system_prompt(
            use_critic_agent=self.use_critic_agent,
        )

        # ── Middlewares ───────────────────────────────────────────────────
        middleware = []
        if self.verbose:
            middleware.extend([
                LoggingAgentMiddleware(),
                LoggingFunctionMiddleware(),
                LoggingChatMiddleware(),
            ])
        middleware.append(TerminationMiddleware())

        # NOTE: agent_framework v1.0.0b260130 uses `chat_client=`
        planner = Agent(
            client=self.model_client,
            instructions=planner_system_prompt,
            name="planner",
            description=PLANNER_DESCRIPTION,
            tools=self.tools,
            middleware=middleware
        )

        if self.use_critic_agent:
            critic_tool_object = CriticTool(llm_provider=self.provider.llm_provider)
            critic = Agent(
                client=self.model_client,
                instructions=CRITIC_AGENT_SYSTEM_PROMPT,
                name="critic",
                description=CRITIC_DESCRIPTION,
                tools=[self._wrap(critic_tool_object.critic_tool)],
                middleware=middleware
            )

            def _planner_ready_for_criticism(resp) -> bool:
                """
                Edge condition: route to critic when planner's last message
                contains 'ready for criticism'.
                resp is AgentExecutorResponse with:
                  .agent_response.messages  — list[Message] from this turn
                  .full_conversation        — full history list[Message]
                Each Message has a .text property for plain-text content.
                """
                messages = (
                    (resp.agent_response.messages if resp.agent_response else None)
                    or resp.full_conversation
                    or []
                )
                if not messages:
                    return False
                last_text = getattr(messages[-1], "text", "") or ""
                return "ready for criticism" in last_text.lower()

            self.workflow = (
                WorkflowBuilder(start_executor=planner)
                .add_edge(planner, critic, condition=_planner_ready_for_criticism)
                .add_edge(critic, planner)   # critic always returns to planner
                .build()
            )
        else:
            # No critic — planner runs to TERMINATE on its own
            self.workflow = WorkflowBuilder(start_executor=planner).build()

    # ── Execution helpers ─────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        Run the VideoQnA workflow (non-streaming).

        Returns:
            dict with keys:
              - result : parsed answer dict  (answer, source, videos)
              - tokens : best-effort token usage dict
        """
        await self._build_workflow()

        events = await self.workflow.run(self.task)
        output_events = events.get_outputs()

        last_content = _extract_last_text(output_events)
        parsed_result = parse_response_to_dict(last_content)

        # Aggregate tokens from all workflow events
        total_usage = {}
        for event in events:
            if event.type in ["data", "output"]:
                # Check for AgentResponse which contains usage_details
                if isinstance(event.data, AgentResponse) and event.data.usage_details:
                    for k, v in event.data.usage_details.items():
                        if v is not None:
                            total_usage[k] = total_usage.get(k, 0) + v

        return {"result": parsed_result, "tokens": total_usage}

    async def run_stream(self):
        """
        Run the VideoQnA workflow in streaming mode.

        Returns:
            ResponseStream of workflow events (iterate with `async for`).
        """
        await self._build_workflow()
        # In agent_framework v1.0.0b260212, streaming is done via
        # workflow.run(stream=True) which returns a ResponseStream directly.
        return self.workflow.run(self.task, stream=True)


# ── Public API ────────────────────────────────────────────────────────────────

async def video_qna(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[Optional[str], "The unique identifier of the video."] = None,
    url: Annotated[Optional[str], "The URL of the video to filter out the search results"] = None,
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent that validates the response."
    ] = True,
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
    verbose: Annotated[bool, "Set to True to enable console logs."] = True,
    cache: Annotated[bool, "Set to True to enable caching."] = False,
    provider: VideoAgentProviderConfig = None,
) -> Any:
    """
    Answer a user query about a video using WorkflowBuilder-based multi-agent orchestration.

    Uses Microsoft's agent_framework (v1.0.0b260212+) with WorkflowBuilder:
    - Planner orchestrates five video-analysis tools to build an answer.
    - Critic (optional) reviews the draft on "ready for criticism" handoff.
    - Workflow terminates when Planner emits TERMINATE.

    Tools available:
    1. get_video_summary    – High-level summaries; video discovery
    2. get_object_collection– Object counts, tracking, appearances
    3. get_context          – Transcript chunks + chapter visual summaries
    4. get_relevant_frames  – Frame discovery by visual query
    5. query_frame          – Vision-model analysis of frames / timestamps

    Args:
        query (str): The question to answer based on video content.
        video_id (Optional[str]): Hash video ID to scope the search.
        url (Optional[str]): Video URL to scope the search.
        use_critic_agent (bool): Enable Critic review loop. Default True.
        stream (bool): Return an async generator of events. Default False.
        provider (VideoAgentProviderConfig): Injected provider configuration.

    Returns:
        Non-streaming: dict with 'result' (answer, source, videos) and 'tokens'.
        Streaming    : async generator of workflow events.
    """
    instance = VideoQnA(
        query=query,
        video_id=video_id,
        url=url,
        use_critic_agent=use_critic_agent,
        verbose=verbose,
        cache=cache,
        provider=provider,
    )

    if stream:
        return await instance.run_stream()

    return await instance.run()


# ── Local smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Replace placeholders with real values before running ──
    from mmct.config.providers import VideoAgentProviderConfig

    provider = VideoAgentProviderConfig(...)   # supply your provider config here

    query = "<placeholder for query>"
    # video_id = "<placeholder for hash video Id>"  # Optional
    # url = "<placeholder for url to filter out the results>"  # Optional
    use_critic_agent = True
    stream = False

    if stream:
        async def _run_stream():
            gen = await video_qna(
                query=query,
                # video_id=video_id,
                # url=url,
                use_critic_agent=use_critic_agent,
                stream=True,
                provider=provider,
            )
            async for event in gen:
                print(event)

        asyncio.run(_run_stream())
    else:
        result = asyncio.run(
            video_qna(
                query=query,
                # video_id=video_id,
                # url=url,
                use_critic_agent=use_critic_agent,
                stream=False,
                provider=provider,
            )
        )
        print(result)
