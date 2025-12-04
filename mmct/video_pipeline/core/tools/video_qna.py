# Importing modules
import asyncio
import json
import os
import re
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from loguru import logger

# Suppress autogen internal logging
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import Swarm, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
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

from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache as DiskCache
from mmct.config.providers import VideoAgentProviderConfig

load_dotenv(override=True)


def parse_response_to_dict(content: str) -> Dict[str, Any]:
    """
    Fast JSON extractor with minimal scanning.
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


class VideoQnA:
    """
    VideoQnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    This class uses dependency injection via VideoAgentProviderConfig to access:
    - llm_provider: For LLM-based reasoning and response generation
    - vectordb_chapter: For retrieving video context and transcripts
    - vectordb_object_registry: For retrieving video summaries and object collections
    - vectordb_keyframes: For searching relevant video frames
    - embedding_provider: For generating embeddings for semantic search
    - storage_provider: For accessing stored video frames

    MMCT consists of:
    - **Planner Agent**: Has access to five tools for comprehensive video analysis:
      1. get_video_summary: Retrieves high-level video summaries (can be called without video_id for discovery)
      2. get_object_collection: Retrieves object descriptions with counts (requires video_id/url, semantic query based on video summary)
      3. get_context: Retrieves transcript chunks and visual summary chapter documents (requires video_id/url)
      4. get_relevant_frames: Gets specific frame names based on visual queries
      5. query_frame: Analyzes downloaded frames with vision models or fetch frames on filter criteria and then analyze them.
    - **Critic Agent**: Validates the planner's output.

    Workflow:
    1. If video_id/url NOT provided → Call get_video_summary first to discover relevant videos
    2. For object-related queries → Use get_object_collection with video_id (semantic query based on summary)
    3. For narrative/dialogue queries → Use get_context with video_id
    4. For visual verification → Use query_frame

    Args:
        query (str): The natural language question to be answered based on the video content.
        provider (VideoAgentProviderConfig): Provider configuration containing all required providers.
        video_id (Optional[str]): The unique identifier of the video.
        url (Optional[str]): URL of the video to filter search results.
        use_critic_agent (bool, optional): Whether to use the critic agent for answer refinement. Defaults to True.
        cache (bool, optional): Whether to enable caching for model responses. Defaults to True.
    """

    def __init__(
        self,
        query: str,
        provider: VideoAgentProviderConfig,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        use_critic_agent: bool = True,
        cache: bool = True,
    ):
        self.query = query
        self.video_id = video_id
        self.use_critic_agent = use_critic_agent
        self.url = url
        self.cache = cache
        self.provider = provider
        self.model_client = self.provider.llm_provider.get_autogen_client()

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

        # Only enable caching if cache parameter is True
        if self.cache:
            use_cache_backend = os.getenv("AUTOGEN_CACHE_BACKEND", "disk")  # "disk" or "redis"
            if use_cache_backend.lower() == "redis":
                # Shared cache across processes
                from autogen_ext.cache_store.redis import RedisStore
                import redis

                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                redis_client = redis.from_url(redis_url)
                store = RedisStore[CHAT_CACHE_VALUE_TYPE](redis_client)  # type: ignore
            else:
                # Local persistent cache
                cache_dir = os.getenv("AUTOGEN_DISK_CACHE_DIR", "./.autogen_ext_cache")
                store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](DiskCache(cache_dir))  # type: ignore

            # Wrap the base model client so AgentChat uses the cached client everywhere
            self.model_client = ChatCompletionCache(self.model_client, store)

        self.tools = [
            get_video_summary_object.get_video_summary,
            get_object_collection_object.get_object_collection,
            get_context_tool_object.get_context,
            get_relevant_frames_object.get_relevant_frames,
            query_frame_object.query_frame,
        ]
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = (
            f"query:{self.query}."
            + (f"\nInstruction:video id:{self.video_id}" if self.video_id is not None else "")
            + (f"\nurl:{self.url}" if self.url is not None else "")
        )

    async def _initialize_agents(self):
        # system prompt for video planner agent with comprehensive tool access
        planner_system_prompt = await get_planner_system_prompt(
            use_critic_agent=self.use_critic_agent,
        )

        # Define Planner agent
        self.planner = AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_client_stream=False,
            description=PLANNER_DESCRIPTION,
            system_message=(f"""{planner_system_prompt}"""),
            tools=self.tools,
            reflect_on_tool_use=True,
            max_tool_iterations=15,  # Reduced from 100 to 15 for faster response
            handoffs=["critic"] if self.use_critic_agent else [],
        )

        text_mention_termination = TextMentionTermination("TERMINATE")
        # max_messages_termination = MaxMessageTermination(max_messages=20)
        termination = text_mention_termination

        if self.use_critic_agent:
            critic_tool_object = CriticTool(llm_provider = self.provider.llm_provider)
            self.critic = AssistantAgent(
                name="critic",
                model_client=self.model_client,
                model_client_stream=False,
                description=CRITIC_DESCRIPTION,
                system_message=(f"{CRITIC_AGENT_SYSTEM_PROMPT}"),
                tools=[critic_tool_object.critic_tool],
                reflect_on_tool_use=False,
                handoffs=["planner"],
            )

            self.team = Swarm(
                participants=[self.planner, self.critic], termination_condition=termination
            )
        else:
            self.team = RoundRobinGroupChat(
                participants=[self.planner], termination_condition=termination
            )

    async def setup(self):
        await self._initialize_agents()

    async def calculate_total_tokens(self, messages) -> dict:
        """
        Calculates total input (prompt_tokens) and output (completion_tokens) tokens
        from a list of message objects from TaskResult containing `models_usage`.

        Args:
            messages (list): List of message objects, each possibly containing `models_usage`.

        Returns:
            dict: {'total_input': int, 'total_output': int}
        """
        total_input = 0
        total_output = 0

        for message in messages:
            usage = getattr(message, "models_usage", None)
            if usage:
                total_input += getattr(usage, "prompt_tokens", 0) or 0
                total_output += getattr(usage, "completion_tokens", 0) or 0

        return {"total_input": total_input, "total_output": total_output}

    async def run(self):
        """
        Run the video QnA workflow and return structured response.

        Returns:
            Dict containing:
            - result: Parsed response dict with answer, source, and videos
            - tokens: Token usage information
        """
        await self.setup()

        result = await self.team.run(task=self.task)
        tokens = await self.calculate_total_tokens(result.messages)

        # Extract and parse the last message content
        last_message = result.messages[-1].content if result.messages else ""
        parsed_result = parse_response_to_dict(last_message)

        return {"result": parsed_result, "tokens": tokens}

    async def run_stream(self):
        await self.setup()
        return self.team.run_stream(task=self.task)


async def video_qna(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[str, "The unique identifier of the video."] = None,
    url: Annotated[str, "The URL of the video to filter out the search results"] = None,
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent that validates the response."
    ] = True,
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
    provider: VideoAgentProviderConfig = None,
    cache: Annotated[bool, "Set to True to enable cache for model responses."] = True,
):
    """
    Video QnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    This function uses dependency injection via VideoAgentProviderConfig to access all required providers
    for video analysis including LLM, vector databases, embeddings, and storage.

    Answers a user query based on the content of a specified video using five complementary tools:
    1. get_video_summary: Retrieves high-level video summary and context (can be called without video_id for discovery)
    2. get_object_collection: Retrieves object descriptions with counts (requires video_id, semantic query based on summary)
    3. get_context: Retrieves transcript and visual summary documents (requires video_id)
    4. get_relevant_frames: Gets specific frame names based on visual queries
    5. query_frame: Analyzes downloaded frames with vision models

    The planner intelligently combines textual and visual information for comprehensive responses.
    With Swarm orchestration, agents can dynamically hand off tasks for better collaboration.

    Workflow: If video_id not provided, get_video_summary is called first to discover relevant videos.

    Args:
        query (str): The question to be answered based on the content of the video.
        video_id (Optional[str]): The unique identifier of the video.
        url (Optional[str]): The URL of the video to filter out the search results.
        use_critic_agent (bool): Set to True to enable a critic agent that validates the response. Defaults to True.
        stream (bool): Set to True to return the response as a stream. Defaults to False.
        providers (VideoAgentProviderConfig): Provider configuration containing all required providers.
        cache (bool): Set to True to enable cache for model responses. Defaults to True.

    Returns:
        Dict containing:
        - result: Parsed response dict with answer, source, and videos
        - tokens: Token usage information
    """

    video_qna_instance = VideoQnA(
        video_id=video_id,
        url=url,
        query=query,
        use_critic_agent=use_critic_agent,
        provider=provider,
        cache=cache,
    )
    if stream:
        response_generator = await video_qna_instance.run_stream()
        messages = await Console(response_generator)

        # Return the final result in consistent format
        if messages:
            if isinstance(messages, list):
                last_message = messages[-1]
            else:
                last_message = messages
            if isinstance(last_message, TaskResult):
                final_content = last_message.messages[-1].content if last_message.messages else ""
            else:
                final_content = getattr(last_message, "content", str(last_message))

            # Parse the response into structured format
            parsed_result = parse_response_to_dict(final_content)

            # Calculate tokens from all messages
            if isinstance(messages, TaskResult):
                tokens = await video_qna_instance.calculate_total_tokens(
                    last_message.messages if isinstance(last_message, TaskResult) else []
                )
            elif isinstance(messages, list) and messages and isinstance(messages[0], TaskResult):
                tokens = await video_qna_instance.calculate_total_tokens(messages)
            else:
                tokens = await video_qna_instance.calculate_total_tokens(
                    last_message.messages if isinstance(last_message, TaskResult) else []
                )

            return {"result": parsed_result, "tokens": tokens}

        return {
            "result": {"answer": "No response generated", "source": [], "videos": []},
            "tokens": {"total_input": 0, "total_output": 0},
        }
    else:
        return await video_qna_instance.run()


if __name__ == "__main__":
    # Example usage - replace with your actual values
    query = "<placeholder for query>"
    # video_id = "<placeholder for hash video Id>" #Optional
    # url = "<placeholder for url to filter out the results>" #Optional
    use_critic_agent = True
    stream = True

    result = asyncio.run(
        video_qna(
            query=query,
            # video_id=video_id, #Optional
            # url=url, #Optional
            use_critic_agent=use_critic_agent,
            stream=stream,
            cache=False,
        )
    )
