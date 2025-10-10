# Importing modules
import asyncio
import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Suppress autogen internal logging
from mmct.custom_logger import logger as _
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

from enum import Enum
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import Swarm, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from mmct.video_pipeline.core.tools.get_context import get_context
from mmct.video_pipeline.core.tools.get_relevant_frames import get_relevant_frames
from mmct.video_pipeline.core.tools.query_frame import query_frame

from mmct.video_pipeline.core.tools.critic import critic_tool
from mmct.video_pipeline.prompts_and_description import (
    get_planner_system_prompt,
    CRITIC_AGENT_SYSTEM_PROMPT,
    PLANNER_DESCRIPTION,
    CRITIC_DESCRIPTION,
)

from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache as DiskCache
from mmct.llm_client import LLMClient  # Keep for backward compatibility

load_dotenv(override=True)


class VideoQnaTools(Enum):
    """
    Enum class for tools - planner has access to all three tools for comprehensive video analysis
    """

    GET_CONTEXT = (get_context,)
    GET_RELEVANT_FRAMES = (get_relevant_frames,)
    QUERY_FRAME = (query_frame,)


class VideoQnA:
    """
    VideoQnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    MMCT consists of:
    - **Planner Agent**: Has access to three tools for comprehensive video analysis:
      1. get_context: Retrieves transcript and visual summary documents
      2. get_relevant_frames: Gets specific frame names based on visual queries
      3. query_frame: Analyzes downloaded frames with vision models
    - **Critic Agent**: Validates or refines the planner's output.

    Workflow (with Swarm orchestration):
    1. Planner starts with get_context for transcript/summary information
    2. If more visual content needed, uses get_relevant_frames for frame selection
    3. Uses query_frame to analyze the downloaded frames visually
    4. Combines textual and visual information for comprehensive answers
    5. Can hand off to critic for validation and refinement
    6. Critic can hand back to planner if revisions are needed

    Args:
        query (str): The natural language question to be answered based on the video content.
        video_id (str): The unique identifier of the video.
        use_critic_agent (bool, optional): Whether to use the critic agent for answer refinement. Defaults to True.
        index_name (str, optional): Vector index name for context retrieval. Defaults to "education-video-index-v2".
    """

    def __init__(
        self,
        query: str,
        video_id:  Optional[str] = None,
        url:  Optional[str] = None,
        use_critic_agent: bool = True,
        index_name: str = "education-video-index-v2",
        llm_provider: Optional[object] = None,
        use_graph_rag: bool = False,
        cache: bool = True
    ):
        self.query = query
        self.video_id = video_id
        self.use_critic_agent = use_critic_agent
        self.index_name = index_name
        self.url = url
        self.use_graph_rag = use_graph_rag
        self.cache = cache

        # Initialize providers if not provided
        if llm_provider is None:
            # Fall back to old pattern for backward compatibility
            service_provider = os.getenv("LLM_PROVIDER", "azure")
            self.model_client = LLMClient(
                autogen=True, service_provider=service_provider
            ).get_client()
        else:
            # Use provider pattern - note: this would need autogen client wrapper
            self.model_client = LLMClient(
                autogen=True, service_provider=os.getenv("LLM_PROVIDER", "azure")
            ).get_client()

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

        self.tools = [get_context, get_relevant_frames, query_frame]
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = (
            f"query:{self.query}."
            + (f"\nInstruction:video id:{self.video_id}" if self.video_id is not None else "")
            + (f"\nurl:{self.url}" if self.url is not None else "")
            + (f"\nUse the index name:{self.index_name} to retrieve context.")
            + (f"\n Set the use_graph_rag as True" if self.use_graph_rag else "")
        )

    async def _initialize_agents(self):
        # system prompt for video planner agent with comprehensive tool access
        planner_system_prompt = await get_planner_system_prompt(
            use_critic_agent=self.use_critic_agent,
        )

        # Define Planner agent - has access to get_context, get_relevant_frames, and query_frame tools
        self.planner = AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_client_stream=False,
            description=PLANNER_DESCRIPTION,
            system_message=(f"""{planner_system_prompt}"""),
            tools=self.tools,
            reflect_on_tool_use=True,
            max_tool_iterations = 100,
            handoffs=["critic"] if self.use_critic_agent else []
        )

        text_mention_termination = TextMentionTermination("TERMINATE")
        max_messages_termination = MaxMessageTermination(max_messages=20)
        termination = text_mention_termination | max_messages_termination

        if self.use_critic_agent:
            self.critic = AssistantAgent(
                name="critic",
                model_client=self.model_client,
                model_client_stream=False,
                description=CRITIC_DESCRIPTION,
                system_message=(f"{CRITIC_AGENT_SYSTEM_PROMPT}"),
                tools=[critic_tool],
                reflect_on_tool_use=False,
                handoffs=["planner"]
            )

            self.team = Swarm(
                participants=[self.planner, self.critic],
                termination_condition=termination
            )
        else:
            self.team = RoundRobinGroupChat(
                participants=[self.planner],
                termination_condition=termination
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
        return TaskResult.messages
        """
        await self.setup()

        result = await self.team.run(task=self.task)
        tokens = await self.calculate_total_tokens(result.messages)
        return {"result": result.messages[-1].content, "tokens": tokens}

    async def run_stream(self):
        await self.setup()
        return self.team.run_stream(task=self.task)


async def video_qna(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[str, "The unique identifier of the video."]=None,
    url: Annotated[str, "The URL of the video to filter out the search results"]=None,
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent that validates the response."
    ] = True,
    index_name: Annotated[
        str, "Vector index name for context retrieval"
    ] = "education-video-index-v2",
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
    llm_provider: Optional[object] = None,
    use_graph_rag: Annotated[bool, "Set to True to use GraphRAG for context retrieval."] = False,
    cache: Annotated[bool, "Set to True to enable cache for model responses."] = True
):
    """
    Video QnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    Answers a user query based on the content of a specified video using three complementary tools:
    1. get_context: Retrieves transcript and visual summary documents
    2. get_relevant_frames: Gets specific frame names based on visual queries
    3. query_frame: Analyzes downloaded frames with vision models

    The planner intelligently combines textual and visual information for comprehensive responses.
    With Swarm orchestration, agents can dynamically hand off tasks for better collaboration.
    """


    video_qna_instance = VideoQnA(
        video_id=video_id,
        url=url,
        query=query,
        use_critic_agent=use_critic_agent,
        index_name=index_name,
        use_graph_rag=use_graph_rag,
        llm_provider=llm_provider,
        cache=cache,
    )
    if stream:
        response_generator = await video_qna_instance.run_stream()
        #return response_generator
        messages = await Console(response_generator)
        if isinstance(messages, TaskResult):
            return messages.messages[-1]
        return messages
    else:
        return await video_qna_instance.run()


if __name__ == "__main__":
    # Example usage - replace with your actual values
    query = "<placeholder for query>"
    # video_id = "<placeholder for hash video Id>" #Optional
    # url = "<placeholder for url to filter out the results>" #Optional
    use_critic_agent = True
    stream = True
    index_name = "<placeholder for index name>"

    result = asyncio.run(
        video_qna(
            query=query,
            #video_id=video_id, #Optional
            # url=url, #Optional
            use_critic_agent=use_critic_agent,
            stream=stream,
            index_name=index_name,
            use_graph_rag=False,
            cache = False
        )
    )