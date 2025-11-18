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
from mmct.video_pipeline.core.tools.get_context import get_context
from mmct.video_pipeline.core.tools.get_relevant_frames import get_relevant_frames
from mmct.video_pipeline.core.tools.query_frame import query_frame
from mmct.video_pipeline.core.tools.get_video_summary import get_video_summary
from mmct.video_pipeline.core.tools.get_object_collection import get_object_collection
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
from mmct.providers.factory import provider_factory

load_dotenv(override=True)


def parse_response_to_dict(content: str) -> Dict[str, Any]:
    """
    Parse the agent response into a standardized dictionary format.
    
    Attempts to extract JSON from the response. If JSON extraction fails,
    creates a structured response from the text content.
    
    Args:
        content: The response content from the agent
        
    Returns:
        Dict containing:
        - answer: The response text (markdown formatted)
        - source: List of sources used (TEXTUAL, VISUAL, or both)
        - videos: List of video metadata with timestamps
    """
    try:
        # Remove TERMINATE keyword
        clean_content = content.replace('TERMINATE', '').strip()
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            parsed_result = json.loads(json_str)
            
            # Validate required keys
            if all(key in parsed_result for key in ['answer', 'source', 'videos']):
                return parsed_result
        
        # Try to find raw JSON object
        json_match = re.search(r'(\{.*\})', clean_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                parsed_result = json.loads(json_str)
                if all(key in parsed_result for key in ['answer', 'source', 'videos']):
                    return parsed_result
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, create structured response from text
        logger.warning("No valid JSON structure found, creating from text content")
        return {
            "answer": clean_content,
            "source": ["TEXTUAL", "VISUAL"],  # Assume both since we can't determine
            "videos": []  # Can't extract video info from unstructured text
        }
        
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return {
            "answer": "Error parsing response",
            "source": [],
            "videos": []
        }


class VideoQnA:
    """
    VideoQnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    MMCT consists of:
    - **Planner Agent**: Has access to five tools for comprehensive video analysis:
      1. get_video_summary: Retrieves high-level video summaries (can be called without video_id for discovery)
      2. get_object_collection: Retrieves object descriptions with counts (requires video_id/url, semantic query based on video summary)
      3. get_context: Retrieves transcript chunks and visual summary chapter documents (requires video_id/url)
      4. get_relevant_frames: Gets specific frame names based on visual queries
      5. query_frame: Analyzes downloaded frames with vision models or fetch frames on filter criteria and then analyze them.
    - **Critic Agent**: Validates or refines the planner's output.

    Workflow:
    1. If video_id/url NOT provided → Call get_video_summary first to discover relevant videos
    2. For object-related queries → Use get_object_collection with video_id (semantic query based on summary)
    3. For narrative/dialogue queries → Use get_context with video_id
    4. For visual verification → Use query_frame

    Args:
        query (str): The natural language question to be answered based on the video content.
        video_id (str): The unique identifier of the video.
        use_critic_agent (bool, optional): Whether to use the critic agent for answer refinement. Defaults to True.
        index_name (str, optional): Vector index name for context retrieval.
    """

    def __init__(
        self,
        query: str,
        video_id:  Optional[str] = None,
        url:  Optional[str] = None,
        use_critic_agent: bool = True,
        index_name: str = None,
        llm_provider: Optional[object] = None,
        cache: bool = True
    ):
        self.query = query
        self.video_id = video_id
        self.use_critic_agent = use_critic_agent
        self.index_name = index_name
        self.url = url
        self.cache = cache

        # Initialize providers if not provided
        if llm_provider is None:
            llm_provider = provider_factory.create_llm_provider()

        self.model_client = llm_provider.get_autogen_client()

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

        self.tools = [get_video_summary, get_object_collection, get_context, get_relevant_frames, query_frame]
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = (
            f"query:{self.query}."
            + (f"\nInstruction:video id:{self.video_id}" if self.video_id is not None else "")
            + (f"\nurl:{self.url}" if self.url is not None else "")
            + (f"\nUse the index name:{self.index_name} to retrieve context.")
        )

    async def _initialize_agents(self):
        # system prompt for video planner agent with comprehensive tool access
        planner_system_prompt = await get_planner_system_prompt(
            use_critic_agent=self.use_critic_agent,
        )

        # Define Planner agent - has access to get_video_summary, get_object_collection, get_context, get_relevant_frames, and query_frame tools
        self.planner = AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_client_stream=False,
            description=PLANNER_DESCRIPTION,
            system_message=(f"""{planner_system_prompt}"""),
            tools=self.tools,
            reflect_on_tool_use=True,
            max_tool_iterations=15,  # Reduced from 100 to 15 for faster response
            handoffs=["critic"] if self.use_critic_agent else []
        )

        text_mention_termination = TextMentionTermination("TERMINATE")
        # max_messages_termination = MaxMessageTermination(max_messages=20)
        termination = text_mention_termination

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
        
        return {
            "result": parsed_result,
            "tokens": tokens
        }

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
    cache: Annotated[bool, "Set to True to enable cache for model responses."] = True
):
    """
    Video QnA with comprehensive multi-tool support for video analysis using Swarm orchestration.

    Answers a user query based on the content of a specified video using five complementary tools:
    1. get_video_summary: Retrieves high-level video summary and context (can be called without video_id for discovery)
    2. get_object_collection: Retrieves object descriptions with counts (requires video_id, semantic query based on summary)
    3. get_context: Retrieves transcript and visual summary documents (requires video_id)
    4. get_relevant_frames: Gets specific frame names based on visual queries
    5. query_frame: Analyzes downloaded frames with vision models

    The planner intelligently combines textual and visual information for comprehensive responses.
    With Swarm orchestration, agents can dynamically hand off tasks for better collaboration.

    Workflow: If video_id not provided, get_video_summary is called first to discover relevant videos.
    """


    video_qna_instance = VideoQnA(
        video_id=video_id,
        url=url,
        query=query,
        use_critic_agent=use_critic_agent,
        index_name=index_name,
        llm_provider=llm_provider,
        cache=cache,
    )
    if stream:
        response_generator = await video_qna_instance.run_stream()
        messages = await Console(response_generator)
        # Stream messages through logger instead of Console
        # messages = []
        # async for message in response_generator:
        #     # Log the message content without the "Agent Message:" prefix
        #     if hasattr(message, 'content') and message.content:
        #         logger.info(f"Agent Message:{message.content}")  # Using : as separator for filtering
        #     messages.append(message)
        
        # Return the final result in consistent format
        if messages:
            if isinstance(messages,list):
                last_message = messages[-1]
            else:
                last_message = messages
            if isinstance(last_message, TaskResult):
                final_content = last_message.messages[-1].content if last_message.messages else ""
            else:
                final_content = getattr(last_message, 'content', str(last_message))
            
            # Parse the response into structured format
            parsed_result = parse_response_to_dict(final_content)
            
            # Calculate tokens from all messages
            tokens = await video_qna_instance.calculate_total_tokens(messages if isinstance(messages[0], TaskResult) else 
                                                                      (last_message.messages if isinstance(last_message, TaskResult) else []))
            
            return {
                "result": parsed_result,
                "tokens": tokens
            }
        
        return {
            "result": {
                "answer": "No response generated",
                "source": [],
                "videos": []
            },
            "tokens": {"total_input": 0, "total_output": 0}
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
    index_name = "<placeholder for index name>"

    result = asyncio.run(
        video_qna(
            query=query,
            #video_id=video_id, #Optional
            # url=url, #Optional
            use_critic_agent=use_critic_agent,
            stream=stream,
            index_name=index_name,
            cache = False
        )
    )