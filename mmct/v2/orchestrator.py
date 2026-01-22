import asyncio
import os
import json
import time
from datetime import datetime
from loguru import logger
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core.model_context import BufferedChatCompletionContext


from mmct.v2.agents.video_agent import VideoAgent
from mmct.v2.agents.image_agent import ImageAgent
from mmct.v2.agents.planner import PlannerAgent
from mmct.v2.agents.critic import CriticAgent
from mmct.v2.stream_handlers import (
    console_stream_generator,
    dict_stream_generator,
    dict_stream_generator_with_console,
)
from mmct.config.providers import VideoAgentProviderConfig, ImageAgentProviderConfig

# Buffer sizes for agent context windows to prevent context explosion
# during multi-turn Planner-Critic loops
PLANNER_BUFFER_SIZE = 15  # Needs more context for orchestration decisions
CRITIC_BUFFER_SIZE = 10   # Just needs recent draft + conversation
VIDEO_AGENT_BUFFER_SIZE = 12  # Needs context for tool selection
IMAGE_AGENT_BUFFER_SIZE = 10  # Needs context for image analysis

_TERMINATE_STRING = "TERMINATE"

async def process_query_v2(
    query: str,
    video_provider: VideoAgentProviderConfig,
    image_provider: ImageAgentProviderConfig,
    video_id: str = None,
    url: str = None,
    image_path: str = None, # Optional, if query starts with an image interaction
    use_critic: bool = True,
    stream: bool = False,
    cache: bool = False,
    use_console: bool = True
):
    """
    Orchestrates the V2 multi-agent flow.
    """
    
    # 1. Initialize Model Client (using video provider's LLM for planner/critic as main)
    # Assuming video provider has the main LLM config we want to use for orchestration
    model_client = video_provider.llm_provider.get_autogen_client()

    # 2. Initialize Agents with BufferedChatCompletionContext to prevent context explosion
    video_agent_wrapper = VideoAgent(
        video_provider, 
        model_client,
        model_context=BufferedChatCompletionContext(buffer_size=VIDEO_AGENT_BUFFER_SIZE)
    )
    image_agent_wrapper = ImageAgent(
        image_provider, 
        model_client, 
        storage_provider=video_provider.storage_provider,
        model_context=BufferedChatCompletionContext(buffer_size=IMAGE_AGENT_BUFFER_SIZE)
    )
    planner_wrapper = PlannerAgent(
        model_client, 
        use_critic=use_critic,
        model_context=BufferedChatCompletionContext(buffer_size=PLANNER_BUFFER_SIZE)
    )
    
    participants = [planner_wrapper.agent, video_agent_wrapper.agent, image_agent_wrapper.agent]
    
    if use_critic:
        critic_wrapper = CriticAgent(
            video_provider, 
            model_client,
            model_context=BufferedChatCompletionContext(buffer_size=CRITIC_BUFFER_SIZE)
        )
        participants.append(critic_wrapper.agent)

    termination = TextMentionTermination(_TERMINATE_STRING)

    team = Swarm(
        participants=participants,
        termination_condition=termination,
        max_turns=20  # Safety limit to prevent infinite loops
    )

    # 5. Define Task
    task_context = f"Query: {query}"
    if video_id:
        task_context += f"\nVideo ID: {video_id}"
    if url:
        task_context += f"\nVideo URL: {url}"
    if image_path:
        task_context += f"\nInitial Image Path: {image_path}"
        
    logger.info(f"Starting V2 Query Processing for: {query}")

    async def _stream_with_cleanup(stream_gen, cleanup_fn):
        """Wrapper that calls cleanup after stream completes."""
        try:
            async for message in stream_gen:
                yield message
        finally:
            cleanup_fn()

    if stream:
        stream_gen = team.run_stream(task=task_context)
        if use_console:
            # Print to console AND return dict stream
            return _stream_with_cleanup(dict_stream_generator_with_console(stream_gen), image_agent_wrapper.cleanup)
        return _stream_with_cleanup(dict_stream_generator(stream_gen), image_agent_wrapper.cleanup)

    if use_console:
        stream_gen = team.run_stream(task=task_context)
        final_result = None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        async for message in console_stream_generator(stream_gen):
            if hasattr(message, "models_usage") and message.models_usage:
                total_prompt_tokens += message.models_usage.prompt_tokens
                total_completion_tokens += message.models_usage.completion_tokens
            if isinstance(message, TaskResult):
                final_result = message
        
        # Cleanup downloaded frames
        image_agent_wrapper.cleanup()
        
        if final_result:
            return {
                "content": final_result.messages[-1].content.strip(_TERMINATE_STRING),
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens
                }
            }
        return None
    
    result = await team.run(task=task_context)
    
    # Cleanup downloaded frames
    image_agent_wrapper.cleanup()
    
    # Calculate token usage from all messages
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for msg in result.messages:
        if hasattr(msg, "models_usage") and msg.models_usage:
            total_prompt_tokens += msg.models_usage.prompt_tokens
            total_completion_tokens += msg.models_usage.completion_tokens
    
    return {
        "content": result.messages[-1].content.strip(_TERMINATE_STRING),
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    }


