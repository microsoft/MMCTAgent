# Importing modules
import asyncio
import os
import sys
from loguru import logger
from dotenv import load_dotenv
from typing import Optional

load_dotenv(override=True)

import ast
from enum import Enum
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent

# from mmct.custom_logger import logger as _
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from mmct.video_pipeline.core.tools.get_video_description import (
    get_video_description,
)
from mmct.video_pipeline.core.tools.query_vision_llm import query_vision_llm
from mmct.video_pipeline.core.tools.query_video_description import (
    query_video_description,
)
from mmct.video_pipeline.core.tools.query_frames_computer_vision import (
    query_frames_computer_vision,
)
from mmct.video_pipeline.core.tools.critic import critic_tool
from mmct.video_pipeline.prompts_and_description import (
    get_planner_system_prompt,
    CRITIC_AGENT_SYSTEM_PROMPT,
    PLANNER_DESCRIPTION,
    CRITIC_DESCRIPTION,
)
from mmct.video_pipeline.utils.helper import load_required_files
from mmct.llm_client import LLMClient  # Keep for backward compatibility
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig

class VideoQnaTools(Enum):
    """
    Enum class for tools
    """
    GET_VIDEO_DESCRIPTION = (get_video_description,)
    QUERY_VIDEO_DESCRIPTION = (query_video_description,)
    QUERY_FRAMES_COMPUTER_VISION = (query_frames_computer_vision,)
    QUERY_VISION_LLM = (query_vision_llm,)


class VideoQnA:
    """
    Performs question answering on a specific video using the MMCT (Multi-modal Critical Thinking) framework.

    MMCT consists of:
    - **Planner Agent**: Selects and invokes tools based on the input query and context.
    - **Critic Agent** (optional): Validates or refines the planner's output.

    This class takes a natural language query and a video ID (typically retrieved from an AI search index),
    and applies a set of tools to generate an answer. Tools can include transcript summarization,
    vision-based reasoning, and LLM-based analysis.

    Users can customize which tools are used by passing a dictionary of callables. These tools are
    available in the `core` module of the `video_pipeline` package and can be enabled or disabled as needed.

    Args:
        query (str): The natural language question to be answered based on the video content.
        video_id (str): The unique identifier of the video, typically retrieved from the AI search index.
        use_critic_agent (bool, optional): Whether to use the critic agent for answer refinement. Defaults to True.
        use_computer_vision_tool (bool, optional): Whether to use Computer Vision tools for visual content analysis. Defaults to True.
        tools (dict, optional): A dictionary mapping tool names to their corresponding callable functions.
            This allows fine-grained control over which tools the planner can use. Defaults to:
            - "GET_VIDEO_DESCRIPTION": `get_video_description`
            - "QUERY_VIDEO_DESCRIPTION": `query_video_description`
            - "QUERY_FRAMES_COMPUTER_VISION": `query_frames_computer_vision`
            - "QUERY_VISION_LLM": `query_vision_llm`

    Note:
        To customize tools, import desired tool functions from `video_pipeline.core.tools.<tool_file>` and
        pass them via the `tools` argument.
    """
    def __init__(
        self,
        query,
        video_id,
        use_critic_agent=True,
        use_computer_vision_tool=True,
        tools: dict = {
            "GET_VIDEO_DESCRIPTION": get_video_description,
            "QUERY_VIDEO_DESCRIPTION": query_video_description,
            "QUERY_FRAMES_COMPUTER_VISION": query_frames_computer_vision,
            "QUERY_VISION_LLM": query_vision_llm,
        },
        llm_provider: Optional[object] = None,
        vision_provider: Optional[object] = None,
        transcription_provider: Optional[object] = None,
    ):
        self.tools = tools
        self.query = query
        self.video_id = video_id
        self.use_critic_agent = use_critic_agent
        self.use_computer_vision_tool = use_computer_vision_tool

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

        self.tools_list = []
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = f"query:{self.query}.,\nInstruction:video id:{self.video_id}"

    async def _initialize_tools(self):
        self.tools = list(self.tools.values())

    async def _initialize_agents(self):
        # system prompt for video planner agent
        planner_system_prompt = await get_planner_system_prompt(
            use_computer_vision_tool=self.use_computer_vision_tool,
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
            )

            selector_prompt = """Select an agent to perform task.

                {roles}

                base on the conversation history, select an agent from {participants} to perform the next task.
                critic agent will come only when planner agent mentions `ready for criticism`.
                Only select one agent.
                """
            self.team = SelectorGroupChat(
                [self.planner, self.critic],
                model_client=self.model_client,
                termination_condition=termination,
                allow_repeated_speaker=True,
                #selector_prompt=selector_prompt,
            )
        else:
            self.team = RoundRobinGroupChat(participants=[self.planner], termination_condition=termination)

    async def setup(self):
        await self._initialize_tools()
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
            usage = getattr(message, 'models_usage', None)
            if usage:
                total_input += getattr(usage, 'prompt_tokens', 0) or 0
                total_output += getattr(usage, 'completion_tokens', 0) or 0

        return {
            "total_input": total_input,
            "total_output": total_output
        }

    async def run(self):
        """
        return TaskResult.messages
        """
        await self.setup()

        if self.use_critic_agent:
            result = await self.team.run(task=self.task)
        else:
            result = await self.team.run(task=self.task)
        tokens = await self.calculate_total_tokens(result.messages)
        return {"result":result.messages[-1].content, "tokens":tokens}

    async def run_stream(self):
        logger.info("Initiating the stream")
        await self.setup()
        if self.use_critic_agent:
            return self.team.run_stream(task=self.task)
        else:
            return self.team.run_stream(task=self.task)


async def video_qna(
    query: Annotated[
        str, "The question to be answered based on the content of the video."
    ],
    video_id: Annotated[str, "The unique identifier of the video."],
    tools: Annotated[list, "enum based tools"],
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent that validates the response."
    ] = True,
    use_computer_vision_tool: Annotated[bool, "whether to use computer vision service or not"] = True,
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
    llm_provider: Optional[object] = None,
    vision_provider: Optional[object] = None,
    transcription_provider: Optional[object] = None
):
    """
    Answers a user query based on the content of a specified video.

    This tool combines various analysis components (e.g., transcript summarization, visual understanding)
    to generate accurate responses. It supports optional response validation via a critic agent
    and can operate in both standard and streaming modes.
    """
    tools = {name_: getattr(VideoQnaTools, name_).value[0] for name_ in tools}
    await load_required_files(
        session_id=video_id
    )  # This downloads the required files from blob like frames, transcripts etc.
    video_qna_instance = VideoQnA(
        video_id=video_id,
        query=query,
        use_critic_agent=use_critic_agent,
        tools=tools,
        use_computer_vision_tool=use_computer_vision_tool,
        llm_provider=llm_provider,
        vision_provider=vision_provider,
        transcription_provider=transcription_provider,
    )
    if stream:
        response_generator = await video_qna_instance.run_stream()
        total_input = 0
        total_output = 0
        async for response in response_generator:
            print(response, flush=True)
            if isinstance(response, TaskResult):
                result = response.messages[-1].content
                if result=="TERMINATE":
                    result = response.messages[-2:] # last two messages
                # print("Total Input", total_input)
                # print("Total Output", total_output)
                
                return {"result":result, "tokens":{"total_input":total_input, "total_output":total_output}}  # Returning final response
            if response.models_usage:
                total_input += response.models_usage.prompt_tokens
                total_output += response.models_usage.completion_tokens


    else:
        return await video_qna_instance.run()
    
if __name__=="__main__":
    # Example usage - replace with your actual values
    query = "example question about the video"
    video_id = "example-video-id"
    use_computer_vision_tool = False
    use_critic_agent = True
    stream = False
    tools = [
        VideoQnaTools.GET_VIDEO_DESCRIPTION,
        VideoQnaTools.QUERY_VIDEO_DESCRIPTION,
        VideoQnaTools.QUERY_VISION_LLM]
    if use_computer_vision_tool:
        tools.append(VideoQnaTools.QUERY_FRAMES_COMPUTER_VISION)
        
    tools = [str(tool.name) for tool in tools]
    
    result = asyncio.run(video_qna(query=query, video_id=video_id, use_computer_vision_tool=use_computer_vision_tool, use_critic_agent=use_critic_agent, stream=stream ,tools=tools))
    print(result)
