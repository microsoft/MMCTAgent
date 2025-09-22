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
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
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
    VideoQnA with comprehensive multi-tool support for video analysis.

    MMCT consists of:
    - **Planner Agent**: Has access to three tools for comprehensive video analysis:
      1. get_context: Retrieves transcript and visual summary documents
      2. get_relevant_frames: Gets specific frame names based on visual queries  
      3. query_frame: Analyzes downloaded frames with vision models
    - **Critic Agent**: Validates or refines the planner's output.

    Workflow:
    1. Planner starts with get_context for transcript/summary information
    2. If more visual content needed, uses get_relevant_frames for frame selection
    3. Uses query_frame to analyze the downloaded frames visually
    4. Combines textual and visual information for comprehensive answers

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
        youtube_url:  Optional[str] = None,
        use_critic_agent: bool = True,
        index_name: str = "education-video-index-v2",
        llm_provider: Optional[object] = None,
    ):
        self.query = query
        self.video_id = video_id
        self.use_critic_agent = use_critic_agent
        self.index_name = index_name
        self.youtube_url = youtube_url

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

        self.tools = [get_context, get_relevant_frames, query_frame]
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = (
            f"query:{self.query}."
            + (f"\nInstruction:video id:{self.video_id}" if self.video_id is not None else "")
            + (f"\nyoutube url:{self.youtube_url}" if self.youtube_url is not None else "")
            + (f"\nUse the index name:{self.index_name} to retrieve context.")
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
            max_tool_iterations = 100
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

            self.team = SelectorGroupChat(
                [self.planner, self.critic],
                model_client=self.model_client,
                termination_condition=termination,
                allow_repeated_speaker=True,
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
        return TaskResult.messages
        """
        await self.setup()

        if self.use_critic_agent:
            result = await self.team.run(task=self.task)
        else:
            result = await self.team.run(task=self.task)
        tokens = await self.calculate_total_tokens(result.messages)
        return {"result": result.messages[-1].content, "tokens": tokens}

    async def run_stream(self):
        await self.setup()
        if self.use_critic_agent:
            return self.team.run_stream(task=self.task)
        else:
            return self.team.run_stream(task=self.task)


async def video_qna(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[str, "The unique identifier of the video."]=None,
    youtube_url: Annotated[str, "The YouTube URL of the video."]=None,
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent that validates the response."
    ] = True,
    index_name: Annotated[
        str, "Vector index name for context retrieval"
    ] = "education-video-index-v2",
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
    llm_provider: Optional[object] = None,
):
    """
    Video QnA with comprehensive multi-tool support for video analysis.

    Answers a user query based on the content of a specified video using three complementary tools:
    1. get_context: Retrieves transcript and visual summary documents
    2. get_relevant_frames: Gets specific frame names based on visual queries
    3. query_frame: Analyzes downloaded frames with vision models
    
    The planner intelligently combines textual and visual information for comprehensive responses.
    """
   

    video_qna_instance = VideoQnA(
        video_id=video_id,
        youtube_url=youtube_url,
        query=query,
        use_critic_agent=use_critic_agent,
        index_name=index_name,
        llm_provider=llm_provider,
    )
    if stream:
        response_generator = await video_qna_instance.run_stream()
        messages = await Console(response_generator)
        if isinstance(messages, TaskResult):
            return messages.messages[-1]
        return messages
    else:
        return await video_qna_instance.run()


if __name__ == "__main__":
    # Example usage - replace with your actual values
    query = "I am seeking clarification on the definition of skew. I attempted to use the formula presented in class, which is defined as skew(node) = height(node.right) - height(node.left). For example, given a tree, I calculated the skew using this formula and obtained the following results: for node Z, the height is 2, yielding a skew of 1 - 0 = 1; for node Y, the height is 1, resulting in a skew of 0 - 0 = 0; and for node X, the height is 0, leading to a skew of 0 - 0 = 0. This seems to suggest that the tree is balanced, as all nodes have a skew within the range {-1, 0, 1}. However, I intuitively believe that this tree should be considered unbalanced, skewed to the right by 2. Based on my observations, I would expect the skews to be calculated as follows: for Z, the height is 2, yielding a skew of 2; for Y, the height is 1, resulting in a skew of 1; and for X, the height remains 0, yielding a skew of 0. Which interpretation is correct? If the second interpretation holds merit, I am uncertain about the correct formula or what aspect of my calculations may be flawed?  what polygon does the instructure used to inclose the example tree? what is tutor having in end when explaining the example?"
    
    #video_id = "b66b839dca7a702429999dbe341a1043c987da554fa4960d339dbd478f29f101B"
    #youtube_url = "https://youtube.com/watch?v=U1JYwHcFfso"
    use_critic_agent = True
    stream = True
    index_name = "education-video-index-v2"

    result = asyncio.run(
        video_qna(
            query=query,
            #video_id=video_id,
            #youtube_url=youtube_url,
            use_critic_agent=use_critic_agent,
            stream=stream,
            index_name=index_name,
        )
    )
    
