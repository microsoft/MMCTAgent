import asyncio
import os
from typing_extensions import Annotated

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult

from mmct.image_pipeline.core.tools.vit import VITTool
from mmct.image_pipeline.core.tools.recog import RECOGTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core.tools.ocr import OCRTool
from mmct.image_pipeline.core.tools.critic import criticTool
from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
)
from mmct.llm_client import LLMClient

from dotenv import load_dotenv
load_dotenv(override=True)


class ImageQnA:
    """
    ImageQnA handles image-based queries using MMCT which includes agents(planner and critic) and tools for tasks like OCR, object detection, and more.
    It can also provide response without critic agent and supports both standard and streaming responses.

    Parameters:
        image_path (str): Path to the image file.
        query (str): Question or prompt related to the image.
        tools_str (str): Comma-separated tool names to use (e.g., 'QueryOCRTool,QueryVITTool,QueryObjectDetectTool,QueryRECOGTool').
        critic_flag (bool): Whether to include a critic agent for review.
        stream (bool, optional): Whether to enable streaming response. Defaults to False.

    Example Usage:
    --------------

    Non-Streaming Response:
    -----------------------
    >>> from mmct.image_pipeline.agents.image_qna import ImageQnA
    >>> import asyncio
    >>> async def run_non_stream():
    >>>     image_qna = ImageQnA(
    >>>         image_path="<image-path>",
    >>>         query="What dishes are available under House Special?",
    >>>         tools_str="QueryVITTool",
    >>>         critic_flag=True,
    >>>         stream=False
    >>>     )
    >>>     result = await image_qna.run()
    >>>     print(result)
    >>> asyncio.run(run_non_stream())

    Streaming Response:
    -------------------
    >>> from mmct.image_pipeline.agents.image_qna import ImageQnA
    >>> from autogen_agentchat.base import TaskResult
    >>> import asyncio
    >>> async def run_stream():
    >>>     image_qna = ImageQnA(
    >>>         image_path="C:/Users/v-amanpatkar/Downloads/menu.png",
    >>>         query=\"""I want to order House Special & Crab Curry. 
    >>>         What will be the order amount? In this same order amount 
    >>>         what are the options that I can consider.\""",
    >>>         tools_str="QueryVITTool,QueryObjectDetectTool",
    >>>         critic_flag=True,
    >>>         stream=True
    >>>     )
    >>>     stream_response = await image_qna.run_stream()
    >>>     async for response in stream_response:
    >>>         if not isinstance(response, TaskResult):
    >>>             print("\\n", response, flush=True)
    >>> asyncio.run(run_stream())
    """
    def __init__(
        self,
        image_path: Annotated[str, "local image path"],
        query: Annotated[str, "query related to image"],
        tools_str: Annotated[str, "comma-separated tools"],
        critic_flag: Annotated[bool, "Include critic agent"],
        stream: Annotated[bool, "Enable streaming response (True/False)"] = False,
    ):
        self.image_path = image_path
        self.query = query
        self.tools_str = tools_str
        self.critic_flag = critic_flag
        self.stream = stream

        service_provider = os.getenv("LLM_PROVIDER", "azure")
        self.model_client = LLMClient(autogen=True, service_provider=service_provider).get_client()

        self.tools_list = []
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

    def _initialize_tools(self):
        tool_map = {
            "QueryObjectDetectTool": ObjectDetectTool,
            "QueryOCRTool": OCRTool,
            "QueryVITTool": VITTool,
            "QueryRECOGTool": RECOGTool,
        }
        self.tools = [tool_map[t] for t in self.tools_str.split(",") if t in tool_map]

    async def _initialize_agents(self):
        planner_prompt = await get_planner_system_prompt(self.tools_str, self.critic_flag)

        self.planner_agent = AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_client_stream=False,
            system_message=planner_prompt,
            tools=self.tools,
            reflect_on_tool_use=True,
        )

        if self.critic_flag:
            critic_prompt = await get_critic_system_prompt(includeMetaGuidelines=True)

            self.critic_agent = AssistantAgent(
                name="critic",
                model_client=self.model_client,
                model_client_stream=False,
                system_message=critic_prompt,
                tools=[criticTool],
                reflect_on_tool_use=False,
            )

            termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

            self.team = SelectorGroupChat(
                [self.planner_agent, self.critic_agent],
                model_client=self.model_client,
                termination_condition=termination,
                allow_repeated_speaker=True,
            )

    async def setup(self):
        self._initialize_tools()
        await self._initialize_agents()

    async def run(self):
        await self.setup()
        task = f"query:{self.query}, image path:{self.image_path}."

        if self.critic_flag:
            task += "\nAlways criticize the final response if planner asks for review and provide feedback."
            result = await self.team.run(task=task)
        else:
            result = await self.planner_agent.run(task=task)

        return result.messages[-1]

    async def run_stream(self):
        await self.setup()
        task = f"query:{self.query}, image path:{self.image_path}."

        if self.critic_flag:
            task += "\nAlways criticize the final response if planner asks for review and provide feedback."
            return self.team.run_stream(task=task)
        else:
            return self.planner_agent.run_stream(task=task)


# =====================
#        ENTRY
# =====================
if __name__ == "__main__":
    image_path = "C:/Users/v-amanpatkar/Downloads/menu.png"
    query = """I want to order House Special & Crab Curry. What will be the order amount?
    In this same order amount what are the options that I can consider."""
    tools = "QueryVITTool,QueryObjectDetectTool"
    critic_flag = True
    stream = True

    async def main():
        image_qna = ImageQnA(image_path=image_path, query=query, tools_str=tools, critic_flag=critic_flag, stream=stream)

        if stream:
            stream_response = await image_qna.run_stream()
            async for response in stream_response:
                if not isinstance(response, TaskResult):
                    print("\n", response, flush=True)
        else:
            result = await image_qna.run()
            print(result)

    asyncio.run(main())
