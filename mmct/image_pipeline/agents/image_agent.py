# importing the required files
import asyncio
import os
from enum import Enum
from typing_extensions import Annotated, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from mmct.image_pipeline.core.tools.vit import vitTool
from mmct.image_pipeline.core.tools.recog import recogTool
from mmct.image_pipeline.core.tools.object_detect import objectDetectTool
from mmct.image_pipeline.core.tools.ocr import ocrTool
from mmct.image_pipeline.core.tools.critic import criticTool
from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
)
from mmct.llm_client import LLMClient
from mmct.image_pipeline.prompts import IMAGE_AGENT_SYSTEM_PROMPT, ImageAgentResponse
from mmct.custom_logger import log_manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)

class ImageQnaTools(Enum):
    VIT = (vitTool,)
    RECOG = (recogTool,)
    OBJECT_DETECTION = (objectDetectTool,)
    OCR = (ocrTool,)


class ImageAgent:
    """
    ImageAgent handles image-based queries using MMCT's modular architecture that integrates a planner agent,
    an optional critic agent, and a configurable set of image-processing tools (e.g., OCR, object detection, VIT, etc.).    

    Key Features:
    -------------
    - Dynamically utilizes image analysis tools selected via an Enum (`ImageQnaTools`) to answer image-related queries.     
    - Supports both standard and streaming response modes.
    - Offers an optional critic agent for reflective feedback and improved accuracy of responses.

    Parameters:
    -----------
    image_path (str):
        Local path to the image file.
    query (str):
        Question or instruction related to the image.
    use_critic_agent (bool):
        Flag to enable or disable the use of a critic agent.
    stream (bool, optional):
        Enable or disable streaming response mode. Defaults to False.
    tools (List[ImageQnaTools]):
        List of tools to use, defined via the ImageQnaTools Enum. Defaults to all tools (OCR, VIT, Object Detection, Recog).
    disable_console_log (bool):
        Boolean flag to disable console logs. Default set to False.
    Example Usage:
    --------------

    Non-Streaming Response:
    >>> from mmct.image_pipeline import ImageAgent, ImageQnaTools
    >>> async def run_non_stream():
    >>>     image_qna = ImageAgent(
    >>>         image_path="path/to/image.jpg",
    >>>         query="What dishes are listed under House Special?",
    >>>         tools=[ImageQnaTools.OCR, ImageQnaTools.VIT],
    >>>         use_critic_agent=True,
    >>>         stream=False
    >>>     )
    >>>     result = await image_qna.run()
    >>>     print(result)
    >>> asyncio.run(run_non_stream())

    Streaming Response:
    >>> from mmct.image_pipeline import ImageAgent, ImageQnaTools
    >>> async def run_stream():
    >>>     image_qna = ImageAgent(
    >>>         image_path="path/to/image.jpg",
    >>>         query="What is the total price for House Special and Crab Curry?",
    >>>         tools=[ImageQnaTools.VIT, ImageQnaTools.OBJECT_DETECTION],
    >>>         use_critic_agent=True,
    >>>         stream=True
    >>>     )
    >>>     async for response in image_qna.run_stream():
    >>>         if not isinstance(response, TaskResult):
    >>>             print("\\n", response, flush=True)
    >>> asyncio.run(run_stream())
    """

    def __init__(
        self,
        image_path: Annotated[str, "local image path"],
        query: Annotated[str, "query related to image"],
        use_critic_agent: Annotated[bool, "Include critic agent"],
        stream: Annotated[bool, "Enable streaming response (True/False)"] = False,
        tools: Annotated[List[ImageQnaTools], "Enum name and value as Enum value"] = [
            ImageQnaTools.OBJECT_DETECTION,
            ImageQnaTools.OCR,
            ImageQnaTools.RECOG,
            ImageQnaTools.VIT,
        ],
        disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
    ):
        try:
            self.image_path = image_path
            self.query = query
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            self.tools_enum = tools
            self.disable_console_log = disable_console_log
            if disable_console_log==False:
                log_manager.enable_console()
            else:
                log_manager.disable_console()
            self.logger = log_manager.get_logger()
            
            service_provider = os.getenv("LLM_PROVIDER", "azure")
            self.model_client = LLMClient(
                autogen=True, service_provider=service_provider
            ).get_client()
            
            self.openai_client = LLMClient(
                service_provider=service_provider, isAsync=True
            ).get_client()
            
            self.model_name = os.getenv(
                "LLM_VISION_MODEL_NAME"
                if service_provider == "azure"
                else "OPENAI_VISION_MODEL_NAME"
            )
            self.logger.info("Initialized the llm model client")
            self.tools_list = []
            self.planner_agent = None
            self.critic_agent = None
            self.team = None
        except Exception as e:
            self.logger.exception(f"Exception occured while constructing the Image Agent: {e}")
            raise

    async def _initialize_tools(self):
        try:
            self.logger.info("Initializing the tools for Image Agent")
            self.tools = [tool.value[0] for tool in self.tools_enum]
            self.tools_str = [tool.name for tool in self.tools_enum]
            self.logger.info("Successfully initialized tools for Image Agent")
        except Exception as e:
            self.logger.exception(f"Exception occured while initializing the tools for Image Agent: {e}")
            raise

    async def _initialize_agents(self):
        try:
            self.logger.info("Retrieving the Planner Agent's system prompt")
            planner_prompt = await get_planner_system_prompt(
                tools_string=self.tools_str,
                criticFlag=self.use_critic_agent,
                includeMetaGuidelines=True,
            )
            self.planner_agent = AssistantAgent(
                name="ImageAgent_planner",
                model_client=self.model_client,
                model_client_stream=False,
                system_message=planner_prompt,
                tools=self.tools,
                reflect_on_tool_use=True,
            )
            self.logger.info("Initialized the Planner Agent")

            termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
                20
            )  # Termination condition

            if self.use_critic_agent:
                self.logger.info("Retrieving the Critic Agent's System Prompt")
                critic_prompt = await get_critic_system_prompt(includeMetaGuidelines=True)

                self.critic_agent = AssistantAgent(
                    name="ImageAgent_critic",
                    model_client=self.model_client,
                    model_client_stream=False,
                    system_message=critic_prompt,
                    tools=[criticTool],
                    reflect_on_tool_use=False,
                )
                self.logger.info("Initialized the Critic Agent")

                selector_prompt = """Select an agent to perform task.

                {roles}

                Current conversation context:
                {history}

                Read the above conversation, then select an agent from {participants} to perform the next task.
                Make sure 'critic' agent comes only when planner ask for criticism or feedback.
                For your information - There are only two agents - 'planner' & 'critic'
                Only select one agent.
                """
                self.team = SelectorGroupChat(
                    [self.planner_agent, self.critic_agent],
                    model_client=self.model_client,
                    termination_condition=termination,
                    allow_repeated_speaker=True,
                    selector_prompt=selector_prompt,
                )
                self.logger.info("Initialized the both Planner and Critic Agent under SelectorGroupChat")
            else:
                self.team = RoundRobinGroupChat(
                    participants=[self.planner_agent], termination_condition=termination
                )
                self.logger.info("Initialized the Planner Agent under RoundRobinGroupChat")
        except Exception as e:
            self.logger.exception("Exception occured while initializing the Agents for Image Agent.")
            raise

    async def setup(self):
        try:
            await self._initialize_tools()
            await self._initialize_agents()
            self.logger.info("Setup Successfully Completed!")
        except Exception as e:
            self.logger.exception(f"Exception occured while performing setup")
            raise

    async def calculate_total_tokens(self, messages) -> dict:
        """
        Calculates total input (prompt_tokens) and output (completion_tokens) tokens
        from a list of message objects from TaskResult containing `models_usage`.

        Args:
            messages (list): List of message objects, each possibly containing `models_usage`.

        Returns:
            dict: {'total_input': int, 'total_output': int}
        """
        try:
            total_input = 0
            total_output = 0
            self.logger.info("Computing the total token usage")
            for message in messages:
                usage = getattr(message, "models_usage", None)
                if usage:
                    total_input += getattr(usage, "prompt_tokens", 0) or 0
                    total_output += getattr(usage, "completion_tokens", 0) or 0

            return {"total_input": total_input, "total_output": total_output}
        except Exception as e:
            self.logger.exception(f"Exception occured while computing the total token count: {e}")
            raise

    async def run(self):
        try:
            await self.setup()
            task = f"query:{self.query}, image path:{self.image_path}."
            self.logger.info("Initializing the MMCT Image Agentic Flow")
            if self.use_critic_agent:
                task += "\nAlways criticize the final response if planner asks for review and provide feedback."
                result = await self.team.run(task=task)
            else:
                result = await self.team.run(task=task)

            tokens = await self.calculate_total_tokens(result.messages)
            self.logger.info(f"Accumulated the response from the Image Agent:\n{result.messages[-1]}")
            return {"result": result.messages[-1].content, "tokens": tokens}
        except Exception as e:
            self.logger.exception(f"Error occured while executing the MMCT Image Agentic Flow: {e}")
            raise

    async def run_stream(self):
        try:
            await self.setup()
            task = f"query:{self.query}, image path:{self.image_path}."
            self.logger.info("Initializing the MMCT Image Agentic Flow")
            if self.use_critic_agent:
                task += "\nAlways criticize the final response if planner asks for review and provide feedback."
                return self.team.run_stream(task=task)
            else:
                return self.team.run_stream(task=task)
        except Exception as e:
            self.logger.exception(f"Exception occured while streaming the MMCT Image Agentic Flow: {e}")
            raise
    
    async def _format_output(self):
        try:    
            self.logger.info("Structuring the AutoGen Output")
            messages = [
                {"role": "system", "content": IMAGE_AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {self.query}"},
                        {"type": "text", "text": f"Context: {self.result}"},
                    ],
                },
            ]

            response = await self.openai_client.beta.chat.completions.parse(
                model=self.model_name,
                temperature=0,
                messages=messages,
                response_format=ImageAgentResponse,
            )
            response_content: ImageAgentResponse = response.choices[0].message.parsed

            return response_content
        except Exception as e:
            self.logger.exception(f"Exception occured while structuring the output: {e}")
            raise 
        
    async def __call__(self):
        try:
            if self.stream:
                response_generator = await self.run_stream()
                total_input = 0
                total_output = 0
                async for response in response_generator:
                    print(response, flush=True)
                    if isinstance(response, TaskResult):
                        result = response.messages[-1].content
                        if result == "TERMINATE":
                            result = response.messages[-2:]  # last two messages
                        print("Total Input", total_input)
                        print("Total Output", total_output)

                        self.result = {
                            "result": result,
                            "tokens": {
                                "total_input": total_input,
                                "total_output": total_output,
                            },
                        }  # Returning final response
                    else:
                        if response.models_usage:
                            total_input += response.models_usage.prompt_tokens
                            total_output += response.models_usage.completion_tokens
            else:
                result = await self.run()
                self.result = result
            return await self._format_output()
        except Exception as e:
            self.logger.exception(f"Exception occured while executing the MMCT Image Agentic Flow.")
            raise

if __name__ == "__main__":
    image_path = r"C:\Users\v-amanpatkar\Downloads\menu.png"
    query = """I want to order House Special & Crab Curry. What will be the order amount?
    In this same order amount what are the options that I can consider."""
    tools = [
        # ImageQnaTools.OBJECT_DETECTION,
        # ImageQnaTools.OCR,
        # ImageQnaTools.RECOG,
        ImageQnaTools.VIT,
    ]
    use_critic_agent = True
    stream = True

    image_qna = ImageAgent(
            image_path=image_path,
            query=query,
            tools=tools,
            use_critic_agent=use_critic_agent,
            stream=stream,
            # disable_console_log=False
        )
    res = asyncio.run(image_qna())
    print(res)