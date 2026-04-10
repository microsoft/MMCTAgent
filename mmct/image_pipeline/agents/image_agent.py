# importing the required files
import asyncio
from enum import Enum
from typing import Any, List, Union, AsyncGenerator
from typing_extensions import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from mmct.image_pipeline.core.tools.vit import VitTool
from mmct.image_pipeline.core.tools.recog import RecogTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core.tools.ocr import OcrTool
from mmct.image_pipeline.core.tools.critic import CriticTool
from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
)
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.utils.error_handler import handle_exceptions
from mmct.image_pipeline.prompts import IMAGE_AGENT_SYSTEM_PROMPT, ImageAgentResponse
from mmct.image_pipeline.config import ImageAgentProviderConfig
from loguru import logger
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)

class ImageQnaTools(str, Enum):
    vit = "vit"
    recog = "recog"
    object_detection = "object_detection"
    ocr = "ocr"


class ImageAgent:
    """Handles image-based queries using MMCT's modular architecture.

    The ImageAgent orchestrates a multi-agent workflow consisting of a planner agent,
    an optional critic agent, and a set of configurable image-processing tools. It
    supports both standard and streaming execution modes.

    Attributes:
        image_path (str): Local path to the image file to be analyzed.
        query (str): The natural language question or instruction related to the image.
        provider (ImageAgentProviderConfig): Configuration for the underlying LLM provider.
        use_critic_agent (bool): Whether to include a critic agent for reflective feedback.
        stream (bool): Whether to enable streaming response mode.
        tools_enum (List[ImageQnaTools]): List of specific tools to enable for analysis.
        disable_console_log (bool): Flag to disable console-based logging for this agent.
        use_console (bool): Whether to use the Console UI for output display.

    Example:
        >>> from mmct.image_pipeline import ImageAgent, ImageQnaTools
        >>> from mmct.image_pipeline.config import ImageAgentProviderConfig
        >>> from mmct.providers.azure import AzureLLMProvider
        >>> 
        >>> provider_config = ImageAgentProviderConfig(
        ...     llm_provider=AzureLLMProvider(
        ...         endpoint="<endpoint>", 
        ...         api_version="<api-version>", 
        ...         deployment_name="<deployment-name>", 
        ...         model_name="<model-name>", 
        ...         api_key="api-key"
        ...     )
        ... )
        >>> 
        >>> async def run_example():
        ...     agent = ImageAgent(
        ...         image_path="path/to/image.jpg",
        ...         query="What dishes are listed under House Special?",
        ...         provider=provider_config,
        ...         tools=[ImageQnaTools.ocr, ImageQnaTools.vit],
        ...         use_critic_agent=True
        ...     )
        ...     result = await agent()
        ...     print(result)
    """

    def __init__(
        self,
        image_path: Annotated[str, "local image path"],
        query: Annotated[str, "query related to image"],
        provider: Annotated[ImageAgentProviderConfig, "Provider configuration for Image Agent"],
        use_critic_agent: Annotated[bool, "Include critic agent"],
        stream: Annotated[bool, "Enable streaming response (True/False)"] = False,
        tools: Annotated[List[ImageQnaTools], "Enum name and value as Enum value"] = [
            ImageQnaTools.object_detection,
            ImageQnaTools.ocr,
            ImageQnaTools.recog,
            ImageQnaTools.vit,
        ],
        disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False,
        use_console: Annotated[bool, "Use Console for output display"] = True
    ):
        try:
            # Initialize logger for this instance
            self.logger = logger
            
            # Initialize providers
            self.llm_provider = provider.llm_provider
            
            # Set instance attributes
            self.image_path = image_path
            self.query = query
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            self.tools_enum = tools
            self.disable_console_log = disable_console_log
            self.use_console = use_console
            
            # Configure console logging
            if not disable_console_log:
                logger.enable("mmct")
            else:
                logger.disable("mmct")
            
            # Initialize client components using providers
            self.model_client = self.llm_provider.get_autogen_client()

            logger.info("Initialized ImageAgent with provider system")

            self.tools_list = []
            self.planner_agent = None
            self.critic_agent = None
            self.team = None
            
        except Exception as e:
            logger.exception(f"Exception occurred while constructing the Image Agent: {e}")
            raise ConfigurationException(f"Failed to initialize ImageAgent: {e}")

    @handle_exceptions(retries=2)
    async def _initialize_tools(self):
        """Initializes the configured vision tools for the Image Agent.

        This method maps requested tool enums to their respective classes,
        instantiates them with the required context, and registers their functional
        entry points for the planner agent.

        Raises:
            ProviderException: If any tool fails to initialize or if an unknown
                tool is requested.
        """
        try:
            logger.info("Initializing the tools for Image Agent")
            self.tools = []
            self.tools_str = [tool.value for tool in self.tools_enum]

            # Map enum members to their respective tool classes
            tool_mapping = {
                ImageQnaTools.vit: VitTool,
                ImageQnaTools.recog: RecogTool,
                ImageQnaTools.object_detection: ObjectDetectTool,
                ImageQnaTools.ocr: OcrTool,
            }

            # Instantiate each tool class and get the method reference
            for tool in self.tools_enum:
                tool_class = tool_mapping.get(tool)
                if not tool_class:
                    logger.warning(f"Unknown tool requested: {tool}")
                    continue

                # Instantiate based on tool type
                if tool_class == VitTool:
                    tool_instance = tool_class(llm_provider=self.llm_provider, img_path=self.image_path)
                    self.tools.append(tool_instance.vit_tool)
                elif tool_class == RecogTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(tool_instance.recog_tool)
                elif tool_class == ObjectDetectTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(tool_instance.object_detect_tool)
                elif tool_class == OcrTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(tool_instance.ocr_tool)

            logger.info("Successfully initialized tools for Image Agent")
        except Exception as e:
            logger.exception(f"Exception occurred while initializing the tools for Image Agent: {e}")
            raise ProviderException(f"Tool initialization failed: {e}", "TOOL_INIT_FAILED")

    @handle_exceptions(retries=2)
    async def _initialize_agents(self):
        """Initializes the Planner and optional Critic agents.

        Configures the AutoGen agents with appropriate system prompts, toolsets,
        and termination conditions. If a critic agent is enabled, it sets up a
        SelectorGroupChat; otherwise, it uses a RoundRobinGroupChat.

        Raises:
            ProviderException: If agent or team configuration fails.
        """
        try:
            logger.info("Retrieving the Planner Agent's system prompt")
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
            logger.info("Initialized the Planner Agent")

            termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
                20
            )  # Termination condition

            if self.use_critic_agent:
                logger.info("Retrieving the Critic Agent's System Prompt")
                critic_prompt = await get_critic_system_prompt(includeMetaGuidelines=True)

                critic_tool_object = CriticTool(llm_provider = self.llm_provider, query=self.query, img_path=self.image_path)

                self.critic_agent = AssistantAgent(
                    name="ImageAgent_critic",
                    model_client=self.model_client,
                    model_client_stream=False,
                    system_message=critic_prompt,
                    tools=[critic_tool_object.critic_tool],
                    reflect_on_tool_use=False,
                )
                logger.info("Initialized the Critic Agent")

                selector_prompt = """Select an agent to perform task.

                {roles}

                Current conversation context:
                {history}

                Read the above conversation, then select an agent from {participants} to perform the next task.
                Make sure 'critic' agent comes only when planner ask for criticism or feedback.
                For your information - There are only two agents - 'planner' & 'critic'
                Only select one agent.

                - Limit the Planner–Critic feedback loop to **maximum 2 rounds**.
                """
                self.team = SelectorGroupChat(
                    [self.planner_agent, self.critic_agent],
                    model_client=self.model_client,
                    termination_condition=termination,
                    allow_repeated_speaker=True,
                    selector_prompt=selector_prompt,
                )
                logger.info("Initialized the both Planner and Critic Agent under SelectorGroupChat")
            else:
                self.team = RoundRobinGroupChat(
                    participants=[self.planner_agent], termination_condition=termination
                )
                logger.info("Initialized the Planner Agent under RoundRobinGroupChat")
        except Exception as e:
            logger.exception("Exception occurred while initializing the Agents for Image Agent.")
            raise ProviderException(f"Agent initialization failed: {e}", "AGENT_INIT_FAILED")

    @handle_exceptions(retries=2)
    async def setup(self):
        """Prepares the ImageAgent for execution.

        Initializes tools and agents required for the multi-agent workflow.

        Raises:
            ProviderException: If tool or agent initialization fails.
        """
        try:
            await self._initialize_tools()
            await self._initialize_agents()
            logger.info("Setup Successfully Completed!")
        except Exception as e:
            logger.exception(f"Exception occurred while performing setup")
            raise ProviderException(f"Setup failed: {e}", "SETUP_FAILED")

    async def calculate_total_tokens(self, messages: List[Any]) -> dict:
        """Calculates accumulated token usage from an execution result.

        Args:
            messages: A list of message objects from a TaskResult, each
                optionally containing `models_usage` metadata.

        Returns:
            dict: A dictionary containing 'total_input' and 'total_output' counts.
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

    @handle_exceptions(retries=2)
    async def run(self) -> dict:
        """Executes the standard ImageAgent reasoning workflow.

        Sets up the environment, runs the group chat team on the query,
        summarizes results, and calculates total token usage.

        Returns:
            dict: result content and token usage counts.

        Raises:
            ProviderException: If the execution or token calculation fails.
        """
        try:
            await self.setup()
            task = f"query:{self.query}, image path:{self.image_path}."
            logger.info("Initializing the MMCT Image Agentic Flow")
            if self.use_critic_agent:
                task += "\nAlways criticize the final response if planner asks for review and provide feedback."
                result = await self.team.run(task=task)
            else:
                result = await self.team.run(task=task)

            tokens = await self.calculate_total_tokens(result.messages)
            logger.info(f"Accumulated the response from the Image Agent: {result.messages[-1]}")
            return {"result": result.messages[-1].content, "tokens": tokens}
        except Exception as e:
            logger.exception(f"Error occurred while executing the MMCT Image Agentic Flow: {e}")
            raise ProviderException(f"ImageAgent execution failed: {e}", "AGENT_EXECUTION_FAILED")

    @handle_exceptions(retries=2)
    async def run_stream(self):
        """Executes the agentic workflow in streaming mode.

        Returns:
            AsyncGenerator: An asynchronous generator yielding chunks of the
                reasoning process and final result.

        Raises:
            ProviderException: If the streaming execution fails.
        """
        try:
            await self.setup()
            task = f"query:{self.query}, image path:{self.image_path}."
            logger.info("Initializing the MMCT Image Agentic Flow")
            if self.use_critic_agent:
                task += "\nAlways criticize the final response if planner asks for review and provide feedback."
                return self.team.run_stream(task=task)
            else:
                return self.team.run_stream(task=task)
        except Exception as e:
            logger.exception(f"Exception occurred while streaming the MMCT Image Agentic Flow: {e}")
            raise ProviderException(f"ImageAgent streaming failed: {e}", "AGENT_STREAMING_FAILED")
    
    @handle_exceptions(retries=2)
    async def _format_output(self):
        """Synthesizes the raw agent output into a structured response model.

        Uses the LLM provider to process the conversation context and extract
        the final answer into an `ImageAgentResponse` Pydantic model.

        Returns:
            ImageAgentResponse: The structured answer with token usage metadata.

        Raises:
            ProviderException: If the output structuring or LLM call fails.
        """
        try:    
            logger.info("Structuring the AutoGen Output")
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

            # Use the provider system for LLM completion
            response = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=0,
                response_format=ImageAgentResponse
            )

            return response["content"]
        except Exception as e:
            logger.exception(f"Exception occurred while structuring the output: {e}")
            raise ProviderException(f"Output formatting failed: {e}", "OUTPUT_FORMAT_FAILED") 
        
    @handle_exceptions(retries=2)
    async def __call__(self) -> Union[ImageAgentResponse, AsyncGenerator[Any, None]]:
        """Main execution entry point for the ImageAgent.

        Depending on `self.stream`, it either runs the full synchronous workflow
        or returns an asynchronous generator for streaming updates.

        Returns:
            Union[ImageAgentResponse, AsyncGenerator]: Final structured response 
                or a stream generator.

        Raises:
            ProviderException: If internal agent execution or formatting fails.
        """
        try:
            if self.stream:
                response_generator = await self.run_stream()
                if not self.use_console:
                    return response_generator
                    
                self.result = await Console(response_generator)
                if isinstance(self.result,TaskResult):
                    self.result = self.result.messages[-1]
            else:
                result = await self.run()
                self.result = result
            return await self._format_output()
        except Exception as e:
            logger.exception(f"Exception occurred while executing the MMCT Image Agentic Flow.")
            raise ProviderException(f"ImageAgent execution failed: {e}", "AGENT_CALL_FAILED")

if __name__ == "__main__":
    # Example usage - replace with your actual values
    image_path = "path/to/your/image.png"
    query = "example question about the image"
    tools = [
        # ImageQnaTools.object_detection,
        # ImageQnaTools.ocr,
        # ImageQnaTools.recog,
        ImageQnaTools.vit,
    ]
    use_critic_agent = True
    stream = True
    use_console = True  # Enable console for local run

    image_qna = ImageAgent(
            image_path=image_path,
            query=query,
            tools=tools,
            use_critic_agent=use_critic_agent,
            stream=stream,
            use_console=use_console,
            # disable_console_log=False
        )
    
    if stream and not use_console:
        async def iterate_stream():
            stream_gen = await image_qna()
            async for chunk in stream_gen:
                print(chunk)
        asyncio.run(iterate_stream())
    else:
        res = asyncio.run(image_qna())
        print(res)