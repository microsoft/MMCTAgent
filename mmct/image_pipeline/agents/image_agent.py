# importing the required files
import asyncio
from enum import Enum
from typing_extensions import Annotated, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from mmct.image_pipeline.core.tools.vit import vit_tool
from mmct.image_pipeline.core.tools.recog import recog_tool
from mmct.image_pipeline.core.tools.object_detect import object_detect_tool
from mmct.image_pipeline.core.tools.ocr import ocr_tool
from mmct.image_pipeline.core.tools.critic import critic_tool
from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
)
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.utils.error_handler import handle_exceptions
from mmct.image_pipeline.prompts import IMAGE_AGENT_SYSTEM_PROMPT, ImageAgentResponse
from loguru import logger
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)

class ImageQnaTools(Enum):
    vit = (vit_tool,)
    recog = (recog_tool,)
    object_detection = (object_detect_tool,)
    ocr = (ocr_tool,)


class ImageAgent:
    """
    ImageAgent handles image-based queries using MMCT's modular architecture that integrates a planner agent,
    an optional critic agent, and a configurable set of image-processing tools (e.g., ocr, object detection, vit, etc.).    

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
        List of tools to use, defined via the ImageQnaTools Enum. Defaults to all tools (ocr, vit, object detection, recog).
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
    >>>         tools=[ImageQnaTools.ocr, ImageQnaTools.vit],
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
    >>>         tools=[ImageQnaTools.vit, ImageQnaTools.object_detection],
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
            ImageQnaTools.object_detection,
            ImageQnaTools.ocr,
            ImageQnaTools.recog,
            ImageQnaTools.vit,
        ],
        disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
    ):
        try:
            # Initialize configuration
            self.config = MMCTConfig()
            
            # Initialize logger for this instance
            self.logger = logger
            
            # Initialize providers
            self.llm_provider = provider_factory.create_llm_provider(
                self.config.llm.provider,
                self.config.llm.model_dump()
            )
            self.vision_provider = provider_factory.create_vision_provider(
                self.config.llm.provider,  # Use same provider as LLM for vision
                self.config.llm.model_dump()
            )
            
            # Set instance attributes
            self.image_path = image_path
            self.query = query
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            self.tools_enum = tools
            self.disable_console_log = disable_console_log
            
            # Configure console logging
            if not disable_console_log:
                logger.enable("mmct")
            else:
                logger.disable("mmct")
            
            # Initialize client components for autogen compatibility
            from mmct.llm_client import LLMClient
            service_provider = self.config.llm.provider
            self.model_client = LLMClient(
                autogen=True, service_provider=service_provider
            ).get_client()
            
            self.openai_client = LLMClient(
                service_provider=service_provider, isAsync=True
            ).get_client()
            
            self.model_name = self.config.llm.model_name
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
        """
        Initialize the tools for Image Agent.
        
        Raises:
            ProviderException: If tool initialization fails
        """
        try:
            logger.info("Initializing the tools for Image Agent")
            self.tools = [tool.value[0] for tool in self.tools_enum]
            self.tools_str = [tool.name for tool in self.tools_enum]
            logger.info("Successfully initialized tools for Image Agent")
        except Exception as e:
            logger.exception(f"Exception occurred while initializing the tools for Image Agent: {e}")
            raise ProviderException(f"Tool initialization failed: {e}", "TOOL_INIT_FAILED")

    @handle_exceptions(retries=2)
    async def _initialize_agents(self):
        """
        Initialize the agents for Image Agent.
        
        Raises:
            ProviderException: If agent initialization fails
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

                self.critic_agent = AssistantAgent(
                    name="ImageAgent_critic",
                    model_client=self.model_client,
                    model_client_stream=False,
                    system_message=critic_prompt,
                    tools=[critic_tool],
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
        """
        Setup the ImageAgent by initializing tools and agents.
        
        Raises:
            ProviderException: If setup fails
        """
        try:
            await self._initialize_tools()
            await self._initialize_agents()
            logger.info("Setup Successfully Completed!")
        except Exception as e:
            logger.exception(f"Exception occurred while performing setup")
            raise ProviderException(f"Setup failed: {e}", "SETUP_FAILED")

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

    @handle_exceptions(retries=2)
    async def run(self):
        """
        Execute the ImageAgent workflow.
        
        Returns:
            Dictionary containing result and token usage
            
        Raises:
            ProviderException: If execution fails
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
        """
        Execute the ImageAgent workflow in streaming mode.
        
        Returns:
            Async generator for streaming responses
            
        Raises:
            ProviderException: If execution fails
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
        """
        Format the output using the LLM provider.
        
        Returns:
            Formatted ImageAgentResponse
            
        Raises:
            ProviderException: If output formatting fails
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
                temperature=self.config.llm.temperature,
                response_format=ImageAgentResponse
            )

            return response
        except Exception as e:
            logger.exception(f"Exception occurred while structuring the output: {e}")
            raise ProviderException(f"Output formatting failed: {e}", "OUTPUT_FORMAT_FAILED") 
        
    @handle_exceptions(retries=2)
    async def __call__(self):
        """
        Main execution method for the ImageAgent.
        
        Returns:
            Formatted ImageAgentResponse
            
        Raises:
            ProviderException: If execution fails
        """
        try:
            if self.stream:
                response_generator = await self.run_stream()
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