# importing the required files
import asyncio
import json
import re
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated
from loguru import logger
from dotenv import load_dotenv, find_dotenv

from agent_framework import Agent, WorkflowBuilder, Workflow, AgentResponse
from agent_framework._tools import FunctionTool
from mmct.video_pipeline.core.tools.custom_middleware import (
    LoggingAgentMiddleware, 
    LoggingChatMiddleware, 
    LoggingFunctionMiddleware, 
    TerminationMiddleware
)

from mmct.image_pipeline.core.tools.vit import VitTool
from mmct.image_pipeline.core.tools.recog import RecogTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core.tools.ocr import OcrTool
from mmct.image_pipeline.core.tools.critic import CriticTool
from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
    IMAGE_AGENT_SYSTEM_PROMPT,
    ImageAgentResponse,
    TokenInfo
)
from mmct.config.providers import ImageAgentProviderConfig
from mmct.utils.error_handler import ProviderException, ConfigurationException, handle_exceptions

load_dotenv(find_dotenv(), override=True)

# Suppress internal logging

def parse_response_to_dict(content: str) -> Dict[str, Any]:
    """
    Fast JSON extractor for ImageAgent.
    Looks for {"Answer": "..."} format.
    """
    def try_parse_json(s: str):
        try:
            data = json.loads(s)
            if "Answer" in data:
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

        # 2. Fast JSON extraction
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            candidate = clean[start : end + 1]
            parsed = try_parse_json(candidate)
            if parsed:
                return parsed

        # Fallback
        return {"Answer": clean}

    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return {"Answer": "Error parsing response"}

def _extract_last_text(output_events) -> str:
    """Pull the final text content from workflow output events."""
    if not output_events:
        return ""
    if isinstance(output_events, list):
        last = output_events[-1]
    else:
        last = output_events
    return getattr(last, "text", str(last))

class ImageQnaTools(Enum):
    vit = VitTool
    recog = RecogTool
    object_detection = ObjectDetectTool
    ocr = OcrTool

class ImageAgent:
    """
    ImageAgent handles image-based queries using MMCT's modular architecture with a planner agent,
    optional critic agent, and configurable image-processing tools.
    
    Parameters:
    -----------
    image_path (str):
        Local path to the image file.
    query (str):
        Question or instruction related to the image.
    provider (ImageAgentProviderConfig):
        Provider configuration.
    use_critic_agent (bool):
        Enable critic agent for reflective feedback.
    stream (bool, optional):
        Enable streaming response mode. Defaults to False.
    tools (List[ImageQnaTools], optional):
        List of tools to use. Defaults to all available tools.
    disable_console_log (bool, optional):
        Disable console logs. Defaults to False.

    Example Usage:
    --------------
    >>> from mmct.image_pipeline import ImageAgent, ImageQnaTools
    >>> from mmct.config.providers import ImageAgentProviderConfig
    >>> from mmct.providers.azure import AzureLLMProvider
    >>> provider_config = ImageAgentProviderConfig(
    >>> llm_provider = AzureLLMProvider(endpoint = "<endpoint>", api_version = "<api-version>", 
    >>> deployment_name = "<deployment-name>", model_name = "<model-name>", api_key = "api-key"
    >>> ))
    >>> async def run_example():
    >>>     image_qna = ImageAgent(
    >>>         image_path="path/to/image.jpg",
    >>>         query="What dishes are listed under House Special?",
    >>>         provider=provider_config,
    >>>         tools=[ImageQnaTools.ocr, ImageQnaTools.vit],
    >>>         use_critic_agent=True
    >>>     )
    >>>     result = await image_qna()
    >>>     print(result)
    >>> asyncio.run(run_example())
    """

    @staticmethod
    def _wrap(fn) -> FunctionTool:
        """Wrap a callable into a FunctionTool for agent_framework JSON schema."""
        return FunctionTool(name=fn.__name__, description=fn.__doc__ or "", func=fn)

    def __init__(
        self,
        image_path: Annotated[str, "local image path"],
        query: Annotated[str, "query related to image"],
        provider: Annotated[ImageAgentProviderConfig, "Provider configuration for Image Agent"],
        use_critic_agent: Annotated[bool, "Include critic agent"] = True,
        stream: Annotated[bool, "Enable streaming response (True/False)"] = False,
        tools: Annotated[List[ImageQnaTools], "List of tools to use"] = [
            ImageQnaTools.object_detection,
            ImageQnaTools.ocr,
            ImageQnaTools.recog,
            ImageQnaTools.vit,
        ],
        verbose: bool = True,
    ):
        try:
            self.logger = logger
            self.image_path = image_path
            self.query = query
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            self.tools_enum = tools
            self.verbose = verbose
            self.provider = provider
            
            # Use agent_framework client
            self.model_client = self.provider.llm_provider.get_agent_framework_client()

            if self.verbose:
                logger.enable("mmct")
            else:
                logger.disable("mmct")

            # ── Tool instantiation ────────────────────────────────────────────
            self.tools = []
            self.tools_str = [tool.name for tool in self.tools_enum]
            
            for tool in self.tools_enum:
                tool_class = tool.value
                if tool_class == VitTool:
                    tool_instance = tool_class(llm_provider=self.provider.llm_provider, img_path=self.image_path)
                    self.tools.append(self._wrap(tool_instance.vit_tool))
                elif tool_class == RecogTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(self._wrap(tool_instance.recog_tool))
                elif tool_class == ObjectDetectTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(self._wrap(tool_instance.object_detect_tool))
                elif tool_class == OcrTool:
                    tool_instance = tool_class(img_path=self.image_path)
                    self.tools.append(self._wrap(tool_instance.ocr_tool))

            self.task = f"query:{self.query}, image path:{self.image_path}."
            if self.use_critic_agent:
                self.task += "\nAlways criticize the final response if planner asks for review and provide feedback."

            self.workflow: Optional[Workflow] = None
            
        except Exception as e:
            logger.exception(f"Exception occurred while constructing the Image Agent: {e}")
            raise ConfigurationException(f"Failed to initialize ImageAgent: {e}")

    async def _build_workflow(self) -> None:
        """Build the WorkflowBuilder graph."""
        planner_system_prompt = await get_planner_system_prompt(
            tools_string=self.tools_str,
            criticFlag=self.use_critic_agent,
            includeMetaGuidelines=True,
        )

        middleware = []
        if self.verbose:
            middleware.extend([
                LoggingAgentMiddleware(),
                LoggingFunctionMiddleware(),
                LoggingChatMiddleware(),
            ])
        middleware.append(TerminationMiddleware())

        planner = Agent(
            client=self.model_client,
            instructions=planner_system_prompt,
            name="planner",
            tools=self.tools,
            middleware=middleware
        )

        if self.use_critic_agent:
            critic_tool_object = CriticTool(llm_provider=self.provider.llm_provider, query=self.query, img_path=self.image_path)
            critic_prompt = await get_critic_system_prompt(includeMetaGuidelines=True)
            
            critic = Agent(
                client=self.model_client,
                instructions=critic_prompt,
                name="critic",
                tools=[self._wrap(critic_tool_object.critic_tool)],
                middleware=middleware
            )

            def _planner_ready_for_criticism(resp) -> bool:
                messages = (
                    (resp.agent_response.messages if resp.agent_response else None)
                    or resp.full_conversation
                    or []
                )
                if not messages:
                    return False
                last_text = getattr(messages[-1], "text", "") or ""
                return "ready for criticism" in last_text.lower()

            self.workflow = (
                WorkflowBuilder(start_executor=planner)
                .add_edge(planner, critic, condition=_planner_ready_for_criticism)
                .add_edge(critic, planner)
                .build()
            )
        else:
            self.workflow = WorkflowBuilder(start_executor=planner).build()

    async def _format_output(self, qna_result: dict) -> ImageAgentResponse:
        """Structure the final output using LLM."""
        try:
            logger.info("Structuring the Output")
            context_text = str(qna_result.get("result", {}))
            messages = [
                {"role": "system", "content": IMAGE_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {self.query}\nContext: {context_text}"}
            ]

            response_dict = await self.provider.llm_provider.chat_completion(
                messages=messages,
                temperature=0,
                response_format=ImageAgentResponse
            )
            
            final_response = response_dict["content"]
            
            # Aggregate tokens
            qna_tokens = qna_result.get("tokens", {})
            formatting_usage = response_dict.get("usage", {})
            
            # Map k:v from qna_tokens and append formatting_usage
            # agent_framework uses input_token_count / output_token_count
            input_tokens = qna_tokens.get("input_token_count", 0) + formatting_usage.get("prompt_tokens", 0)
            output_tokens = qna_tokens.get("output_token_count", 0) + formatting_usage.get("completion_tokens", 0)
            
            final_response.tokens = TokenInfo(input_token=input_tokens, output_token=output_tokens)
            
            return final_response
            
        except Exception as e:
            logger.exception(f"Output formatting failed: {e}")
            raise ProviderException(f"Output formatting failed: {e}", "OUTPUT_FORMAT_FAILED")

    async def __call__(self) -> ImageAgentResponse:
        """Main execution method."""
        try:
            await self._build_workflow()
            
            if self.stream:
                # User mentioned non-streaming only, but keeping param for compatibility
                # Simply running normally for now as they requested non-streaming
                pass

            events = await self.workflow.run(self.task)
            output_events = events.get_outputs()
            last_content = _extract_last_text(output_events)
            parsed_result = parse_response_to_dict(last_content)

            total_usage = {}
            for event in events:
                if event.type in ["data", "output"]:
                    if isinstance(event.data, AgentResponse) and event.data.usage_details:
                        for k, v in event.data.usage_details.items():
                            if v is not None:
                                total_usage[k] = total_usage.get(k, 0) + v

            qna_result = {"result": parsed_result, "tokens": total_usage}
            return await self._format_output(qna_result)

        except Exception as e:
            logger.exception(f"ImageAgent execution failed: {e}")
            return ImageAgentResponse(
                response=f"ImageAgent execution failed: {str(e)}",
                tokens=TokenInfo(input_token=0, output_token=0)
            )

if __name__ == "__main__":
    async def main():
        print("ImageAgent updated. Run manual validation as requested.")

    asyncio.run(main())