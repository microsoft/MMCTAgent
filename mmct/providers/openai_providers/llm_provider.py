from loguru import logger
from openai import AsyncOpenAI
from typing import Dict, Any, List
from mmct.providers.base import LLMProvider
from pydantic import BaseModel as PydanticBaseModel
from autogen_ext.models.openai import OpenAIChatCompletionClient
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.config.llm_model_capabilities import MODEL_CAPABILITIES, LLMModelCapabilities

class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using OpenAI."""
        try:
            model_name = self.config.get("model_name")
            
            # Build the request kwargs, filtering out parameters we'll handle separately
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in LLMModelCapabilities.model_fields.keys()
            }

            # Build the API call args
            call_args: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                **filtered_kwargs,
            }

            caps = MODEL_CAPABILITIES.get(model_name)
            if caps is None:
                logger.warning(f"No capabilities defined for model `{model_name}`, using non-reasoning capabilities.")
                caps = MODEL_CAPABILITIES.get("non-reasoning")
            else:
                logger.info(f"Model capabilities set to: {caps}")
            
            for param in LLMModelCapabilities.model_fields.keys():
                # If caps is None, assume full support; otherwise check capability
                if caps is None or getattr(caps, param, False):
                    call_args[param] = kwargs.get(param, self.config.get(param))

            response_format = kwargs.get("response_format")
            if response_format and isinstance(response_format, type) and issubclass(response_format, PydanticBaseModel):
                if 'response_format' in call_args:
                    del call_args['response_format']

                response = await self.client.chat.completions.parse(
                    **call_args,
                    response_format=response_format
                )
                return {
                    "content": response.choices[0].message.parsed,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                response = await self.client.chat.completions.create(**call_args)
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise ProviderException(f"OpenAI chat completion failed: {e}")

    def get_autogen_client(self):
        """Get autogen-compatible client for OpenAI."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required for autogen client")

            model_name = self.config.get("model_name")
            timeout = self.config.get("timeout", 200)

            caps = MODEL_CAPABILITIES.get(model_name)

            if caps is None:
                logger.warning(f"No capabilities defined for model `{model_name}`, using non-reasoning capabilities.")
                caps = MODEL_CAPABILITIES.get("non-reasoning")
            else:
                logger.info(f"Model capabilities set to: {caps}")
            
            # Apply model capabilities using a loop for cleaner code
            model_args: Dict[str, Any] = {}
           
            for param in LLMModelCapabilities.model_fields.keys():
                # If caps is None, assume full support; otherwise check capability
                if caps is None or getattr(caps, param, False):
                    model_args[param] = self.config.get(param)

            return OpenAIChatCompletionClient(
                api_key=api_key,
                timeout=timeout,
                model=model_name,
                **model_args,
            )
        except Exception as e:
            raise ProviderException(f"Failed to create OpenAI autogen client: {e}")

    async def close(self):
        """Close the LLM client and cleanup resources."""
        if self.client:
            logger.info("Closing OpenAI LLM client")
            await self.client.close()
