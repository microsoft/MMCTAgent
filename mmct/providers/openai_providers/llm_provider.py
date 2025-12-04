from mmct.providers.base import BaseLLMProvider
from loguru import logger
from mmct.utils.error_handler import ProviderException, ConfigurationException
from typing import Dict, Any, List, Optional
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from openai import AsyncOpenAI, OpenAI
from autogen_ext.models.openai import OpenAIChatCompletionClient


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self,  api_key:str, model_name:str, timeout:Optional[int] = 200, max_retries:Optional[int] = 2):
        if not api_key:
                raise ConfigurationException("OpenAI API key is required!")
        
        if not model_name:
            raise ConfigurationException("OpenAI model name is required!")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = model_name
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            return AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using OpenAI."""
        try:
            temperature = kwargs.get("temperature", self.config.get("llm_temperature", 0.0))
            max_tokens = kwargs.get("max_tokens", 4000)
            response_format = kwargs.get("response_format")
            
            # Remove temperature, max_tokens, and response_format from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "response_format"]}
            
            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self.client.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **filtered_kwargs
                )
                
                return {
                    "content": response.choices[0].message.parsed,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                # Standard completion without structured output
                completion_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **filtered_kwargs
                }
                
                if response_format:
                    completion_kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**completion_kwargs)
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise ProviderException(f"OpenAI chat completion failed: {e}")

    def get_autogen_client(self, **kwargs):
        """Get autogen-compatible client for OpenAI."""
        try:
            temperature = kwargs.get("temperature",0)
            return OpenAIChatCompletionClient(
                api_key=self.api_key,
                timeout=self.timeout,
                model=self.model_name,
                temperature=temperature
            )
        except Exception as e:
            raise ProviderException(f"Failed to create OpenAI autogen client: {e}")

    async def close(self):
        """Close the LLM client and cleanup resources."""
        if self.client:
            logger.info("Closing OpenAI LLM client")
            await self.client.close()
