from mmct.providers.base import LLMProvider
from loguru import logger
from openai import AsyncAzureOpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider
from mmct.utils.error_handler import ProviderException, ConfigurationException
from typing import Dict, Any, List
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.credentials import AzureCredentials


class AzureLLMProvider(LLMProvider):
    """Azure OpenAI LLM provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required")
            
            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential, 
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=max_retries,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")
                
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    max_retries=max_retries,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name")
            if not deployment_name:
                raise ConfigurationException("Azure OpenAI deployment name is required")
            
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.0))
            max_tokens = kwargs.get("max_tokens", 4000)
            response_format = kwargs.get("response_format")
            
            # Remove temperature, max_tokens, and response_format from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "response_format"]}
            
            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self.client.chat.completions.parse(
                    model=deployment_name,
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
                    "model": deployment_name,
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
            logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise ProviderException(f"Azure OpenAI chat completion failed: {e}")

