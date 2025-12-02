from mmct.providers.base import BaseLLMProvider
from loguru import logger
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider
from mmct.utils.error_handler import ProviderException, ConfigurationException
from typing import Dict, Any, List
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.credentials import AzureCredentials
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class AzureReasoningLLMProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider implementation for reasoning models.
    
    This provider is specifically designed for reasoning models (e.g., o1, o3-mini)
    that do not support certain parameters like `temperature`, `top_p`, `presence_penalty`,
    `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, and `max_tokens`.
    
    For non-reasoning models, use `AzureLLMProvider` instead.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            endpoint = self.config.get("llm_endpoint")
            api_version = self.config.get("llm_api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("llm_use_managed_identity", True)
            timeout = self.config.get("llm_timeout", 200)
            max_retries = self.config.get("llm_max_retries", 2)

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
                api_key = self.config.get("llm_api_key")
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
        """Generate chat completion using Azure OpenAI reasoning models.
        
        Note: This method filters out parameters that are not supported by reasoning models:
        - temperature
        - top_p
        - presence_penalty
        - frequency_penalty
        - logprobs
        - top_logprobs
        - logit_bias
        - max_tokens
        """
        try:
            deployment_name = self.config.get("llm_deployment_name")
            if not deployment_name:
                raise ConfigurationException("Azure OpenAI deployment name is required")

            response_format = kwargs.get("response_format")
            
            # List of parameters not supported by reasoning models
            unsupported_params = [
                "temperature", 
                "max_tokens", 
                "top_p", 
                "presence_penalty", 
                "frequency_penalty", 
                "logprobs", 
                "top_logprobs", 
                "logit_bias",
                "response_format"  # Also filter response_format from kwargs
            ]
            
            # Filter out unsupported parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
            
            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self.client.chat.completions.parse(
                    model=deployment_name,
                    messages=messages,
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
            logger.error(f"Azure OpenAI reasoning model chat completion failed: {e}")
            raise ProviderException(f"Azure OpenAI reasoning model chat completion failed: {e}")

    def get_autogen_client(self):
        """Get autogen-compatible client for Azure OpenAI reasoning models.
        
        Note: This method does not pass temperature or other unsupported parameters
        to the autogen client for reasoning models.
        """
        try:
            endpoint = self.config.get("llm_endpoint")
            deployment_name = self.config.get("llm_deployment_name")
            api_version = self.config.get("llm_api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("llm_use_managed_identity", True)
            timeout = self.config.get("llm_timeout", 200)

            if not endpoint or not deployment_name:
                raise ConfigurationException("Azure OpenAI endpoint and deployment name are required for autogen client")

            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential,
                    "https://cognitiveservices.azure.com/.default"
                )
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=timeout
                    # Note: temperature and other unsupported parameters are not passed
                )
            else:
                api_key = self.config.get("llm_api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")

                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    timeout=timeout
                    # Note: temperature and other unsupported parameters are not passed
                )
        except Exception as e:
            raise ProviderException(f"Failed to create Azure OpenAI reasoning model autogen client: {e}")

    async def close(self):
        """Close the LLM client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI Reasoning LLM client")
            await self.client.close()
