from loguru import logger
from typing import Dict, Any, List
from openai import AsyncAzureOpenAI
from mmct.providers.base import LLMProvider
from pydantic import BaseModel as PydanticBaseModel
from azure.identity import get_bearer_token_provider
from mmct.providers.credentials import AzureCredentials
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from mmct.config.llm_model_capabilities import MODEL_CAPABILITIES, LLMModelCapabilities

class AzureLLMProvider(LLMProvider):
    """Azure OpenAI LLM provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
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

            # Build the request kwargs
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in LLMModelCapabilities.model_fields.keys()
            }

            # Build the API call args
            call_args: Dict[str, Any] = {
                "model": deployment_name,
                "messages": messages,
                **filtered_kwargs,
            }

            model_name = self.config.get("model_name")  # assuming you store model name in config
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

            response_format = kwargs.pop("response_format", None)
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
            logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise ProviderException(f"Azure OpenAI chat completion failed: {e}")

    def get_autogen_client(self):
        """Get autogen-compatible client for Azure OpenAI."""
        try:
            endpoint = self.config.get("endpoint")
            deployment_name = self.config.get("deployment_name")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)

            model_name = self.config.get("model_name")
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
                    timeout=timeout,
                    **model_args,
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")

                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    timeout=timeout,
                    **model_args,
                )
        except Exception as e:
            raise ProviderException(f"Failed to create Azure OpenAI autogen client: {e}")

    async def close(self):
        """Close the LLM client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI LLM client")
            await self.client.close()
