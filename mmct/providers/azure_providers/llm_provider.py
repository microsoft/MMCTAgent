"""Azure OpenAI LLM provider implementation.

This module provides the AzureLLMProvider class, which implements the 
BaseLLMProvider interface for integrating with Azure OpenAI services, 
including support for structured outputs and AutoGen clients.
"""

import json
from typing import Dict, Any, List, Union, Optional
from loguru import logger
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from pydantic import BaseModel

from mmct.providers.base import BaseLLMProvider
from mmct.utils.error_handler import ProviderException, ConfigurationException, handle_exceptions, convert_exceptions
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class AzureLLMProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider implementation.

    This provider handles authentication (Key or Token-based), client 
    initialization, and response generation via Azure OpenAI. It supports 
    strict JSON schemas for structured outputs and provides an interface 
    for AutoGen's multi-agent framework.

    Attributes:
        endpoint (str): The Azure OpenAI service endpoint.
        deployment_name (str): The name of the specific LLM deployment.
        api_version (str): The version of the Azure OpenAI API to use.
        credentials (Union[AzureKeyCredential, AsyncTokenCredential], optional): 
            Identity-based credentials.
        api_key (str, optional): Key-based authentication string.
        timeout (int): Global request timeout in seconds.
        max_retries (int): Maximum retry attempts for failed requests.
        model_name (str, optional): The underlying model name (e.g., gpt-4o).
        client (AsyncAzureOpenAI): The initialized async client.
    """

    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        model_name: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = 200,
        max_retries: Optional[int] = 2,
    ):
        """Initializes the AzureLLMProvider.

        Args:
            endpoint: Azure OpenAI endpoint URL.
            deployment_name: Name of the LLM deployment.
            model_name: Friendly name for the model (e.g., 'gpt-4o').
            api_version: Azure OpenAI API version.
            credentials: Azure credentials for token-based authentication.
                Mutually exclusive with `api_key`.
            api_key: API key for key-based authentication.
                Mutually exclusive with `credentials`.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.

        Raises:
            ConfigurationException: If required fields are missing or if both
                `credentials` and `api_key` are provided.
        """

        if not endpoint:
            raise ConfigurationException("Azure OpenAI endpoint is required for LLM Provider!")

        if not deployment_name:
            raise ConfigurationException(
                "Azure OpenAI deployment name is required for LLM Provider!"
            )

        # Validate that exactly one of credentials or api_key is provided
        if credentials is None and api_key is None:
            raise ConfigurationException("Either credentials or api_key must be provided!")

        if credentials is not None and api_key is not None:
            raise ConfigurationException(
                "Only one of credentials or api_key should be provided, not both!"
            )

        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.credentials = credentials
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = model_name
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncAzureOpenAI:
        """Initializes the Azure OpenAI client with either credentials or API key.

        Returns:
            AsyncAzureOpenAI: The initialized asynchronous client.

        Raises:
            ProviderException: If client initialization fails.
        """
        try:
            if self.credentials is not None:
                # Use credentials with token-based authentication
                token_provider = get_bearer_token_provider(
                    self.credentials, "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
            else:
                # Use API key authentication
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Generates a chat completion using Azure OpenAI.

        Supports standard completion as well as structured outputs via 
        Pydantic models in the `response_format` argument.

        Args:
            messages: A list of message dictionaries.
            **kwargs: Additional parameters like temperature, max_tokens, 
                and response_format. If `response_format` is a Pydantic 
                BaseModel class, the provider will enforce structured output 
                using Azure OpenAI's strict JSON mode.

        Returns:
            Dict[str, Any]: A response dictionary containing content, usage, 
                model, and finish reason.

        Raises:
            ProviderException: If the completion request fails.
        """
        try:
            temperature = kwargs.get("temperature", 0)
            max_tokens = kwargs.get("max_tokens", 4000)
            response_format = kwargs.get("response_format")

            # Remove parameters to avoid duplicate arguments in the API call
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens", "response_format"]
            }

            # Handle structured outputs if response_format is a Pydantic model class
            if (
                response_format
                and isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                # Build strict JSON schema (required for Azure OpenAI structured output)
                schema = self._build_strict_schema(response_format)
                
                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    **filtered_kwargs,
                )
                
                content = response.choices[0].message.content
                if isinstance(content, str):
                    parsed = response_format.model_validate(json.loads(content))
                else:
                    parsed = content

                return {
                    "content": parsed,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                }
            else:
                # Standard completion
                completion_kwargs = {
                    "model": self.deployment_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **filtered_kwargs,
                }

                if response_format:
                    completion_kwargs["response_format"] = response_format

                response = await self.client.chat.completions.create(**completion_kwargs)

                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                }
        except Exception as e:
            logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise ProviderException(f"Azure OpenAI chat completion failed: {e}")

    def _build_strict_schema(self, model: type) -> Dict[str, Any]:
        """Generates a JSON schema compatible with Azure OpenAI Strict mode.

        Azure OpenAI structured outputs require all properties to be present 
        in the 'required' array and disallow 'additionalProperties'.

        Args:
            model: The Pydantic model class to convert.

        Returns:
            Dict[str, Any]: The converted strict JSON schema.
        """
        schema = model.model_json_schema()
        self._make_schema_strict(schema)
        return schema
    
    def _make_schema_strict(self, schema: Dict[str, Any]) -> None:
        """Recursively enforces strict mode on a JSON schema dictionary.

        Args:
            schema: The schema dictionary to modify in-place.
        """
        # Ensure top-level is strict
        if "type" in schema and schema["type"] == "object":
            schema["additionalProperties"] = False
        
        # Process $defs first
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                self._make_schema_strict(def_schema)
        
        # Make properties required
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

            # Strip sibling keywords from $ref properties for Azure compatibility
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict) and "$ref" in prop_schema:
                    prop_schema.pop("description", None)
                    prop_schema.pop("title", None)

            # Recurse nested objects
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    self._make_schema_strict(prop_schema)
        
        # Handle array items
        if "items" in schema and isinstance(schema["items"], dict):
            self._make_schema_strict(schema["items"])
        
        # Handle combinations
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for item in schema[key]:
                    if isinstance(item, dict):
                        self._make_schema_strict(item)
        
        # Handle additionalProperties objects
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            self._make_schema_strict(schema["additionalProperties"])

    def get_autogen_client(self, **kwargs: Any) -> AzureOpenAIChatCompletionClient:
        """Returns an AutoGen-compatible client for Azure OpenAI.

        Args:
            **kwargs: Additional overrides for the completion client.

        Returns:
            AzureOpenAIChatCompletionClient: The configured AutoGen client.

        Raises:
            ProviderException: If client creation fails.
        """
        try:
            model = self.model_name if self.model_name else self.deployment_name

            # Explicit capabilities for deployment-agnostic mapping
            model_info = {
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-4o",
                "structured_output": True,
                "multiple_system_messages": True,
            }

            if self.credentials is not None:
                token_provider = get_bearer_token_provider(
                    self.credentials, "https://cognitiveservices.azure.com/.default"
                )
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=self.deployment_name,
                    model=model,
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=self.timeout,
                    model_info=model_info,
                )
            else:
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=self.deployment_name,
                    model=model,
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    model_info=model_info,
                )
        except Exception as e:
            raise ProviderException(f"Failed to create Azure OpenAI autogen client: {e}")

    async def close(self) -> None:
        """Closes the Azure OpenAI client and releases underlying resources."""
        if self.client:
            logger.info("Closing Azure OpenAI LLM client")
            await self.client.close()

    async def generate_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generates a JSON response from a prompt.

        Args:
            prompt: The instruction to send to the LLM.
            **kwargs: Parameters passed to `chat_completion`.

        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        
        content = response.get("content", "{}")
        
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return {}
        elif isinstance(content, dict):
            return content
        
        return {}
