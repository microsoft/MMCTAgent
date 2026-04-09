from mmct.providers.base import BaseLLMProvider
from loguru import logger
from openai import AsyncAzureOpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from mmct.utils.error_handler import ProviderException, ConfigurationException
from typing import Dict, Any, List, Union, Optional
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class AzureLLMProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider implementation."""

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
        """Initialize AzureLLMProvider.

        Args:
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the LLM deployment
            api_version: Azure OpenAI API version (default: 2024-08-01-preview)
            credentials: Azure credentials for token-based authentication (mutually exclusive with api_key)
            api_key: API key for key-based authentication (mutually exclusive with credentials)
            timeout: Request timeout in seconds (default: 200)
            max_retries: Maximum number of retry attempts (default: 2)

        Raises:
            ConfigurationException: If neither credentials nor api_key is provided,
                                   or if both are provided, or if required fields are missing
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

    def _initialize_client(self):
        """Initialize Azure OpenAI client with either credentials or API key."""
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
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Azure OpenAI."""
        try:

            temperature = kwargs.get("temperature",0)
            max_tokens = kwargs.get("max_tokens",4000)
            response_format = kwargs.get("response_format")

            # Remove temperature, max_tokens, and response_format from kwargs to avoid duplicate arguments
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens", "response_format"]
            }

            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel

            if (
                response_format
                and isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                # Build JSON schema with all properties required (Azure OpenAI requirement)
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
                
                # Parse the JSON response into the Pydantic model
                import json
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
                # Standard completion without structured output
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

    def _build_strict_schema(self, model: type) -> dict:
        """
        Build a JSON schema with all properties required for Azure OpenAI strict mode.
        
        Azure OpenAI structured outputs require all properties in the schema
        to be listed in the 'required' array. This method generates a schema
        and recursively makes all properties required.
        
        Args:
            model: The Pydantic model class
            
        Returns:
            JSON schema dict with all properties required
        """
        schema = model.model_json_schema()
        self._make_schema_strict(schema)
        return schema
    
    def _make_schema_strict(self, schema: dict) -> None:
        """Recursively make schema strict for Azure OpenAI (adds additionalProperties: false)."""
        
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
            
            # Recurse nested objects
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    self._make_schema_strict(prop_schema)
        
        # Handle array items
        if "items" in schema and isinstance(schema["items"], dict):
            self._make_schema_strict(schema["items"])
        
        # Handle anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for item in schema[key]:
                    if isinstance(item, dict):
                        self._make_schema_strict(item)
        
        # Handle additionalProperties objects
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            self._make_schema_strict(schema["additionalProperties"])

    def get_autogen_client(self, **kwargs):
        """Get autogen-compatible client for Azure OpenAI."""
        try:
            model = self.model_name if self.model_name else self.deployment_name

            # Provide model_info for custom deployment names not recognized by autogen
            model_info = {
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-5",
                "structured_output": True,
                "multiple_system_messages": True,
            }

            if self.credentials is not None:
                # Use credentials with token-based authentication
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

    async def close(self):
        """Close the LLM client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI LLM client")
            await self.client.close()

    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a JSON response from a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments passed to chat_completion
            
        Returns:
            Parsed JSON response as a dictionary
        """
        import json
        
        messages = [{"role": "user", "content": prompt}]
        
        # Request JSON output format
        response = await self.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        
        content = response.get("content", "{}")
        
        # Parse JSON response
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return {}
        elif isinstance(content, dict):
            return content
        
        return {}
