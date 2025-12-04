"""Azure OpenAI LLM client implementation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import tiktoken
from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI, APIError, APITimeoutError, RateLimitError

from mmct_agent.core.exceptions import LLMError
from mmct_agent.core.types import Message, StreamChunk, TokenUsage, ToolCall
from mmct_agent.llm.base import BaseLLMClient, LLMConfig, LLMResponse
from mmct_agent.tools.base import ToolDefinition
from mmct_agent.observability.logging import get_logger

logger = get_logger(__name__)


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI LLM client with streaming, retry logic, and token tracking.
    
    Supports two authentication methods:
    1. API Key: Pass `api_key` parameter
    2. Azure AD/Entra ID: Pass `azure_credential` parameter (e.g., DefaultAzureCredential)
    
    Example with API key:
        ```python
        client = AzureOpenAIClient(
            api_key="your-api-key",
            endpoint="https://your-resource.openai.azure.com/",
            deployment="gpt-4",
        )
        ```
    
    Example with Azure credential:
        ```python
        from azure.identity import DefaultAzureCredential
        
        client = AzureOpenAIClient(
            azure_credential=DefaultAzureCredential(),
            endpoint="https://your-resource.openai.azure.com/",
            deployment="gpt-4",
        )
        ```
    """
    
    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key: str | None = None,
        azure_credential: TokenCredential | None = None,
        api_version: str = "2024-02-15-preview",
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize Azure OpenAI client.
        
        Args:
            endpoint: Azure OpenAI endpoint URL.
            deployment: Deployment name.
            api_key: Azure OpenAI API key. Either this or azure_credential must be provided.
            azure_credential: Azure credential object (e.g., DefaultAzureCredential).
                Either this or api_key must be provided.
            api_version: API version to use.
            config: Additional LLM configuration.
            
        Raises:
            ValueError: If neither api_key nor azure_credential is provided,
                or if both are provided.
        """
        super().__init__(config)
        
        # Validate authentication parameters
        if api_key and azure_credential:
            raise ValueError(
                "Only one of 'api_key' or 'azure_credential' should be provided, not both."
            )
        if not api_key and not azure_credential:
            raise ValueError(
                "Either 'api_key' or 'azure_credential' must be provided for authentication."
            )
        
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self._auth_method = "api_key" if api_key else "azure_ad"
        
        # Build client based on authentication method
        if api_key:
            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.endpoint,
                api_version=api_version,
                timeout=self.config.timeout_seconds,
                max_retries=0,  # We handle retries ourselves
            )
        else:
            # Use Azure AD token credential
            token_provider = get_bearer_token_provider(
                azure_credential,
                "https://cognitiveservices.azure.com/.default",
            )
            self._client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=self.endpoint,
                api_version=api_version,
                timeout=self.config.timeout_seconds,
                max_retries=0,  # We handle retries ourselves
            )
        
        # Initialize tokenizer
        try:
            self._encoding = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "azure_openai"
    
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Azure OpenAI.
        
        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters.
            
        Returns:
            LLMResponse with completion result.
            
        Raises:
            LLMError: If the API call fails after retries.
        """
        start_time = time.perf_counter()
        
        # Prepare request parameters
        request_params = self._build_request_params(messages, tools, **kwargs)
        
        # Execute with retry logic
        response = await self._execute_with_retry(
            lambda: self._client.chat.completions.create(**request_params)
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse response
        choice = response.choices[0]
        message = choice.message
        
        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [ToolCall.from_openai(tc) for tc in message.tool_calls]
        
        # Build token usage
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        self._update_token_usage(usage)
        
        # Log at debug level with concise format
        tool_info = f", {len(tool_calls)} tool calls" if tool_calls else ""
        logger.debug(
            f"  ⚡ LLM: {usage.prompt_tokens}→{usage.completion_tokens} tokens, {latency_ms:.0f}ms{tool_info}"
        )
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            token_usage=usage,
            finish_reason=choice.finish_reason,
            model=response.model,
            latency_ms=latency_ms,
            raw_response=response,
        )
    
    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.
        
        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters.
            
        Yields:
            StreamChunk objects with partial content.
            
        Raises:
            LLMError: If the API call fails after retries.
        """
        start_time = time.perf_counter()
        
        # Prepare request parameters
        request_params = self._build_request_params(messages, tools, stream=True, **kwargs)
        
        # Execute with retry logic
        stream = await self._execute_with_retry(
            lambda: self._client.chat.completions.create(**request_params)
        )
        
        # Track tool calls during streaming
        tool_call_chunks: dict[int, dict[str, Any]] = {}
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        async for chunk in stream:
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Handle content
            if delta.content:
                full_content += delta.content
                yield StreamChunk(content=delta.content, is_complete=False)
            
            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_call_chunks[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_chunks[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_chunks[idx]["arguments"] += tc.function.arguments
            
            # Check for completion
            if choice.finish_reason:
                # Get usage from final chunk if available
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                
                # Build final tool calls
                final_tool_calls = None
                if tool_call_chunks:
                    import json
                    final_tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=json.loads(tc["arguments"]) if tc["arguments"] else {},
                        )
                        for tc in tool_call_chunks.values()
                    ]
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Estimate tokens if not provided
                if prompt_tokens == 0:
                    prompt_tokens = await self.count_messages_tokens(messages)
                if completion_tokens == 0:
                    completion_tokens = await self.count_tokens(full_content)
                
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                self._update_token_usage(usage)
                
                logger.debug(
                    "LLM stream completion",
                    extra={
                        "deployment": self.deployment,
                        "latency_ms": latency_ms,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "has_tool_calls": final_tool_calls is not None,
                    },
                )
                
                yield StreamChunk(
                    content="",
                    is_complete=True,
                    tool_calls=final_tool_calls,
                    token_usage=usage,
                )
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens.
        """
        return len(self._encoding.encode(text))
    
    async def count_messages_tokens(self, messages: list[Message]) -> int:
        """Count tokens in a list of messages.
        
        Args:
            messages: Messages to count tokens for.
            
        Returns:
            Total token count.
        """
        total = 0
        for msg in messages:
            # Base tokens per message (role, separators, etc.)
            total += 4
            if msg.content:
                total += await self.count_tokens(msg.content)
            if msg.name:
                total += await self.count_tokens(msg.name)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += await self.count_tokens(tc.name)
                    import json
                    args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                    total += await self.count_tokens(args_str)
        # Priming tokens
        total += 3
        return total
    
    def _build_request_params(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request parameters for the API call.
        
        Args:
            messages: Conversation messages.
            tools: Tool definitions.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.
            
        Returns:
            Request parameters dictionary.
        """
        params: dict[str, Any] = {
            "model": self.deployment,
            "messages": [msg.to_openai_dict() for msg in messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": stream,
        }
        
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
            
        if self.config.stop:
            params["stop"] = self.config.stop
        
        # Add tools if provided
        if tools:
            params["tools"] = [tool.to_openai_dict() for tool in tools]
            params["tool_choice"] = "auto"
        
        # Add stream options for usage tracking
        if stream:
            params["stream_options"] = {"include_usage": True}
        
        # Merge extra params
        params.update(self.config.extra_params)
        params.update(kwargs)
        
        return params
    
    async def _execute_with_retry(self, func: Any) -> Any:
        """Execute a function with retry logic.
        
        Args:
            func: Async function to execute.
            
        Returns:
            Function result.
            
        Raises:
            LLMError: If all retries fail.
        """
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func()
            except APITimeoutError as e:
                last_error = e
                logger.warning(
                    f"Azure OpenAI timeout (attempt {attempt + 1}/{self.config.max_retries + 1})",
                    extra={"deployment": self.deployment},
                )
            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Azure OpenAI rate limit (attempt {attempt + 1}/{self.config.max_retries + 1})",
                    extra={"deployment": self.deployment},
                )
            except APIError as e:
                last_error = e
                # Don't retry on client errors (4xx except 429)
                if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise LLMError(
                        message=f"Azure OpenAI API error: {e.message}",
                        provider=self.provider_name,
                        status_code=e.status_code,
                        retry_count=attempt,
                        details={"deployment": self.deployment},
                    ) from e
                logger.warning(
                    f"Azure OpenAI API error (attempt {attempt + 1}/{self.config.max_retries + 1}): {e.message}",
                    extra={"deployment": self.deployment, "status_code": e.status_code},
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)
        
        raise LLMError(
            message=f"Azure OpenAI API call failed after {self.config.max_retries + 1} attempts: {last_error}",
            provider=self.provider_name,
            retry_count=self.config.max_retries + 1,
            details={"deployment": self.deployment, "last_error": str(last_error)},
        )
