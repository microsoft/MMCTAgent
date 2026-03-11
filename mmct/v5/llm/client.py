"""Structured LLM client for V5 pipeline.

Wraps AutoGen's model_client.create() with:
- JSON mode (json_output=True)
- Pydantic structured output (json_output=SomeModel)
- Token tracking
- Retry on parse failure
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel
from autogen_core.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    CreateResult,
)

T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient:
    """Wrapper for structured LLM calls with token tracking.

    Uses AutoGen's model_client.create() under the hood.
    All calls use json_output=True for reliable JSON responses.

    Attributes:
        model_client: AutoGen ChatCompletionClient.
        total_prompt_tokens: Accumulated prompt tokens across all calls.
        total_completion_tokens: Accumulated completion tokens across all calls.
    """

    def __init__(self, model_client):
        self.model_client = model_client
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _track_usage(self, result: CreateResult) -> None:
        if result.usage:
            self.total_prompt_tokens += result.usage.prompt_tokens
            self.total_completion_tokens += result.usage.completion_tokens

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """Call LLM and parse response as JSON dict.

        Args:
            system_prompt: System message content.
            user_message: User message content.
            max_retries: Retries on JSON parse failure.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If response cannot be parsed as JSON after retries.
        """
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_message, source="user"),
        ]

        for attempt in range(1 + max_retries):
            result = await self.model_client.create(
                messages=messages,
                json_output=True,
            )
            self._track_usage(result)

            content = result.content
            if isinstance(content, str):
                content = content.strip()
                # Strip markdown code fences if present
                if content.startswith("```"):
                    lines = content.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    content = "\n".join(lines).strip()

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        logger.warning(
                            f"JSON parse failed (attempt {attempt + 1}), retrying: {e}"
                        )
                        messages.append(
                            AssistantMessage(content=content, source="assistant")
                        )
                        messages.append(
                            UserMessage(
                                content=f"Your response was not valid JSON: {e}. "
                                "Please respond with ONLY valid JSON.",
                                source="user",
                            )
                        )
                    else:
                        raise ValueError(
                            f"Failed to parse JSON after {1 + max_retries} attempts: {e}"
                        ) from e
            else:
                raise ValueError(f"Unexpected response type: {type(content)}")

        raise ValueError("Unreachable")

    async def call_typed(
        self,
        system_prompt: str,
        user_message: str,
        response_type: Type[T],
        max_retries: int = 1,
    ) -> T:
        """Call LLM and parse response into a Pydantic model.

        Uses json_output=response_type for structured output when supported,
        falls back to json_output=True + manual parsing.

        Args:
            system_prompt: System message content.
            user_message: User message content.
            response_type: Pydantic model class to parse into.
            max_retries: Retries on parse failure.

        Returns:
            Instance of response_type.
        """
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_message, source="user"),
        ]

        for attempt in range(1 + max_retries):
            try:
                result = await self.model_client.create(
                    messages=messages,
                    json_output=response_type,
                )
                self._track_usage(result)

                content = result.content
                if isinstance(content, response_type):
                    return content

                # Structured output returned as string — parse manually
                if isinstance(content, str):
                    text = content.strip()
                    if text.startswith("```"):
                        lines = text.split("\n")
                        lines = [l for l in lines if not l.strip().startswith("```")]
                        text = "\n".join(lines).strip()
                    parsed = json.loads(text)
                    return response_type.model_validate(parsed)

                raise ValueError(f"Unexpected response type: {type(content)}")

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Structured output parse failed (attempt {attempt + 1}): {e}"
                    )
                    # Fall back to json_output=True
                    result = await self.model_client.create(
                        messages=messages,
                        json_output=True,
                    )
                    self._track_usage(result)
                    content = result.content if isinstance(result.content, str) else str(result.content)
                    parsed = json.loads(content.strip())
                    return response_type.model_validate(parsed)
                raise

    def get_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }

    def reset_usage(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
