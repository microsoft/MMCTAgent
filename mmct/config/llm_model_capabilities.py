# mmct/config/llm_model_capabilities.py

from pydantic import BaseModel, Field
from typing import Dict

class LLMModelCapabilities(BaseModel):
    """LLM model capabilities."""
    temperature: bool = Field(..., description="Whether the LLM supports `temperature`.")
    top_p: bool = Field(..., description="Whether the LLM supports `top_p`.")
    presence_penalty: bool = Field(..., description="Whether the LLM supports `presence_penalty`.")
    frequency_penalty: bool = Field(..., description="Whether the LLM supports `frequency_penalty`.")
    logprobs: bool = Field(..., description="Whether the LLM supports `logprobs`.")
    top_logprobs: bool = Field(..., description="Whether the LLM supports `top_logprobs`.")
    logit_bias: bool = Field(..., description="Whether the LLM supports `logit_bias`.")
    max_tokens: bool = Field(..., description="Whether the LLM supports specifying `max_tokens`.")
    max_completion_tokens: bool = Field(...,description="Whether the LLM supports specifying `max_completion_tokens`. This is used for reasoning models.")
    reasoning_effort: bool = Field(..., description="Whether the LLM supports reasoning effort.")

# dictionary of model names and their capabilities
MODEL_CAPABILITIES: Dict[str, LLMModelCapabilities] = {
    "o3": LLMModelCapabilities(
        temperature=False,
        top_p=False,
        presence_penalty=False,
        frequency_penalty=False,
        logprobs=False,
        top_logprobs=False,
        logit_bias=False,
        max_tokens=False,
        max_completion_tokens=True,
        reasoning_effort=True
    ),
    "o3-mini": LLMModelCapabilities(
        temperature=False,
        top_p=False,
        presence_penalty=False,
        frequency_penalty=False,
        logprobs=False,
        top_logprobs=False,
        logit_bias=False,
        max_tokens=False,
        max_completion_tokens=True,
        reasoning_effort=True
    ),
    "gpt-5": LLMModelCapabilities(
        temperature=False,
        top_p=False,
        presence_penalty=False,
        frequency_penalty=False,
        logprobs=False,
        top_logprobs=False,
        logit_bias=False,
        max_tokens=False,
        max_completion_tokens=True,
        reasoning_effort=True
    ),
    "gpt-5-mini": LLMModelCapabilities(
        temperature=False,
        top_p=False,
        presence_penalty=False,
        frequency_penalty=False,
        logprobs=False,
        top_logprobs=False,
        logit_bias=False,
        max_tokens=False,
        max_completion_tokens=True,
        reasoning_effort=True
    ),
    "gpt-4.1": LLMModelCapabilities(
        temperature=True,
        top_p=True,
        presence_penalty=True,
        frequency_penalty=True,
        logprobs=True,
        top_logprobs=True,
        logit_bias=True,
        max_tokens=True,
        max_completion_tokens=False,
        reasoning_effort=False
    ),
    "gpt-4.1-mini": LLMModelCapabilities(
        temperature=True,
        top_p=True,
        presence_penalty=True,
        frequency_penalty=True,
        logprobs=True,
        top_logprobs=True,
        logit_bias=True,
        max_tokens=True,
        max_completion_tokens=False,
        reasoning_effort=False
    ),
    "gpt-4o": LLMModelCapabilities(
        temperature=True,
        top_p=True,
        presence_penalty=True,
        frequency_penalty=True,
        logprobs=True,
        top_logprobs=True,
        logit_bias=True,
        max_tokens=True,
        max_completion_tokens=False,
        reasoning_effort=False
    ),
    "gpt-4o-mini": LLMModelCapabilities(
        temperature=True,
        top_p=True,
        presence_penalty=True,
        frequency_penalty=True,
        logprobs=True,
        top_logprobs=True,
        logit_bias=True,
        max_tokens=True,
        max_completion_tokens=False,
        reasoning_effort=False
    ),
    "non-reasoning": LLMModelCapabilities(
        temperature=True,
        top_p=True,
        presence_penalty=True,
        frequency_penalty=True,
        logprobs=True,
        top_logprobs=True,
        logit_bias=True,
        max_tokens=True,
        max_completion_tokens=False,
        reasoning_effort=False
    ),
    "reasoning": LLMModelCapabilities(
        temperature=False,
        top_p=False,
        presence_penalty=False,
        frequency_penalty=False,
        logprobs=False,
        top_logprobs=False,
        logit_bias=False,
        max_tokens=False,
        max_completion_tokens=True,
        reasoning_effort=True
    )
}