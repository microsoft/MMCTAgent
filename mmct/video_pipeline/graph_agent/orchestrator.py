"""Graph Orchestrator — coordinates the swarm-based (graph) pipeline.

This module orchestrates the query pipeline with a Neo4j graph backend using an 
agentic swarm architecture. The workflow typically involves:
1. Planner analyzes query and creates a retrieval plan.
2. VideoAgent searches and traverses the Neo4j knowledge graph.
3. ImageAgent performs perceptual analysis on keyframes.
4. CriticAgent validates the generated answer (optional).
5. Planner synthesizes the final answer with evidence and citations.

The system uses AutoGen Swarm for agent handoffs and context management.
"""

import json
import time
from typing import Any, Dict, Optional, AsyncGenerator, Union, List
from dataclasses import dataclass, field

from loguru import logger
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import FunctionCallTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    HandoffMessage,
)
from autogen_core.model_context import BufferedChatCompletionContext

from mmct.acl import AccessCheckCallback
from mmct.video_pipeline.graph_agent.agents.planner_agent import PlannerAgent, SUBMIT_TOOL_NAME, reset_handoff_counter
from mmct.video_pipeline.graph_agent.agents.video_agent import VideoAgent
from mmct.video_pipeline.graph_agent.agents.image_agent import ImageAgent
from mmct.video_pipeline.graph_agent.agents.critic_agent import CriticAgent
from mmct.video_pipeline.graph_agent.query.neo4j_provider import Neo4jQueryProvider
from mmct.video_pipeline.graph_agent.schemas import QueryResponse
from mmct.image_pipeline.config import ImageAgentProviderConfig

_log = logger.bind(component="agent")


# Buffer sizes for agent context windows
PLANNER_BUFFER_SIZE = 15
VIDEO_AGENT_BUFFER_SIZE = 12
IMAGE_AGENT_BUFFER_SIZE = 10
CRITIC_BUFFER_SIZE = 10

_TERMINATE_STRING = "TERMINATE"


@dataclass
class TokenUsage:
    """Tracks accumulated token usage across the swarm execution.

    Attributes:
        prompt_tokens (int): Total input tokens used.
        completion_tokens (int): Total output tokens generated.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _log_message(message: Any, token_usage: TokenUsage, request_id: str = "") -> None:
    """Logs a formatted agentic message or event via loguru.

    Tracks accumulated token usage and logs different message types
    (tool calls, handoffs, text results) with structured extras for
    Azure Monitor filtering.

    Args:
        message: The message object or event (from AutoGen).
        token_usage: A TokenUsage object to accumulate LLM costs. This object
            is modified in-place to track prompt and completion tokens.
        request_id: Optional correlation ID for the request.
    """
    source = getattr(message, "source", "Unknown")
    rid = f"[{request_id}] " if request_id else ""
    bound = _log.bind(request_id=request_id, agent=source)

    if hasattr(message, "models_usage") and message.models_usage:
        token_usage.prompt_tokens += message.models_usage.prompt_tokens
        token_usage.completion_tokens += message.models_usage.completion_tokens

    if isinstance(message, TaskResult):
        bound.info(f"{rid}[System] Task Completed")
        bound.info(f"{rid}Messages: {len(message.messages)}")
        bound.info(f"{rid}Finish reason: {message.stop_reason}")
        bound.info(f"{rid}Prompt tokens: {token_usage.prompt_tokens}")
        bound.info(f"{rid}Completion tokens: {token_usage.completion_tokens}")
    elif isinstance(message, TextMessage):
        text = message.content
        if len(text) > 500:
            text = text[:500] + "… [truncated]"
        bound.info(f"{rid}[{source}] {text}")
    elif isinstance(message, HandoffMessage):
        bound.info(f"{rid}[{source}] Handoff → {message.target}")
        if message.content:
            bound.info(f"{rid}  Message: {message.content}")
    elif isinstance(message, ToolCallRequestEvent):
        content = message.content
        if isinstance(content, list):
            for tc in content:
                bound.bind(tool_name=tc.name).info(f"{rid}[{source}] Tool Call: {tc.name}")
                bound.info(f"{rid}  Args: {tc.arguments}")
        else:
            bound.info(f"{rid}[{source}] Tool Call: {content}")
    elif isinstance(message, ToolCallExecutionEvent):
        content = message.content
        if isinstance(content, list):
            for tr in content:
                result_text = str(tr.content)
                if len(result_text) > 500:
                    result_text = result_text[:500] + "… [truncated]"
                bound.info(f"{rid}[{source}] Tool Result ({tr.call_id}): {result_text}")
        else:
            bound.info(f"{rid}[{source}] Tool Result: {content}")
    else:
        content = str(getattr(message, "content", str(message)))
        bound.info(f"{rid}[{source}] [{type(message).__name__}] {content}")

    if hasattr(message, "models_usage") and message.models_usage and not isinstance(message, TaskResult):
        bound.info(
            f"{rid}[Tokens: +{message.models_usage.prompt_tokens} prompt, "
            f"+{message.models_usage.completion_tokens} completion]"
        )


def _try_extract_json(content: str) -> Optional[str]:
    """Attempts to extract a JSON substring from a raw message string.

    Args:
        content: The raw text content possibly containing JSON.

    Returns:
        Optional[str]: The extracted JSON string, or None if no valid JSON block 
            was identified.
    """
    cleaned = content.strip()

    if "```json" in cleaned:
        start_idx = cleaned.find("```json") + len("```json")
        end_idx = cleaned.find("```", start_idx)
        if end_idx != -1:
            return cleaned[start_idx:end_idx].strip()

    brace_idx = cleaned.find("{")
    if brace_idx != -1:
        potential_json = cleaned[brace_idx:]
        potential_json = potential_json.rstrip().rstrip(_TERMINATE_STRING).rstrip()
        if potential_json.endswith("}"):
            return potential_json

    return None


def _extract_json_from_messages(messages: list) -> Optional[str]:
    """Finds and extracts the final structured response from a list of messages.

    Args:
        messages: List of agentic messages from the swarm execution.

    Returns:
        Optional[str]: The extracted JSON response content.
    """
    if not messages:
        return None

    # Primary: find submit_final_answer tool result
    for msg in reversed(messages):
        if isinstance(msg, ToolCallExecutionEvent):
            for execution in msg.content:
                if execution.name == SUBMIT_TOOL_NAME:
                    return execution.content

    # Fallback: scan planner text messages for JSON
    best_with_sources = None
    best_with_answer = None

    for msg in reversed(messages):
        source = getattr(msg, "source", "")
        content = getattr(msg, "content", "")
        if not content or not isinstance(content, str):
            continue
        if source and source != "planner":
            continue

        json_str = _try_extract_json(content)
        if not json_str:
            continue

        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(parsed, dict) or "answer" not in parsed:
            continue

        if parsed.get("sources") and len(parsed["sources"]) > 0:
            if best_with_sources is None:
                best_with_sources = json_str
                break

        if best_with_answer is None:
            best_with_answer = json_str

    return best_with_sources or best_with_answer


def _parse_response(json_str: str) -> Dict[str, Any]:
    """Parses a JSON string into a dictionary with error fallback.

    Args:
        json_str: The raw JSON string to parse.

    Returns:
        Dict[str, Any]: The parsed response dictionary.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {"answer": json_str, "sources": [], "parse_error": str(e)}


class GraphOrchestrator:
    """Orchestrates the agentic swarm-based video query pipeline.

    The GraphOrchestrator coordinates an autonomous swarm of specialized agents:
    - **PlannerAgent**: Analyzes the query, creates retrieval plans, and 
        synthesizes the final response.
    - **VideoAgent**: Interfaces with the Neo4j knowledge graph to retrieve 
        video metadata, scenes, and visual features.
    - **ImageAgent**: Perceptually analyzes specific keyframes when detailed 
        visual context is required.
    - **CriticAgent**: Performs a reflective validation pass to ensure 
        accuracy and logical consistency.

    Attributes:
        model_client: The primary LLM client for agent reasoning.
        neo4j_provider: Provider for interacting with the Neo4j graph store.
        storage_provider: Optional provider for retrieving image assets.
        image_llm_provider: Optional vision-capable LLM for the ImageAgent.
        use_critic (bool): Whether to enable the critic validation pass.
        max_turns (int): Maximum allowed conversation turns in the swarm.
        video_catalog (str, optional): Pre-generated summary of available videos.
    """

    def __init__(
        self,
        model_client: Any,
        neo4j_provider: Neo4jQueryProvider,
        storage_provider: Optional[Any] = None,
        image_llm_provider: Optional[Any] = None,
        use_critic: bool = True,
        max_turns: int = 20,
        video_catalog: Optional[str] = None,
        acl_callback: Optional[AccessCheckCallback] = None,
    ):
        """Initializes the GraphOrchestrator.

        Args:
            model_client: AutoGen-compatible LLM client.
            neo4j_provider: Instance of Neo4jQueryProvider for graph access.
            storage_provider: Optional provider for keyframe retrieval.
            image_llm_provider: Optional LLM config for vision reasoning.
            use_critic: If True, includes a critic agent in the swarm.
            max_turns: Maximum conversation turns before termination.
            video_catalog: Optional text summary for planner tool selection context.
            acl_callback: Optional access-check callback. When provided, the
                VideoAgent's tools are wrapped with ACL post-filters; when
                None, tools run unwrapped. Pipeline-level enforcement of
                ACL_ENABLED is the source of truth for when this must be set.
        """
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.storage_provider = storage_provider
        self.image_llm_provider = image_llm_provider
        self.use_critic = use_critic
        self.max_turns = max_turns
        self.video_catalog = video_catalog
        self._acl_callback = acl_callback

        self._agents_initialized = False
        self._image_agent_wrapper = None

    def _initialize_agents(self) -> None:
        """Initializes all agents with appropriate buffered contexts.
        
        This method is lazy-loaded before the swarm is built.
        """
        if self._agents_initialized:
            return

        self._planner_wrapper = PlannerAgent(
            model_client=self.model_client,
            use_critic=self.use_critic,
            model_context=BufferedChatCompletionContext(buffer_size=PLANNER_BUFFER_SIZE),
            video_catalog=self.video_catalog,
        )

        self._video_agent_wrapper = VideoAgent(
            model_client=self.model_client,
            neo4j_provider=self.neo4j_provider,
            model_context=BufferedChatCompletionContext(buffer_size=VIDEO_AGENT_BUFFER_SIZE),
            acl_callback=self._acl_callback,
        )

        if self.image_llm_provider:
            image_provider = ImageAgentProviderConfig(llm_provider=self.image_llm_provider)
            self._image_agent_wrapper = ImageAgent(
                provider=image_provider,
                model_client=self.model_client,
                storage_provider=self.storage_provider,
                model_context=BufferedChatCompletionContext(buffer_size=IMAGE_AGENT_BUFFER_SIZE),
            )

        if self.use_critic:
            self._critic_wrapper = CriticAgent(
                model_client=self.model_client,
                model_context=BufferedChatCompletionContext(buffer_size=CRITIC_BUFFER_SIZE),
            )

        self._agents_initialized = True

    def _build_swarm(self) -> Swarm:
        """Constructs the AutoGen Swarm team with participants and termination conditions.

        Returns:
            Swarm: The configured swarm instance.
        """
        self._initialize_agents()
        reset_handoff_counter()

        participants = [
            self._planner_wrapper.agent,
            self._video_agent_wrapper.agent,
        ]

        if self._image_agent_wrapper:
            participants.append(self._image_agent_wrapper.agent)

        if self.use_critic:
            participants.append(self._critic_wrapper.agent)

        termination = FunctionCallTermination(SUBMIT_TOOL_NAME) | TextMentionTermination(_TERMINATE_STRING)

        return Swarm(
            participants=participants,
            termination_condition=termination,
            max_turns=self.max_turns,
        )

    def _build_task(
        self,
        query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
    ) -> str:
        """Constructs the task instruction string for the swarm.

        Args:
            query: The user question.
            video_id: Single video constraint.
            video_ids: Multiple video constraints.

        Returns:
            str: The formatted task string.
        """
        task = f"Query: {query}"
        if video_id:
            task += f"\nVideo ID: {video_id}"
        elif video_ids:
            task += f"\nVideo IDs: {', '.join(video_ids)}"
        else:
            task += "\nScope: Cross-video search (all videos)"
        return task

    async def query(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Processes a query through the graph swarm pipeline.

        Args:
            user_query: The natural language question.
            video_id: Optional single video ID scope.
            video_ids: Optional list of video ID scopes.
            request_id: Optional correlation ID for identification.

        Returns:
            Dict[str, Any]: A response dictionary containing the answer, 
                sources, and execution metrics.
        """
        start_time = time.time()
        token_usage = TokenUsage()
        rid = f"[{request_id}] " if request_id else ""
        qlog = _log.bind(request_id=request_id, query=user_query, video_id=video_id)

        qlog.info(f"{'=' * 60}")
        qlog.info(f"{rid}Graph Query Pipeline")
        qlog.info(f"{'=' * 60}")
        qlog.info(f"{rid}Query: {user_query}")
        if video_id:
            qlog.info(f"{rid}Video: {video_id}")
        elif video_ids:
            _log.info(f"{rid}Videos: {video_ids}")
        else:
            _log.info(f"{rid}Scope: Cross-video search")
        _log.info(f"{'-' * 60}")

        team = self._build_swarm()
        task = self._build_task(user_query, video_id, video_ids)

        try:
            final_result = None
            async for message in team.run_stream(task=task):
                _log_message(message, token_usage, request_id=request_id)
                if isinstance(message, TaskResult):
                    final_result = message

            if final_result is None:
                raise RuntimeError("No TaskResult received from swarm")

            response = self._process_result(final_result)
            elapsed = time.time() - start_time
            response["elapsed_seconds"] = elapsed
            response["token_usage"] = {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
            }

            _log.info(f"{'=' * 60}")
            _log.info(f"{rid}Query completed in {elapsed:.2f}s")
            _log.info(f"{'=' * 60}")

            return response

        finally:
            if self._image_agent_wrapper:
                self._image_agent_wrapper.cleanup()

    async def query_stream(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        request_id: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Processes a query with real-time streaming updates.

        Args:
            user_query: User's natural language question.
            video_id: Optional single video ID scope.
            video_ids: Optional list of video ID scopes.
            request_id: Optional correlation ID.

        Yields:
            Dict[str, Any]: Status messages or the final structured response.
        """
        start_time = time.time()
        token_usage = TokenUsage()
        rid = f"[{request_id}] " if request_id else ""

        _log.info(f"{'=' * 60}")
        _log.info(f"{rid}Graph Query Pipeline (Streaming)")
        _log.info(f"{'=' * 60}")
        _log.info(f"{rid}Query: {user_query}")
        _log.info(f"{'-' * 60}")

        team = self._build_swarm()
        task = self._build_task(user_query, video_id, video_ids)

        try:
            async for message in team.run_stream(task=task):
                _log_message(message, token_usage, request_id=request_id)

                if isinstance(message, TaskResult):
                    response = self._process_result(message)
                    elapsed = time.time() - start_time
                    response["elapsed_seconds"] = elapsed
                    response["token_usage"] = {
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                    }
                    _log.info(f"{'=' * 60}")
                    _log.info(f"{rid}Query completed in {elapsed:.2f}s")
                    _log.info(f"{'=' * 60}")
                    yield {"type": "final", "data": response}
                else:
                    content = getattr(message, "content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    yield {
                        "type": "message",
                        "agent": getattr(message, "source", "unknown"),
                        "content": content,
                    }
        finally:
            if self._image_agent_wrapper:
                self._image_agent_wrapper.cleanup()

    def _process_result(self, result: TaskResult) -> Dict[str, Any]:
        """Extracts and formats the final answer from a TaskResult.

        Args:
            result: The completed task result from the AutoGen team.

        Returns:
            Dict[str, Any]: A dictionary containing the final answer, 
                sources, and total tokens.
        """
        msgs = result.messages
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for msg in msgs:
            if hasattr(msg, "models_usage") and msg.models_usage:
                total_prompt_tokens += msg.models_usage.prompt_tokens
                total_completion_tokens += msg.models_usage.completion_tokens

        json_content = _extract_json_from_messages(msgs)

        if json_content:
            parsed = _parse_response(json_content)
        else:
            logger.warning("No JSON response found in messages")
            last_content = ""
            for msg in reversed(msgs):
                content = getattr(msg, "content", "")
                if content and isinstance(content, str) and _TERMINATE_STRING not in content:
                    last_content = content
                    break
            parsed = {"answer": last_content or "No response generated", "sources": []}

        return {
            "answer": parsed.get("answer", ""),
            "sources": parsed.get("sources", []),
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
        }

    async def close(self) -> None:
        """Closes any active database connections and shared providers."""
        if self.neo4j_provider:
            await self.neo4j_provider.close()


async def process_query(
    query: str,
    model_client: Any,
    neo4j_provider: Neo4jQueryProvider,
    video_id: Optional[str] = None,
    video_ids: Optional[List[str]] = None,
    storage_provider: Optional[Any] = None,
    image_llm_provider: Optional[Any] = None,
    use_critic: bool = True,
    stream: bool = False,
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Convenience functional entry point for the graph pipeline.

    Args:
        query: User's natural language question.
        model_client: LLM client for agent reasoning.
        neo4j_provider: Neo4j graph connection.
        video_id: Optional ID to restrict search.
        video_ids: Optional list of IDs to restrict search.
        storage_provider: Optional frame storage provider.
        image_llm_provider: Optional vision-capable LLM provider.
        use_critic: Whether to enable answer validation.
        stream: If True, returns an AsyncGenerator for streaming updates.

    Returns:
        Union[Dict[str, Any], AsyncGenerator]: Structured response dictionary or
            a streaming generator.

    Raises:
        ConfigurationException: If ACL_ENABLED=true. This convenience entry
            point bypasses the ACL gate; callers must use VideoQueryPipeline
            instead so the callback + user_identifier_context contract is
            enforced.
    """
    from config.provider_config import get_settings
    from mmct.utils.error_handler import ConfigurationException

    if get_settings().acl_enabled:
        raise ConfigurationException(
            "process_query() bypasses the ACL gate; use VideoQueryPipeline "
            "instead when ACL_ENABLED=true."
        )

    orchestrator = GraphOrchestrator(
        model_client=model_client,
        neo4j_provider=neo4j_provider,
        storage_provider=storage_provider,
        image_llm_provider=image_llm_provider,
        use_critic=use_critic,
    )

    if stream:
        return orchestrator.query_stream(query, video_id, video_ids)

    return await orchestrator.query(query, video_id, video_ids)
