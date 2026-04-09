"""Graph Orchestrator — coordinates the swarm-based (graph) pipeline.

Orchestrates the query pipeline with Neo4j graph backend:
1. Planner analyzes query and creates a retrieval plan
2. VideoAgent searches/traverses Neo4j graph
3. ImageAgent analyzes keyframes (if needed)
4. Critic validates answer (optional)
5. Planner synthesizes final answer with citations

Uses AutoGen Swarm for agent handoffs and BufferedChatCompletionContext
for context management.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, AsyncGenerator
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

from mmct.video_pipeline.graph.agents.planner_agent import PlannerAgent, SUBMIT_TOOL_NAME, reset_handoff_counter
from mmct.video_pipeline.graph.agents.video_agent import VideoAgent
from mmct.video_pipeline.graph.agents.image_agent import ImageAgent
from mmct.video_pipeline.graph.agents.critic_agent import CriticAgent
from mmct.video_pipeline.graph.query.neo4j_provider import Neo4jQueryProvider
from mmct.video_pipeline.graph.schemas import QueryResponse
from mmct.image_pipeline.config import ImageAgentProviderConfig


# ANSI color codes for console output
class Colors:
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# Buffer sizes for agent context windows
PLANNER_BUFFER_SIZE = 15
VIDEO_AGENT_BUFFER_SIZE = 12
IMAGE_AGENT_BUFFER_SIZE = 10
CRITIC_BUFFER_SIZE = 10

_TERMINATE_STRING = "TERMINATE"


@dataclass
class TokenUsage:
    """Tracks token usage across the query."""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _format_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _print_message_to_console(message, token_usage: TokenUsage, request_id: str = "") -> None:
    """Print a message to console with colored formatting."""
    source = getattr(message, "source", "Unknown")
    now = _format_timestamp()
    rid = f"[{request_id}] " if request_id else ""

    if hasattr(message, "models_usage") and message.models_usage:
        token_usage.prompt_tokens += message.models_usage.prompt_tokens
        token_usage.completion_tokens += message.models_usage.completion_tokens

    if isinstance(message, TaskResult):
        print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.MAGENTA}[System]{Colors.RESET}: Task Completed.", flush=True)
        print(f"\n{rid}{'-' * 10} {Colors.CYAN}Summary{Colors.RESET} {'-' * 10}", flush=True)
        print(f"{rid}Messages: {len(message.messages)}", flush=True)
        print(f"{rid}Finish reason: {message.stop_reason}", flush=True)
        print(f"{rid}Prompt tokens: {token_usage.prompt_tokens}", flush=True)
        print(f"{rid}Completion tokens: {token_usage.completion_tokens}", flush=True)
    elif isinstance(message, TextMessage):
        print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.GREEN}[{source}]{Colors.RESET}: {message.content}", flush=True)
    elif isinstance(message, HandoffMessage):
        print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.CYAN}[{source}]{Colors.RESET}: Handoff → {Colors.BOLD}{message.target}{Colors.RESET}", flush=True)
        if message.content:
            print(f"{Colors.GRAY}{rid}  Message: {message.content}{Colors.RESET}", flush=True)
    elif isinstance(message, ToolCallRequestEvent):
        content = message.content
        if isinstance(content, list):
            for tc in content:
                print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.BLUE}[{source}]{Colors.RESET}: Tool Call: {Colors.BOLD}{tc.name}{Colors.RESET}", flush=True)
                print(f"{Colors.GRAY}{rid}  Args: {tc.arguments}{Colors.RESET}", flush=True)
        else:
            print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.BLUE}[{source}]{Colors.RESET}: Tool Call: {content}", flush=True)
    elif isinstance(message, ToolCallExecutionEvent):
        content = message.content
        if isinstance(content, list):
            for tr in content:
                print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.YELLOW}[{source}]{Colors.RESET}: Tool Result ({tr.call_id}):", flush=True)
                print(f"{Colors.GRAY}{rid}{tr.content}{Colors.RESET}", flush=True)
        else:
            print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.YELLOW}[{source}]{Colors.RESET}: Tool Result: {content}", flush=True)
    else:
        content = str(getattr(message, "content", str(message)))
        print(f"\n{Colors.GRAY}[{now}]{Colors.RESET} {rid}{Colors.RED}[{source}]{Colors.RESET}: [{type(message).__name__}] {content}", flush=True)

    if hasattr(message, "models_usage") and message.models_usage and not isinstance(message, TaskResult):
        print(f"{Colors.GRAY}{rid}[Tokens: +{message.models_usage.prompt_tokens} prompt, +{message.models_usage.completion_tokens} completion]{Colors.RESET}", flush=True)


def _try_extract_json(content: str) -> Optional[str]:
    """Try to extract a JSON string from a message content."""
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
    """Extract the final answer JSON from swarm messages."""
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
    """Parse JSON response string into dictionary."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {"answer": json_str, "sources": [], "parse_error": str(e)}


class GraphOrchestrator:
    """Orchestrates the swarm-based (graph) multi-agent pipeline.

    Coordinates:
    - PlannerAgent: Query analysis, plan creation, answer synthesis
    - VideoAgent: Neo4j graph search and traversal
    - ImageAgent: Keyframe visual analysis (optional)
    - CriticAgent: Answer validation (optional)

    Uses AutoGen Swarm for agent handoffs.
    """

    def __init__(
        self,
        model_client,
        neo4j_provider: Neo4jQueryProvider,
        embedding_provider,
        image_embedding_provider=None,
        storage_provider=None,
        image_llm_provider=None,
        use_critic: bool = True,
        max_turns: int = 20,
        video_catalog: Optional[str] = None,
    ):
        """Initialize the orchestrator.

        Args:
            model_client: AutoGen model client (from get_llm_provider().get_autogen_client()).
            neo4j_provider: Neo4jQueryProvider instance.
            embedding_provider: Text embedding provider.
            image_embedding_provider: Optional image embedding provider.
            storage_provider: Optional storage provider for frame download.
            image_llm_provider: Optional LLM provider for ImageAgent.
            use_critic: Whether to use Critic validation.
            max_turns: Maximum Swarm conversation turns.
            video_catalog: Optional pre-generated video catalog for Planner context.
        """
        self.model_client = model_client
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.storage_provider = storage_provider
        self.image_llm_provider = image_llm_provider
        self.use_critic = use_critic
        self.max_turns = max_turns
        self.video_catalog = video_catalog

        self._agents_initialized = False
        self._image_agent_wrapper = None

    def _initialize_agents(self):
        """Initialize all agents with buffered contexts."""
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
            embedding_provider=self.embedding_provider,
            image_embedding_provider=self.image_embedding_provider,
            model_context=BufferedChatCompletionContext(buffer_size=VIDEO_AGENT_BUFFER_SIZE),
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
        """Build the AutoGen Swarm team."""
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
        video_ids: Optional[list] = None,
    ) -> str:
        """Build the task context string."""
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
        video_ids: Optional[list] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Process a query through the graph pipeline.

        Args:
            user_query: User's question about video content.
            video_id: Optional single video ID to scope search.
            video_ids: Optional list of video IDs to scope search.
            request_id: Optional request ID for log correlation.

        Returns:
            Dictionary with answer, sources, token_usage, elapsed_seconds.
        """
        start_time = time.time()
        token_usage = TokenUsage()
        rid = f"[{request_id}] " if request_id else ""

        now = _format_timestamp()
        print(f"\n{'=' * 60}", flush=True)
        print(f"{rid}{Colors.CYAN}Graph Query Pipeline{Colors.RESET}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"{Colors.GRAY}[{now}]{Colors.RESET} {rid}Query: {user_query}", flush=True)
        if video_id:
            print(f"{Colors.GRAY}[{now}]{Colors.RESET} {rid}Video: {video_id}", flush=True)
        elif video_ids:
            print(f"{Colors.GRAY}[{now}]{Colors.RESET} {rid}Videos: {video_ids}", flush=True)
        else:
            print(f"{Colors.GRAY}[{now}]{Colors.RESET} {rid}Scope: Cross-video search", flush=True)
        print(f"{'-' * 60}", flush=True)

        team = self._build_swarm()
        task = self._build_task(user_query, video_id, video_ids)

        try:
            final_result = None
            async for message in team.run_stream(task=task):
                _print_message_to_console(message, token_usage, request_id=request_id)
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

            print(f"\n{'=' * 60}", flush=True)
            print(f"{rid}{Colors.GREEN}Query completed in {elapsed:.2f}s{Colors.RESET}", flush=True)
            print(f"{'=' * 60}\n", flush=True)

            return response

        finally:
            if self._image_agent_wrapper:
                self._image_agent_wrapper.cleanup()

    async def query_stream(
        self,
        user_query: str,
        video_id: Optional[str] = None,
        video_ids: Optional[list] = None,
        request_id: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query with streaming output.

        Yields:
            {"type": "message", "agent": str, "content": str}
            {"type": "final", "data": {...same as query()...}}
        """
        start_time = time.time()
        token_usage = TokenUsage()
        rid = f"[{request_id}] " if request_id else ""

        now = _format_timestamp()
        print(f"\n{'=' * 60}", flush=True)
        print(f"{rid}{Colors.CYAN}Graph Query Pipeline (Streaming){Colors.RESET}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"{Colors.GRAY}[{now}]{Colors.RESET} {rid}Query: {user_query}", flush=True)
        print(f"{'-' * 60}", flush=True)

        team = self._build_swarm()
        task = self._build_task(user_query, video_id, video_ids)

        try:
            async for message in team.run_stream(task=task):
                _print_message_to_console(message, token_usage, request_id=request_id)

                if isinstance(message, TaskResult):
                    response = self._process_result(message)
                    elapsed = time.time() - start_time
                    response["elapsed_seconds"] = elapsed
                    response["token_usage"] = {
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                    }
                    print(f"\n{'=' * 60}", flush=True)
                    print(f"{rid}{Colors.GREEN}Query completed in {elapsed:.2f}s{Colors.RESET}", flush=True)
                    print(f"{'=' * 60}\n", flush=True)
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
        """Process TaskResult into response dictionary."""
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

    async def close(self):
        """Close resources and connections."""
        if self.neo4j_provider:
            await self.neo4j_provider.close()


async def process_query(
    query: str,
    model_client,
    neo4j_provider: Neo4jQueryProvider,
    embedding_provider,
    video_id: Optional[str] = None,
    video_ids: Optional[list] = None,
    image_embedding_provider=None,
    storage_provider=None,
    image_llm_provider=None,
    use_critic: bool = True,
    stream: bool = False,
) -> Dict[str, Any]:
    """Convenience function to process a query through the graph pipeline."""
    orchestrator = GraphOrchestrator(
        model_client=model_client,
        neo4j_provider=neo4j_provider,
        embedding_provider=embedding_provider,
        image_embedding_provider=image_embedding_provider,
        storage_provider=storage_provider,
        image_llm_provider=image_llm_provider,
        use_critic=use_critic,
    )

    if stream:
        return orchestrator.query_stream(query, video_id, video_ids)

    return await orchestrator.query(query, video_id, video_ids)
