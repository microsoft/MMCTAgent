# Importing modules
import asyncio
import json
import os
import re
import logging
import time
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from loguru import logger

# Suppress autogen internal logging
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult

from mmct.video_pipeline.core.tools.nptel.prompts import PLANNER_DESCRIPTION, get_planner_system_prompt
from mmct.video_pipeline.core.tools.nptel.get_relevant_video_segments import get_relevant_video_segments
from mmct.video_pipeline.core.tools.nptel.get_relevant_videos import get_relevant_videos
from mmct.video_pipeline.core.tools.nptel.get_frame_analysis import get_frame_analysis
from mmct.video_pipeline.core.tools.nptel.ger_relevant_frames import get_relevant_frames

from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache as DiskCache
from mmct.providers.factory import provider_factory

load_dotenv(override=True)

def _configure_model_client():
    """Create (or reuse) the cached Autogen model client using a local disk cache."""

    llm_provider = provider_factory.create_llm_provider()
    model_client = llm_provider.get_autogen_client()

    cache_dir = os.getenv("AUTOGEN_DISK_CACHE_DIR", "./.autogen_ext_cache")
    store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](DiskCache(cache_dir))  # type: ignore

    return ChatCompletionCache(model_client, store)


MODEL_CLIENT = _configure_model_client()
PLANNER_PROMPT_BASE = get_planner_system_prompt()
TOOLS = [
    get_relevant_video_segments,
    get_relevant_videos,
    get_frame_analysis,
    get_relevant_frames,
]
TERMINATION_CONDITION = TextMentionTermination("TERMINATE")
def parse_response_to_dict(content: str) -> Dict[str, Any]:
    """Parse agent JSON output and normalize the mandatory source references."""

    def _format_timestamp(value: Any) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return "00:00:00"
            try:
                value = float(stripped)
            except ValueError:
                parts = stripped.split(":")
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    hours, minutes, seconds = (int(part) for part in parts)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                return stripped

        if isinstance(value, (int, float)):
            seconds_total = max(0, float(value))
            hours = int(seconds_total // 3600)
            minutes = int((seconds_total % 3600) // 60)
            seconds = int(seconds_total % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return "00:00:00"

    def _normalize_segments(segments: Any) -> List[List[str]]:
        normalized: List[List[str]] = []
        if isinstance(segments, list):
            for item in segments:
                if isinstance(item, dict):
                    start_val = item.get("start") or item.get("start_time")
                    end_val = item.get("end") or item.get("end_time") or start_val
                    normalized.append([
                        _format_timestamp(start_val),
                        _format_timestamp(end_val),
                    ])
                elif isinstance(item, (list, tuple)) and item:
                    start_val = item[0]
                    end_val = item[1] if len(item) > 1 else item[0]
                    normalized.append([
                        _format_timestamp(start_val),
                        _format_timestamp(end_val),
                    ])
                elif isinstance(item, (int, float, str)):
                    timestamp = _format_timestamp(item)
                    normalized.append([timestamp, timestamp])
        elif isinstance(segments, (int, float, str)):
            timestamp = _format_timestamp(segments)
            normalized.append([timestamp, timestamp])
        return normalized

    def _normalize_sources(raw_sources: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_sources, list):
            return []
        sources: List[Dict[str, Any]] = []
        for idx, entry in enumerate(raw_sources, start=1):
            if not isinstance(entry, dict):
                continue
            video_url = entry.get("video_url") or entry.get("url")
            if not video_url:
                continue
            source_entry = {
                "label": entry.get("label") or f"Video {idx}",
                "video_url": video_url,
                "segments": _normalize_segments(entry.get("segments") or entry.get("timestamps"))
            }
            if not source_entry["segments"] and entry.get("start_time") is not None:
                source_entry["segments"].append([
                    _format_timestamp(entry.get("start_time")),
                    _format_timestamp(entry.get("end_time") or entry.get("start_time"))
                ])
            sources.append(source_entry)
        return sources

    try:
        clean_content = content.replace("TERMINATE", "").strip()
        json_blocks = [
            match.group(1)
            for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", clean_content, re.DOTALL)
        ]
        if not json_blocks:
            brace_match = re.search(r"(\{.*\})", clean_content, re.DOTALL)
            if brace_match:
                json_blocks = [brace_match.group(1)]

        for block in json_blocks:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue

            answer = parsed.get("answer")
            sources = parsed.get("sources") or parsed.get("videos")
            if isinstance(answer, str) and sources is not None:
                normalized_sources = _normalize_sources(sources)
                if not normalized_sources:
                    logger.warning("Parsed response missing valid sources; continuing search")
                    continue
                return {
                    "answer": answer.strip(),
                    "sources": normalized_sources,
                }

        logger.warning("No valid JSON structure with sources found, returning fallback content")
        return {
            "answer": clean_content or "No response generated",
            "sources": [],
        }

    except Exception as exc:
        logger.error(f"Failed to parse response: {exc}")
        return {
            "answer": "Error parsing response",
            "sources": [],
        }


def _build_task(query: str, index_name: str, url: Optional[str]) -> str:
    lines = [
        f"User query: {query}",
        f"Vector index name: {index_name}",
    ]
    if url:
        lines.append(f"Focus video URL: {url}")
        lines.append("Prioritize this video while still citing any other supporting videos explicitly.")
    else:
        lines.append("No fixed video URL provided; search across multiple videos if needed.")
    lines.append("Always cite every source with video URL and timestamp range in seconds.")
    return "\n".join(lines)


def _build_system_message(index_name: str, url: Optional[str]) -> str:
    base_prompt = PLANNER_PROMPT_BASE
    context_lines = [
        f"Use the search index named '{index_name}' for all retrieval calls.",
        "Ensure the final JSON strictly follows the required schema.",
    ]
    if url:
        context_lines.append(f"The user supplied video URL filter is: {url}")
    else:
        context_lines.append("No URL filter supplied; discover relevant videos first.")
    return f"{base_prompt}\n\nContext:\n" + "\n".join(context_lines)


def _create_team(index_name: str, url: Optional[str]) -> RoundRobinGroupChat:
    planner_system_prompt = _build_system_message(index_name, url)

    planner = AssistantAgent(
        name="planner",
        model_client=MODEL_CLIENT,
        model_client_stream=False,
        description=PLANNER_DESCRIPTION,
        system_message=planner_system_prompt,
        tools=TOOLS,
        reflect_on_tool_use=True,
        max_tool_iterations=15,
        handoffs=[],
    )

    return RoundRobinGroupChat(
        participants=[planner],
        termination_condition=TERMINATION_CONDITION
    )


def _calculate_total_tokens(messages) -> dict:
    total_input = 0
    total_output = 0

    for message in messages:
        usage = getattr(message, "models_usage", None)
        if usage:
            total_input += getattr(usage, "prompt_tokens", 0) or 0
            total_output += getattr(usage, "completion_tokens", 0) or 0

    return {"total_input": total_input, "total_output": total_output}


async def _run_team_stream(team: RoundRobinGroupChat, task: str) -> tuple[str, Dict[str, int]]:
    """Execute the planner via streaming, returning the final message content and token usage."""

    response_generator = team.run_stream(task=task)
    streamed_items: List[Any] = []

    async for item in response_generator:
        if hasattr(item, "content") and getattr(item, "content"):
            print(f"Agent Message:{item.content}")
        streamed_items.append(item)

    tokens = {"total_input": 0, "total_output": 0}
    final_content = ""

    if not streamed_items:
        return final_content, tokens

    final_task = next((entry for entry in reversed(streamed_items) if isinstance(entry, TaskResult)), None)

    if final_task is not None:
        final_messages = final_task.messages or []
        tokens = _calculate_total_tokens(final_messages)
        if final_messages:
            final_content = final_messages[-1].content or ""
    else:
        last_item = streamed_items[-1]
        final_content = getattr(last_item, "content", str(last_item))

    return final_content, tokens


async def video_qna(
    query: Annotated[str, "Natural language question to be answered using lecture videos."],
    index_name: Annotated[str, "Vector index name that stores the multimedia content."],
    url: Annotated[Optional[str], "Optional target video URL to focus retrieval" ] = None,
) -> Dict[str, Any]:
    """Entry point exposed as a tool for answering video questions with mandatory citations."""

    start_time = time.perf_counter()
    task = _build_task(query=query, index_name=index_name, url=url)
    team = _create_team(index_name=index_name, url=url)
    final_content, tokens = await _run_team_stream(team=team, task=task)
    parsed_result = parse_response_to_dict(final_content)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(
        "video_qna latency: %.2f ms (query='%s', index='%s', url='%s')" % (
            elapsed_ms,
            query,
            index_name,
            url or "<none>",
        )
    )

    return {
        **parsed_result,
        "tokens": tokens,
    }


if __name__ == "__main__":
    example_query = "How and why are images represented as 2D matrices?"
    example_index = "kv-nptel-longer-chapter"
    example_url = "https://www.youtube.com/watch?v=9r8ph2pb9aw"

    response = asyncio.run(
        video_qna(
            query=example_query,
            index_name=example_index,
            url=example_url,
        )
    )
    print(json.dumps(response, indent=2))