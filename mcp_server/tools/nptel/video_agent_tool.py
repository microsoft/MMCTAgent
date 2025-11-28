"""MCP tools for NPTEL-specific video agents."""

from typing import Optional
import sys

from loguru import logger

from mcp_server.server import mcp
from mmct.video_pipeline.core.tools.nptel.video_qna import video_qna


@mcp.tool(
	name="video_agent_tool",
	description="""Answer lecture-style multimedia questions using the NPTEL-specific video_qna tool.

This tool runs the planner-only workflow that searches indexed lectures, pulls precise
segments/frames, and returns JSON with an answer plus cited video timestamps.

Inputs:
- query (required str): Natural language question.
- index_name (required str): Vector index containing the NPTEL corpus.
- url (optional str): Specific video URL to prioritize while still discovering other clips.

Output:
- answer: Markdown-formatted response with inline video citations.
- sources: List of cited videos with URL and HH:MM:SS segment ranges.
- tokens: Token usage from the orchestrator.
""",
)
async def video_agent_tool(
	query: str,
	index_name: str,
	url: Optional[str] = None,
):
	"""Expose the NPTEL video_qna planner as an MCP tool."""

	logger.remove()
	logger.add(
		sys.stdout,
		format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
		"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
		level="INFO",
		colorize=True,
	)

	logger.info("nptel_video_qna_tool invoked")
	logger.info("query={} | index={} | url={}", query, index_name, url or "<none>")

	try:
		return await video_qna(query=query, index_name=index_name, url=url)
	except Exception as exc:  # pragma: no cover - defensive logging
		logger.error("video_qna execution failed: %s", exc)
		logger.exception(exc)
		return {
			"answer": f"Error processing query: {exc}",
			"sources": [],
			"tokens": {"total_input": 0, "total_output": 0},
		}
