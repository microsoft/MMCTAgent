"""MCP tool exposing the NPTEL get_relevant_frames utility."""

import sys
from typing import Optional

from loguru import logger

from mcp_server.server import mcp
from mmct.video_pipeline.core.tools.nptel.ger_relevant_frames import get_relevant_frames


@mcp.tool(
	name="get_relevant_frames_tool",
	description="""Return visually-relevant frame timestamps for a YouTube lecture.

Inputs:
- query (str, required): Visual description to search for (e.g., "derivative plot").
- url (str, required): YouTube URL to inspect.
- index_name (str, required): Base index name; the keyframe index is derived internally.
- top_k (int, optional): Number of frames to retrieve (default 5).
- start_time / end_time (float, optional): Restrict search to a time window (seconds).

Output: JSON array of {"timestamp": float, "blob_url": str | None} entries.
""",
)
async def get_relevant_frames_tool(
	query: str,
	url: str,
	index_name: str,
	top_k: int = 5,
	start_time: Optional[float] = None,
	end_time: Optional[float] = None,
):
	"""Wrapper that calls get_relevant_frames and returns the raw matches."""

	logger.remove()
	logger.add(
		sys.stdout,
		format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
		"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
		level="INFO",
		colorize=True,
	)

	logger.info(
		"nptel_get_relevant_frames_tool invoked | query={} | url={} | index={} | top_k={} | window=({},{})",
		query,
		url,
		index_name,
		top_k,
		start_time,
		end_time,
	)

	try:
		frames = await get_relevant_frames(
			query=query,
			url=url,
			index_name=index_name,
			top_k=top_k,
			start_time=start_time,
			end_time=end_time,
		)
		return {"frames": frames}
	except Exception as exc:  # pragma: no cover - defensive logging
		logger.error("get_relevant_frames failed: %s", exc)
		logger.exception(exc)
		return {"frames": [], "error": str(exc)}
