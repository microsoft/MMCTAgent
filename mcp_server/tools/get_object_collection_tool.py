from typing import Annotated, Optional, List, Dict, Any
from mmct.video_pipeline.core.tools.get_object_collection import GetObjectCollection
from mcp_server.config import get_video_agent_provider
from mcp_server.server import mcp
from loguru import logger

try:
    logger.info("Instantiating GetObjectCollection")
    _provider = get_video_agent_provider()
    _get_object_collection = GetObjectCollection(
        vectordb_object_registry=_provider.vectordb_object_registry,
    )
    logger.info("Successfully instantiated GetObjectCollection")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating GetObjectCollection: {e}")


@mcp.tool(name="get_object_collection_tool")
async def get_object_collection_tool(
    object_names: Annotated[
        List[str], "extensive list of possible object names related to the query to retrieve"
    ],
    video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
    url: Annotated[Optional[str], "url of the video"] = None,
) -> List[Dict[str, Any]]:
    return await _get_object_collection.get_object_collection(
        object_names=object_names,
        video_id=video_id,
        url=url,
    )
