import tempfile
import os
from pathlib import Path
from fastapi import HTTPException, UploadFile
from loguru import logger
from mmct.video_pipeline import IngestionPipeline
from mmct.video_pipeline.utils.helper import get_file_hash, remove_file
from app.config import get_ingestion_provider

# Path to temporal graph ingestion pipeline config
TEMPORAL_GRAPH_PIPELINE_CONFIG = Path(__file__).parent.parent.parent.parent / \
    "mmct/video_pipeline/core/ingestion/experiments/temporal_graph_ingestion.yaml"


async def ingest_graph_direct(file: UploadFile, body: dict):
    """
    Direct graph-based video ingestion using IngestionPipeline with temporal graph extraction.

    Uses temporal_graph_ingestion.yaml pipeline config with:
    - Local fastembed models for event/object embeddings (384-dim, CPU)
    - Parallel chapter processing for events and objects
    - Batch object deduplication using embedding similarity

    Args:
        file: Uploaded video file
        body: Request body containing ingestion parameters

    Returns:
        Success message with video_id and extraction stats
    """
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    path = tmp.name
    vid = await get_file_hash(path)

    try:
        # Get ingestion provider configuration
        provider = get_ingestion_provider()

        # Create IngestionPipeline with temporal graph pipeline config
        pipeline = IngestionPipeline(
            video_path=path,
            video_id=vid,
            language=body["language"],
            transcript_path=body.get("transcript_path"),
            url=body.get("url"),
            provider=provider,
            pipeline_config_path=str(TEMPORAL_GRAPH_PIPELINE_CONFIG),
            verbosity=2,
        )
        await pipeline.run()

        logger.info(f"Successfully ingested video {vid} with temporal graph extraction")

    except Exception as e:
        logger.error(f"Graph ingestion failed: {e}")
        raise HTTPException(500, f"Graph ingestion failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)
        await remove_file(video_id=vid)

    return {
        "message": "success",
        "video_id": vid,
        "pipeline": "graph_ingestion",
    }
