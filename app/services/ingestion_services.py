import tempfile, os
from fastapi import HTTPException, UploadFile
from loguru import logger
from mmct.video_pipeline import IngestionPipeline
from mmct.video_pipeline.utils.helper import get_file_hash, remove_file
from utilities.event_hub_handler import EventHubHandler
from dotenv import load_dotenv
from config import get_ingestion_provider

load_dotenv(override=True)

try:
    logger.info("Creating an instance of event hub handler for ingestion")
    ingestion_event_hub_handler = EventHubHandler(hub_name=os.getenv("INGESTION_EVENT_HUB_NAME"))
    logger.info(
        f"Successfully created Event Hub handler for: {os.getenv('INGESTION_EVENT_HUB_NAME')}"
    )
except Exception as e:
    logger.exception(f"Exception occurred while instantiating the Event Hub class: {e}")
    ingestion_event_hub_handler = None


async def ingest_direct(file: UploadFile, body: dict):
    """
    Direct video ingestion using IngestionPipeline with configured providers.

    Args:
        file: Uploaded video file
        body: Request body containing ingestion parameters

    Returns:
        Success message
    """
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    path = tmp.name
    vid = await get_file_hash(path)

    try:
        # Get provider configuration
        provider = get_ingestion_provider()

        # Create IngestionPipeline with provider
        pipeline = IngestionPipeline(
            video_path=path,
            video_id=vid,
            language=body["language"],
            transcript_path=body.get("transcript_path"),
            url=body.get("url"),
            provider=provider,
            verbosity=2,
        )
        await pipeline.run()

        logger.info(f"Successfully ingested video {vid}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)
        await remove_file(video_id=vid)

    return {"message": "success", "video_id": vid}


async def ingest_queue(file: UploadFile, body: dict):
    """
    Queue-based video ingestion - uploads to blob storage and sends event to Event Hub.

    Args:
        file: Uploaded video file
        body: Request body containing ingestion parameters

    Returns:
        Success or failure message
    """
    if ingestion_event_hub_handler is None:
        raise HTTPException(
            500,
            "Event Hub handler failed to initialize. Check Event Hub configuration and credentials.",
        )

    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    path = tmp.name
    vid = await get_file_hash(path)
    ext = os.path.splitext(file.filename)[1]

    try:
        # Get storage provider from ingestion config
        provider = get_ingestion_provider()
        blob_storage_manager = provider.storage_provider

        container_name = os.getenv("VIDEO_CONTAINER_NAME")
        logger.info(f"Uploading file {vid}{ext} to container {container_name}")

        blob_url = await blob_storage_manager.upload_file(
            folder_name=container_name, file_name=f"{vid}{ext}", src_file_path=path
        )
        logger.info(f"Successfully uploaded file to blob storage: {blob_url}")

        # Prepare payload for event hub
        payload = {
            "video_id": vid,
            "video_blob_name": f"{vid}{ext}",
            "video_blob_url": blob_url,
            "language": str(body["language"]),
            "transcript_path": body.get("transcript_path"),
            "url": body.get("url"),
        }

        logger.info(f"Sending event to Event Hub with payload: {payload}")
        message = await ingestion_event_hub_handler.produce_event(payload=payload)
        logger.info(f"Event Hub response: {message}")

        if message.get("success"):
            return {"message": "produced event"}
        return {"message": f"fail: {message.get('message')}"}
    except Exception as e:
        logger.exception(f"Queue ingestion failed: {e}")
        raise HTTPException(500, f"Queue ingestion failed: {str(e)}")
    finally:
        os.remove(path)
        await remove_file(video_id=vid)
