import tempfile, os
from fastapi import HTTPException, UploadFile
from loguru import logger
from mmct.video_pipeline import IngestionPipeline
from mmct.video_pipeline.utils.helper import get_file_hash, remove_file
from utilities.event_hub_handler import EventHubHandler
from mmct.blob_store_manager import BlobStorageManager
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    logger.info("Creating an instance of event hub handler for ingestion")
    ingestion_event_hub_handler = EventHubHandler(
        hub_name=os.getenv("INGESTION_EVENT_HUB_NAME")
    )
    logger.info(f"Successfully created Event Hub handler for: {os.getenv('INGESTION_EVENT_HUB_NAME')}")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating the Event Hub class: {e}")
    ingestion_event_hub_handler = None

blob_storage_manager = None  # Will be initialized async in functions

async def ingest_direct(file: UploadFile, body: dict):
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read()); tmp.close()
    path = tmp.name
    vid = await get_file_hash(path)
    try:
        pipeline = IngestionPipeline(
            video_path=path,
            index_name=body["index_name"],
            transcription_service=body["transcription_service"],
            language=body["language"],
            use_computer_vision_tool=body["use_computer_vision_tool"]
        )
        await pipeline()
    except Exception as e:
        logger.error(e); raise HTTPException(500, "Ingestion failed")
    finally:
        if os.path.exists(path):
            os.remove(path)
        await remove_file(video_id=vid)
    return {"message": "success"}

async def ingest_queue(file: UploadFile, body: dict):
    if ingestion_event_hub_handler is None:
        raise HTTPException(500, "Event Hub handler failed to initialize. Check Event Hub configuration and credentials.")
    
    # Initialize blob storage manager
    blob_storage_manager = await BlobStorageManager.create(account_url=os.getenv("BLOB_ACCOUNT_URL"))
    
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read()); tmp.close()
    path = tmp.name
    vid = await get_file_hash(path)
    ext = os.path.splitext(file.filename)[1]
    try:
        container_name = os.getenv("VIDEO_CONTAINER_NAME")
        logger.info(f"Uploading file {vid}{ext} to container {container_name}")
        
        blob_url = await blob_storage_manager.upload_file(
            container=container_name,
            blob_name=f"{vid}{ext}",
            file_path=path
        )
        logger.info(f"Successfully uploaded file to blob storage: {blob_url}")
        
        body['transcription_service'] = str(body['transcription_service'])
        body['language'] = str(body["language"])
        payload = body | {
            "video_id": vid,
            "video_blob_name": f"{vid}{ext}",
            "video_blob_url": blob_url
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
