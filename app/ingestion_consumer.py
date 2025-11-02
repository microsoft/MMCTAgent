from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.video_pipeline.utils.helper import remove_file
from mmct.providers.factory import provider_factory
from utilities.event_hub_handler import EventHubHandler
from utilities.execution_timer import ExecutionTimer
from azure.eventhub import EventData, PartitionContext
from dotenv import load_dotenv, find_dotenv
from loguru import logger
import asyncio
import json
import os
import sys

logger.add(sys.stdout, level="INFO", colorize=True)

load_dotenv(find_dotenv(), override=True)
    
try:
    logger.info("Creating an instance of event hub handler for ingestion")
    ingestion_event_hub_handler = EventHubHandler(hub_name=os.getenv("INGESTION_EVENT_HUB_NAME"))
except Exception as e:
    logger.exception(f"Exception occured while instantiating the Event Hub class: {e}")
    raise

blob_storage_manager = provider_factory.create_storage_provider()
    
async def on_event(partition_context: PartitionContext, event: EventData):
    with ExecutionTimer() as timer:
        try:
            logger.info("Ingestion Event Captured")
            payload = json.loads(event.body_as_str(encoding="UTF-8"))
            logger.info("Fetching the payload for the provided event")
            
            index_name = payload.get("index_name", None)
            video_id = payload.get("video_id", None)
            language = payload.get("language", None)
            transcription_service = payload.get("transcription_service", None)
            use_computer_vision_tool = payload.get("use_computer_vision_tool", None)
            video_blob_name = payload.get('video_blob_name', None)
            video_blob_url = payload.get('video_blob_url', None)
            logger.info("Successfully fetched payload from the event hub!")

            logger.info("Creating an instance of blob storage manager to handle operations related to blob")
        
            transcription_service = transcription_service.split('.')[-1]
            language = language.split('.')[-1]

            if video_id:
                logger.info("Retrieving the video from the Blob!")
                await blob_storage_manager.download_from_url(file_url=video_blob_url, save_folder=await get_media_folder())
                logger.info("Successfully retrieved the video from blob")

                ingestion = IngestionPipeline(
                    hash_video_id=video_id,
                    video_path=os.path.join(await get_media_folder(), video_blob_name),
                    index_name=index_name,
                    transcription_service=TranscriptionServices[transcription_service],
                    language=Languages[language],
                    use_computer_vision_tool=use_computer_vision_tool
                )
                await ingestion()
            else:
                raise Exception("Exception occurred because video_id is NULL")

            # ✅ Mark the event as processed
            await partition_context.update_checkpoint(event)

        except Exception as e:
            logger.exception(f"Exception: {e}")
        finally:
            if video_id:
                logger.info("Removing the media files")
                await remove_file(video_id=video_id)

            

# Background consumer task
async def consume():
    async with ingestion_event_hub_handler.consumer:
        await ingestion_event_hub_handler.consumer.receive(on_event=on_event)

if __name__=="__main__":
    asyncio.run(consume())