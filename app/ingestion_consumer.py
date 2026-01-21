"""
Ingestion Consumer for Event Hub-based video ingestion.

This script consumes ingestion events from Azure Event Hub, downloads videos
from blob storage, runs the IngestionPipeline with configured providers, and
performs cleanup.
"""
from mmct.video_pipeline import IngestionPipeline, Languages
from mmct.video_pipeline.utils.helper import get_media_folder, remove_file
from utilities.event_hub_handler import EventHubHandler
from utilities.execution_timer import ExecutionTimer
from azure.eventhub import EventData, PartitionContext
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from config import get_ingestion_provider
import asyncio
import json
import os
import sys

logger.add(sys.stdout, level="INFO", colorize=True)

load_dotenv(find_dotenv(), override=True)
    
try:
    logger.info("Creating an instance of event hub handler for ingestion")
    ingestion_event_hub_handler = EventHubHandler(hub_name=os.getenv("INGESTION_EVENT_HUB_NAME"))
    logger.info(f"Successfully created Event Hub handler for: {os.getenv('INGESTION_EVENT_HUB_NAME')}")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating the Event Hub class: {e}")
    raise
    
async def on_event(partition_context: PartitionContext, event: EventData):
    """
    Event handler for ingestion events from Event Hub.
    
    Args:
        partition_context: Event Hub partition context
        event: Event data containing ingestion payload
    """
    with ExecutionTimer() as timer:
        video_id = None
        try:
            logger.info("Ingestion Event Captured")
            payload = json.loads(event.body_as_str(encoding="UTF-8"))
            logger.info("Fetching the payload for the provided event")
            
            video_id = payload.get("video_id", None)
            language = payload.get("language", None)
            video_blob_name = payload.get('video_blob_name', None)
            video_blob_url = payload.get('video_blob_url', None)
            transcript_path = payload.get('transcript_path', None)
            url = payload.get('url', None)
            logger.info("Successfully fetched payload from the event hub!")

            logger.info("Creating an instance of blob storage manager to handle operations related to blob")
        
            # Parse language enum
            if language:
                # Handle both "Languages.ENGLISH_UNITED_STATES" and "ENGLISH_UNITED_STATES" formats
                language_value = language.split('.')[-1]
                language = Languages[language_value]

            if video_id:
                # Get provider configuration
                provider = get_ingestion_provider()
                blob_storage_manager = provider.storage_provider
                
                logger.info("Retrieving the video from the Blob!")
                media_folder = await get_media_folder()
                await blob_storage_manager.download_from_url(
                    file_url=video_blob_url, 
                    save_folder=media_folder
                )
                logger.info("Successfully retrieved the video from blob")

                # Create and run ingestion pipeline
                # Create and run ingestion pipeline
                ingestion = IngestionPipeline(
                    video_path=os.path.join(media_folder, video_blob_name),
                    video_id=video_id,
                    language=language,
                    transcript_path=transcript_path,
                    url=url,
                    provider=provider,
                    verbosity=2
                )
                await ingestion.run()
                
                logger.info(f"Successfully completed ingestion for video {video_id}")
            else:
                raise Exception("Exception occurred because video_id is NULL")

            # ✅ Mark the event as processed
            await partition_context.update_checkpoint(event)

        except Exception as e:
            logger.exception(f"Exception during ingestion: {e}")
        finally:
            if video_id:
                logger.info("Removing the media files")
                await remove_file(video_id=video_id)
            logger.info(f"Event processing completed in {timer.elapsed_time:.2f}s")

            

# Background consumer task
async def consume():
    """Main consumer loop for processing ingestion events."""
    logger.info("Starting ingestion consumer...")
    async with ingestion_event_hub_handler.consumer:
        await ingestion_event_hub_handler.consumer.receive(on_event=on_event)

if __name__=="__main__":
    asyncio.run(consume())