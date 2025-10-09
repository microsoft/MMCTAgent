"""
Standalone example script for the Lively video ingestion pipeline.
Run this from the lively directory: python run_example.py
"""

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from ingestion import run_ingestion

# Load environment variables from lively/.env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log environment variable status
logger.info(f"Loading .env from: {env_path}")
logger.info(f"SEARCH_SERVICE_ENDPOINT: {os.getenv('SEARCH_SERVICE_ENDPOINT')}")
logger.info(f"AZURE_STORAGE_ACCOUNT_URL: {os.getenv('AZURE_STORAGE_ACCOUNT_URL')}")


async def main():
    """Run a simple ingestion example."""
    logger.info("=" * 80)
    logger.info("Lively Video Ingestion Pipeline - Simple Example")
    logger.info("=" * 80)

    try:
        # Make sure you've run 'az login' first!
        result = await run_ingestion(
            video_path="/home/v-amanpatkar/work/demo/What Makes People Engage With Math  Grant Sanderson  TEDxBerkeley.mp4",
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX_NAME"),
            storage_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            # Optional fallbacks (uncomment if not using Azure CLI):
            # storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            # search_api_key=os.getenv("SEARCH_API_KEY"),
            motion_threshold=1.5,
            sample_fps=5,
            # Optional: YouTube URL for the video
            youtube_url="https://www.youtube.com/watch?v=s_L-fp8gDzY"
        )

        logger.info("=" * 80)
        logger.info("Results:")
        logger.info(f"  Video ID: {result.video_id}")
        logger.info(f"  Keyframes extracted: {len(result.keyframe_metadata)}")
        logger.info(f"  Embeddings generated: {len(result.frame_embeddings)}")
        logger.info(f"  Blob URLs created: {len(result.blob_urls)}")
        logger.info("=" * 80)

        logger.info("\nSuccess! ✓")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
