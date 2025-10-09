"""
Example usage of the Lively video ingestion pipeline.

To run this example:
    python -m ingestion.example
Or from parent directory:
    python -c "from ingestion import example; import asyncio; asyncio.run(example.main())"
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Use try-except to support both direct execution and module import
try:
    # When run as module: python -m ingestion.example
    from .pipeline import run_ingestion, IngestionPipeline
    from .core import (
        KeyframeExtractionConfig,
        EmbeddingConfig,
        BlobStorageConfig,
        SearchIndexConfig
    )
except ImportError:
    # When run directly: python example.py (from ingestion directory)
    from pipeline import run_ingestion, IngestionPipeline
    from core import (
        KeyframeExtractionConfig,
        EmbeddingConfig,
        BlobStorageConfig,
        SearchIndexConfig
    )

# Load environment variables from lively/.env
load_dotenv(Path(__file__).parent.parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def simple_example():
    """Simple example using the convenience function with Azure CLI authentication."""
    logger.info("Running simple example...")

    try:
        # Recommended: Use Azure CLI authentication (run 'az login' first)
        result = await run_ingestion(
            video_path="/home/v-amanpatkar/work/demo/videoplayback.mp4",
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name="test_lively",
            storage_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            # Optional fallbacks:
            # storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            # search_api_key=os.getenv("SEARCH_API_KEY"),
            motion_threshold=2.5,
            sample_fps=1
        )

        logger.info("=" * 80)
        logger.info("Simple Example Results:")
        logger.info(f"  Video ID: {result.video_id}")
        logger.info(f"  Keyframes extracted: {len(result.keyframe_metadata)}")
        logger.info(f"  Embeddings generated: {len(result.frame_embeddings)}")
        logger.info(f"  Blob URLs created: {len(result.blob_urls)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Simple example failed: {e}")
        raise


async def advanced_example():
    """Advanced example with custom configurations."""
    logger.info("Running advanced example...")

    try:
        # Custom keyframe extraction configuration
        keyframe_config = KeyframeExtractionConfig(
            motion_threshold=1.0,      # Higher threshold = fewer keyframes
            sample_fps=2,               # Sample 2 frames per second
            max_frame_width=800,        # Resize for faster processing
            debug_mode=False            # Don't keep local files
        )

        # Custom embedding configuration
        embedding_config = EmbeddingConfig(
            clip_model_name="openai/clip-vit-base-patch32",
            batch_size=16,              # Process 16 images at once
            device="auto",              # Use GPU if available
            max_image_size=224
        )

        # Custom blob storage configuration
        blob_config = BlobStorageConfig(
            container_name="keyframes",
            upload_batch_size=10,
            generate_sas_token=True,    # Generate SAS URLs for secure access
            sas_expiry_hours=24
        )

        # Custom search index configuration
        search_config = SearchIndexConfig(
            index_name="video-keyframes-index",
            embedding_model="openai/clip-vit-base-patch32",  # Will determine vector_dimensions automatically
            hnsw_m=4,
            hnsw_ef_construction=400,
            hnsw_ef_search=500,
            batch_upload_size=100
        )

        # Create pipeline with custom configurations
        # Uses Azure CLI authentication by default
        pipeline = IngestionPipeline(
            video_path="/path/to/your/video.mp4",
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name="video-keyframes-index",
            storage_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            # Optional fallbacks:
            # storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            # search_api_key=os.getenv("SEARCH_API_KEY"),
            keyframe_config=keyframe_config,
            embedding_config=embedding_config,
            blob_config=blob_config,
            search_config=search_config
        )

        # Run the pipeline
        result = await pipeline.run()

        logger.info("=" * 80)
        logger.info("Advanced Example Results:")
        logger.info(f"  Video ID: {result.video_id}")
        logger.info(f"  Keyframes extracted: {len(result.keyframe_metadata)}")
        logger.info(f"  Embeddings generated: {len(result.frame_embeddings)}")
        logger.info(f"  Blob URLs created: {len(result.blob_urls)}")
        logger.info("=" * 80)

        # Print sample keyframe information
        if result.keyframe_metadata:
            logger.info("\nSample Keyframes:")
            for i, frame in enumerate(result.keyframe_metadata[:5]):
                logger.info(
                    f"  Frame {frame.frame_number}: "
                    f"{frame.timestamp_seconds:.2f}s "
                    f"(motion: {frame.motion_score:.3f})"
                )

    except Exception as e:
        logger.error(f"Advanced example failed: {e}")
        raise


async def batch_processing_example():
    """Example of processing multiple videos."""
    logger.info("Running batch processing example...")

    video_paths = [
        "/path/to/video1.mp4",
        "/path/to/video2.mp4",
        "/path/to/video3.mp4",
    ]

    results = []

    for video_path in video_paths:
        try:
            logger.info(f"\nProcessing: {video_path}")

            result = await run_ingestion(
                video_path=video_path,
                search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
                index_name="video-keyframes-index",
                storage_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
                motion_threshold=0.8,
                sample_fps=1
            )

            results.append({
                'video_path': video_path,
                'video_id': result.video_id,
                'keyframes': len(result.keyframe_metadata),
                'embeddings': len(result.frame_embeddings)
            })

            logger.info(f"✓ Completed: {video_path}")

        except Exception as e:
            logger.error(f"✗ Failed to process {video_path}: {e}")
            continue

    # Summary
    logger.info("=" * 80)
    logger.info("Batch Processing Summary:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n{i}. {os.path.basename(result['video_path'])}")
        logger.info(f"   Video ID: {result['video_id']}")
        logger.info(f"   Keyframes: {result['keyframes']}")
        logger.info(f"   Embeddings: {result['embeddings']}")
    logger.info("=" * 80)


async def main():
    """Main function to run examples."""
    print("\n" + "=" * 80)
    print("Lively Video Ingestion Pipeline - Examples")
    print("=" * 80 + "\n")

    # Choose which example to run
    print("Select an example to run:")
    print("1. Simple example (quick start)")
    print("2. Advanced example (custom configurations)")
    print("3. Batch processing example (multiple videos)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        await simple_example()
    elif choice == "2":
        await advanced_example()
    elif choice == "3":
        await batch_processing_example()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    asyncio.run(main())
