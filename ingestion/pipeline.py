"""
Main ingestion pipeline for video keyframe extraction and indexing.
"""

import os
import logging
import shutil
from typing import Optional
from .processing import KeyframeExtractor, CLIPEmbeddingsGenerator
from .search import KeyframeSearchIndex
from .storage import BlobStorageManager
from .core import (
    ProcessingContext,
    get_file_hash,
    get_media_folder,
    KeyframeExtractionConfig,
    EmbeddingConfig,
    BlobStorageConfig,
    SearchIndexConfig
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main orchestration pipeline for keyframe extraction, embedding generation,
    blob storage upload, and Azure AI Search indexing.
    """

    def __init__(
        self,
        video_path: str,
        search_endpoint: str,
        index_name: str = "video-keyframes-index",
        storage_account_url: Optional[str] = None,
        storage_connection_string: Optional[str] = None,
        search_api_key: Optional[str] = None,
        keyframe_config: Optional[KeyframeExtractionConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        blob_config: Optional[BlobStorageConfig] = None,
        search_config: Optional[SearchIndexConfig] = None,
        youtube_url: Optional[str] = None
    ):
        """
        Initialize the ingestion pipeline.

        Authentication priority:
        - Azure CLI credentials (recommended)
        - DefaultAzureCredential (Managed Identity, etc.)
        - API keys / Connection strings (fallback)

        Args:
            video_path: Path to the video file
            search_endpoint: Azure AI Search endpoint URL
            index_name: Name of the search index
            storage_account_url: Azure Storage account URL (recommended - uses Azure CLI)
            storage_connection_string: Optional storage connection string (fallback)
            search_api_key: Optional Azure Search API key (fallback)
            keyframe_config: Optional keyframe extraction configuration
            embedding_config: Optional embedding generation configuration
            blob_config: Optional blob storage configuration
            search_config: Optional search index configuration
            youtube_url: Optional YouTube URL for the video
        """
        self.video_path = video_path
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.storage_account_url = storage_account_url
        self.storage_connection_string = storage_connection_string
        self.search_api_key = search_api_key
        self.youtube_url = youtube_url

        # Initialize configurations
        self.keyframe_config = keyframe_config or KeyframeExtractionConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.blob_config = blob_config or BlobStorageConfig()
        self.search_config = search_config or SearchIndexConfig(
            search_endpoint=search_endpoint,
            index_name=index_name
        )

        # Components (initialized on demand)
        self.keyframe_extractor = None
        self.embeddings_generator = None
        self.blob_manager = None
        self.search_index = None

    async def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize keyframe extractor
            self.keyframe_extractor = KeyframeExtractor(self.keyframe_config)
            logger.info("Initialized keyframe extractor")

            # Initialize embeddings generator
            self.embeddings_generator = CLIPEmbeddingsGenerator(self.embedding_config)
            logger.info("Initialized embeddings generator")

            # Initialize blob storage manager
            self.blob_manager = BlobStorageManager(
                storage_account_url=self.storage_account_url,
                connection_string=self.storage_connection_string,
                config=self.blob_config
            )
            logger.info("Initialized blob storage manager")

            # Initialize search index
            self.search_index = KeyframeSearchIndex(
                search_endpoint=self.search_endpoint,
                index_name=self.index_name,
                api_key=self.search_api_key,
                config=self.search_config
            )
            logger.info("Initialized search index manager")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _extract_keyframes(self, context: ProcessingContext) -> ProcessingContext:
        """Extract keyframes from video."""
        logger.info(f"Extracting keyframes from video: {context.video_path}")

        context.keyframe_metadata = await self.keyframe_extractor.extract_keyframes(
            video_path=context.video_path,
            video_id=context.video_id
        )

        logger.info(f"Extracted {len(context.keyframe_metadata)} keyframes")
        return context

    async def _generate_embeddings(self, context: ProcessingContext) -> ProcessingContext:
        """Generate CLIP embeddings for keyframes."""
        if not context.keyframe_metadata:
            logger.warning("No keyframes to generate embeddings for")
            return context

        logger.info(f"Generating embeddings for {len(context.keyframe_metadata)} keyframes")

        context.frame_embeddings = await self.embeddings_generator.process_frames(
            frame_metadata_list=context.keyframe_metadata,
            video_id=context.video_id
        )

        logger.info(f"Generated {len(context.frame_embeddings)} embeddings")
        return context

    async def _upload_to_blob_storage(self, context: ProcessingContext) -> ProcessingContext:
        """Upload keyframes to blob storage."""
        if not context.frame_embeddings:
            logger.warning("No frame embeddings to upload")
            return context

        logger.info(f"Uploading {len(context.frame_embeddings)} keyframes to blob storage")

        context.frame_embeddings = await self.blob_manager.upload_keyframes_batch(
            frame_embeddings=context.frame_embeddings,
            video_id=context.video_id
        )

        # Collect blob URLs
        context.blob_urls = [fe.blob_url for fe in context.frame_embeddings if fe.blob_url]

        logger.info(f"Uploaded {len(context.blob_urls)} keyframes to blob storage")
        return context

    async def _index_embeddings(self, context: ProcessingContext) -> ProcessingContext:
        """Index embeddings in Azure AI Search."""
        if not context.frame_embeddings:
            logger.warning("No frame embeddings to index")
            return context

        logger.info(f"Indexing {len(context.frame_embeddings)} embeddings to Azure AI Search")

        success = await self.search_index.upload_frame_embeddings(
            frame_embeddings=context.frame_embeddings,
            video_id=context.video_id,
            video_path=context.video_path,
            youtube_url=context.youtube_url
        )

        if success:
            logger.info("Successfully indexed embeddings")
        else:
            logger.warning("Indexing completed with warnings")

        return context

    async def _cleanup_local_files(self, context: ProcessingContext):
        """Clean up local keyframe files."""
        media_folder = await get_media_folder()
        keyframes_dir = os.path.join(media_folder, "keyframes", context.video_id)

        if os.path.exists(keyframes_dir):
            try:
                shutil.rmtree(keyframes_dir)
                logger.info(f"Cleaned up local keyframes directory: {keyframes_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up local files: {e}")

    async def run(self) -> ProcessingContext:
        """
        Run the complete ingestion pipeline.

        Returns:
            ProcessingContext with results
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting video ingestion pipeline")
            logger.info(f"Video: {self.video_path}")
            logger.info("=" * 80)

            # Generate video ID
            video_id = await get_file_hash(self.video_path)
            logger.info(f"Video ID: {video_id}")

            # Create processing context
            context = ProcessingContext(
                video_id=video_id,
                video_path=self.video_path,
                youtube_url=self.youtube_url
            )

            # Initialize components
            await self._initialize_components()

            # Run pipeline steps
            context = await self._extract_keyframes(context)
            context = await self._generate_embeddings(context)
            context = await self._upload_to_blob_storage(context)
            context = await self._index_embeddings(context)

            # Cleanup
            await self._cleanup_local_files(context)

            # Cleanup components
            if self.embeddings_generator:
                self.embeddings_generator.cleanup()
            if self.blob_manager:
                self.blob_manager.close()
            if self.search_index:
                await self.search_index.close()

            logger.info("=" * 80)
            logger.info("Ingestion pipeline completed successfully")
            logger.info(f"Total keyframes processed: {len(context.frame_embeddings)}")
            logger.info(f"Blob URLs created: {len(context.blob_urls)}")
            logger.info("=" * 80)

            return context

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            raise

    async def __call__(self) -> ProcessingContext:
        """Allow the pipeline to be called directly."""
        return await self.run()


async def run_ingestion(
    video_path: str,
    search_endpoint: str,
    index_name: str = "video-keyframes-index",
    storage_account_url: Optional[str] = None,
    storage_connection_string: Optional[str] = None,
    search_api_key: Optional[str] = None,
    motion_threshold: float = 0.8,
    sample_fps: int = 1,
    youtube_url: Optional[str] = None
) -> ProcessingContext:
    """
    Convenience function to run the ingestion pipeline.

    Authentication priority:
    - Azure CLI credentials (recommended - run 'az login' first)
    - DefaultAzureCredential (Managed Identity, etc.)
    - API keys / Connection strings (fallback)

    Args:
        video_path: Path to the video file
        search_endpoint: Azure AI Search endpoint URL
        index_name: Name of the search index
        storage_account_url: Azure Storage account URL (recommended)
        storage_connection_string: Optional storage connection string (fallback)
        search_api_key: Optional Azure Search API key (fallback)
        motion_threshold: Motion threshold for keyframe extraction
        sample_fps: Frames per second to sample
        youtube_url: Optional YouTube URL for the video

    Returns:
        ProcessingContext with results
    """
    keyframe_config = KeyframeExtractionConfig(
        motion_threshold=motion_threshold,
        sample_fps=sample_fps
    )

    pipeline = IngestionPipeline(
        video_path=video_path,
        search_endpoint=search_endpoint,
        index_name=index_name,
        storage_account_url=storage_account_url,
        storage_connection_string=storage_connection_string,
        search_api_key=search_api_key,
        keyframe_config=keyframe_config,
        youtube_url=youtube_url
    )

    return await pipeline.run()
