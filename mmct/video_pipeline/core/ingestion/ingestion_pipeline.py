import asyncio
import aiofiles
from typing import Optional, Annotated, Dict, List, Any
import os
import shutil
from loguru import logger
import gc

from mmct.config.providers import IngestionProviderConfig
from mmct.providers.base import  BaseTranscriptionProvider
from mmct.video_pipeline.utils.helper import (
    get_file_hash,
    get_media_folder,
)
from mmct.video_pipeline.core.ingestion.utils.helper import (
    check_video_already_ingested,
    split_video_if_needed,
    load_srt,
    split_transcript_by_time,
    adjust_transcript_timestamps,
    get_video_duration,
)

from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.core.ingestion.transcription.transcription_services import (
    TranscriptionServices,
)

from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import (
    KeyframeExtractionConfig,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_processor import (
    KeyframeProcessor,
)

from mmct.video_pipeline.core.ingestion.chapter_generator.chapter_ingestion_pipeline import (
    ChapterIngestionPipeline,
)
from mmct.video_pipeline.core.ingestion.video_compression.video_compression import VideoCompressor
from mmct.video_pipeline.core.ingestion.embedding_orchestrator import EmbeddingOrchestrator
from mmct.video_pipeline.core.ingestion.upload_orchestrator import UploadOrchestrator
from mmct.video_pipeline.core.ingestion.cleanup_manager import CleanupManager
from dotenv import load_dotenv, find_dotenv
from mmct.utils.logging_config import log_manager
from dataclasses import dataclass

# Load environment variables
load_dotenv(find_dotenv(), override=True)


@dataclass
class ProcessingContext:
    """Context object to hold processing state for a single video."""

    hash_id: str
    video_path: str
    video_extension: str
    transcript: Optional[str] = None
    transcript_path: Optional[str] = None
    frames: Optional[List] = None
    timestamps: Optional[List] = None
    base64_frames: Optional[List] = None
    blob_urls: Optional[Dict[str, str]] = None
    local_resources: Optional[List[str]] = None
    video_url: Optional[str] = None
    chapter_responses: Optional[Any] = None
    chapter_transcripts: Optional[Any] = None
    is_already_ingested: Optional[bool] = None
    parent_id: Optional[str] = None  # Original video ID (for both split and non-split cases)
    parent_duration: Optional[float] = None  # Original video duration in seconds
    video_duration: Optional[float] = None  # Duration of this specific video part in seconds

    def __post_init__(self):
        if self.blob_urls is None:
            self.blob_urls = {}
        if self.local_resources is None:
            self.local_resources = []


class IngestionPipeline:
    """
    IngestionPipeline handles the ingestion to prepare it for use with the VideoAgent system.

    Attributes:
        video_path (str): Path to the video file to be ingested.
        provider (IngestionProviderConfig): Configuration object containing all service providers
            (LLM, embedding ,Image embedding, vector database (for chapters, keyframes, object registry), storage, transcription, etc.) required for the ingestion pipeline.
        language (Languages, optional): Language of the video (only Languages Enum), used for transcription.
            Required when transcript_path is not provided. Defaults to None.
        url (str, optional): Optional URL associated with the video for video metadata.
        transcript_path (str, optional): Path to an existing transcript file (.srt format).
            When provided, transcription is skipped and language parameter is not required.
        disable_console_log (bool):
            Boolean flag to disable console logs. Default set to False.
        frame_stacking_grid_size (int): Grid size for frame stacking optimization.
            Values >1 enable stacking (e.g., 4 = 2x2 grid), 1 disables stacking. Defaults to 4.
    Example Usage:
    ---------------
    >>> from mmct.video_pipeline.ingestion import IngestionPipeline
    >>> from mmct.video_pipeline.language import Languages
    >>> from mmct.config.providers import IngestionProviderConfig
    >>> from mmct.providers.azure import (
    >>>     AzureLLMProvider,
    >>>     AzureEmbeddingProvider,
    >>>     AISearchChapterProvider,
    >>>     AISearchKeyframesProvider,
    >>>     AISearchObjectCollectionProvider,
    >>>     AzureStorageProvider,
    >>>     WhisperTranscriptionProvider
    >>>    )     # Note: Image Embedding provider is also required which is clip based provider.
    >>>    from mmct.providers.local import ClipImageEmbeddingProvider
    >>> import asyncio

    >>> async def run_ingestion():
    >>>     # Configure providers (transcription, LLM, embedding, search, storage, etc.)
    >>>     provider = IngestionProviderConfig(
    >>>        llm_provider=AzureOpenAILLMProvider(endpoint = "<some-endpoint>",api_version="<api-version>",...),
    >>>        embedding_provider=AzureOpenAIEmbeddingProvider(...),
    >>>        vectordb_chapter=AISearchChapterProvider(...),
    >>>        vectordb_object_registry=AISearchObjectCollectionProvider(...),
    >>>        vectordb_keyframes=AISearchKeyframesProvider(...),
    >>>        storage_provider=AzureBlobStorageProvider(...),
    >>>        image_embedding_provider=ClipImageEmbeddingProvider(...),
    >>>        transcription_provider=WhisperTranscriptionProvider(...)
    >>>     )  # Configure your providers
    >>>
    >>>     ingestion = IngestionPipeline(
    >>>         video_path="<valid-video-path>",
    >>>         provider=provider,
    >>>         language=Languages.TELUGU_INDIA
    >>>     )
    >>>     await ingestion.run()
    >>>
    >>> asyncio.run(run_ingestion())

    """

    def __init__(
        self,
        video_path: Annotated[str, "Local path to the video file to be ingested"],
        provider: Annotated[
            IngestionProviderConfig,
            "Configuration object containing all service providers (LLM, embedding, image embedding, search, storage, transcription, and vector database - chapters, object collection, keyframes) required for the ingestion pipeline",
        ],
        language: Annotated[
            Optional[Languages],
            "Language of the video (Languages Enum), required only when transcript_path is not provided",
        ] = None,
        url: Annotated[
            Optional[str], "Optional URL associated with the video for metadata enrichment"
        ] = None,
        transcript_path: Annotated[
            Optional[str],
            "Path to an existing transcript file (.srt); skips transcription if provided",
        ] = None,
        disable_console_log: Annotated[
            bool, "Boolean flag to disable console logs during ingestion"
        ] = False,
        frame_stacking_grid_size: Annotated[
            int, "Grid size for frame horizontal stacking (>1 enables stacking, 1 disables)"
        ] = 4,
        keyframe_config: Annotated[
            Optional[Dict[str, float]],
            "Configuration for keyframe extraction thresholds (e.g., { 'motion_threshold': 1.5, 'sample_fps': 2})",
        ] = {"motion_threshold": 1.5, "sample_fps": 2},
    ):
        try:
            logger.info("Successfully retrieved the MMCT config")
        except Exception as e:
            logger.exception(f"Exception occurred while fetching the MMCT config: {e}")
            raise Exception(f"Exception occurred while fetching the MMCT config: {e}")

        if disable_console_log == False:
            log_manager.enable_console()
        else:
            log_manager.disable_console()
        self.logger = log_manager.get_logger()

        # Validate that language is provided if transcript_path is not provided
        if not transcript_path and not language:
            raise ValueError("language parameter is required when transcript_path is not provided")

        self.keyframe_container = "keyframes"
        self.video_path = video_path
        self.transcript_path = transcript_path
        _, self.video_extension = os.path.splitext(self.video_path)
        self.url = url
        self.language = language
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.keyframe_config = keyframe_config
        self.original_video_path = video_path
        self.provider = provider

    async def _get_blob_manager(self):
        """
        Return the storage provider instance.
        """
        return self.provider.storage_provider

    async def _check_and_compress_video(self, video_path: str) -> str:
        """
        Check if video file size exceeds 500 MB and compress if needed.
        Runs compression in a thread pool to avoid blocking the event loop.

        Args:
            video_path: Path to the video file to check and compress

        Returns:
            str: Path to the video (compressed if needed, original otherwise)
        """
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            self.logger.info(
                f"Video file size for {os.path.basename(video_path)}: {file_size_mb:.2f} MB"
            )

            if file_size_mb > 500:
                self.logger.info(
                    f"Video file size ({file_size_mb:.2f} MB) exceeds 500 MB threshold. Starting compression..."
                )

                # Create compressed directory in media folder
                media_folder = await get_media_folder()
                compressed_dir = os.path.join(media_folder, "compressed")
                os.makedirs(compressed_dir, exist_ok=True)

                # Initialize video compressor
                compressor = VideoCompressor(
                    input_path=video_path, target_size_mb=500, output_dir=compressed_dir
                )

                # Compress the video in a thread pool to avoid blocking the event loop
                await asyncio.to_thread(compressor.compress)

                # Return compressed path if successful
                compressed_path = compressor.output_path
                if os.path.exists(compressed_path):
                    compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                    self.logger.info(
                        f"Video compressed successfully. New size: {compressed_size_mb:.2f} MB"
                    )
                    self.logger.info(f"Using compressed video: {compressed_path}")
                    return compressed_path
                else:
                    self.logger.warning("Compression failed, using original video")
                    return video_path
            else:
                self.logger.info(
                    "Video file size is within acceptable limits, no compression needed"
                )
                return video_path
        except Exception as e:
            self.logger.warning(f"Exception occurred during video compression check: {e}")
            return video_path

    async def _perform_early_ingestion_check(self) -> bool:
        """
        Perform early ingestion check to avoid unnecessary processing.

        Returns:
            bool: True if should continue processing, False if already ingested
        """
        try:
            self.logger.info("Performing early ingestion check...")

            # Generate hash ID for initial check
            video_hash_id = await get_file_hash(self.video_path)

            # Check if video already exists in the index
            is_already_ingested = await check_video_already_ingested(
                hash_id=video_hash_id, search_provider=self.provider.vectordb_chapter
            )

            if is_already_ingested:

                self.logger.info(
                    f"Video with hash_id {video_hash_id} already ingested. Skipping pipeline - no processing needed."
                )
                return False

            self.logger.info("Video not found in index. Proceeding with full ingestion pipeline...")
            return True

        except Exception as e:
            self.logger.exception(f"Exception occurred during early ingestion check: {e}")
            raise

    async def _process_single_video_part(
        self,
        video_path: str,
        part_hash_id: str,
        part_index: int,
        parent_id: str,
        parent_duration: float,
        video_split_time: Optional[float] = None,
    ) -> None:
        """
        Process a single video part with full ingestion pipeline.
        Handles compression, keyframe extraction, transcription, semantic chunking, and file uploads.

        Args:
            video_path: Path to the video part file
            part_hash_id: Hash ID for this specific video part
            part_index: Index of this part (0 for Part A, 1 for Part B)
            parent_id: Hash ID of the original video (before splitting)
            parent_duration: Duration of the original video in seconds
            video_split_time: Time in seconds where video was split (required if split into 2 parts)
        """
        try:
            self.logger.info(f"Starting processing of video part: {os.path.basename(video_path)}")
            self.logger.info(f"Part Hash ID: {part_hash_id}")

            # Step 1: Compress video if needed
            video_path = await self._check_and_compress_video(video_path)

            # Create processing context for this video part
            _, video_extension = os.path.splitext(video_path)
            # Calculate duration of this video part
            part_duration = await get_video_duration(video_path)
            context = ProcessingContext(
                hash_id=part_hash_id,
                video_path=video_path,
                video_extension=video_extension,
                transcript_path=None,  # Will be set in Step 3
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=part_duration,
            )

            # Get blob manager
            blob_manager = await self._get_blob_manager()

            # Set keyframes blob URL
            context.keyframes_blob_folder_url = await blob_manager.get_file_url(
                file_name=f"{context.hash_id}"
            )

            # ============================================================
            # PHASE 1: LOCAL PROCESSING (No external I/O)
            # ============================================================

            # Step 2: Extract keyframes and save metadata to JSON (no embeddings, no upload)
            self.logger.info(f"[PHASE 1] Starting local processing for {part_hash_id}")
            keyframe_config = KeyframeExtractionConfig(
                motion_threshold=self.keyframe_config["motion_threshold"],
                sample_fps=self.keyframe_config["sample_fps"],
            )
            keyframe_processor = KeyframeProcessor(
                keyframe_config=keyframe_config,
            )
            # For Part B (part_index=1), pass video_split_time as offset_time
            offset_time = video_split_time if part_index == 1 else None

            keyframe_json_path = await keyframe_processor.process_keyframes(
                video_path=video_path,
                video_hash_id=part_hash_id,
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=part_duration,
                offset_time=offset_time,
            )
            self.logger.info(f"[PHASE 1] Keyframe metadata saved to {keyframe_json_path}")

            # Step 3: Prepare transcript for this part
            transcript_path = None
            if self.transcript_path:
                if video_split_time is not None:
                    # Video was split - need to split transcript too
                    self.logger.info(f"Splitting transcript for part {part_index}...")
                    transcript_content = await load_srt(self.transcript_path)

                    # Split transcript by time to match video split
                    part_a_srt, part_b_srt = split_transcript_by_time(
                        transcript_content, video_split_time
                    )

                    # Select the appropriate part based on part_index
                    selected_transcript = part_a_srt if part_index == 0 else part_b_srt

                    # Save transcript chunk to temporary file
                    media_folder = await get_media_folder()
                    transcript_path = os.path.join(media_folder, f"transcript_{part_hash_id}.srt")
                    async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                        await f.write(selected_transcript)
                    self.logger.info(f"Created transcript chunk: {transcript_path}")
                else:
                    # Single video - use transcript as-is
                    transcript_path = self.transcript_path
                    context.transcript = await load_srt(transcript_path)
                    self.logger.info(f"Using provided transcript: {transcript_path}")

                # Update context with prepared transcript path
                context.transcript_path = transcript_path
            else:
                # Generate transcription
                # For Part B (part_index=1), pass offset to adjust timestamps
                time_offset = video_split_time if part_index == 1 else 0.0
                context = await self._get_transcription(context, time_offset=time_offset, transcription_provider = self.provider.transcription_provider)
                self.logger.info(f"[PHASE 1] Transcript generated for part {part_hash_id}")

            # Step 4: Generate semantic chapters and save to JSON (no embeddings, no upload)
            context = await self._generate_semantic_chapters(context = context, keyframe_index_name=self.provider.vectordb_keyframes.index_name, url = self.url)
            self.logger.info(
                f"[PHASE 1] Chapters and objects saved to JSON for part {part_hash_id}"
            )
            self.logger.info(f"[PHASE 1] Local processing completed successfully")

            # ============================================================
            # PHASE 2: PARALLEL EMBEDDING GENERATION
            # ============================================================
            self.logger.info(f"[PHASE 2] Starting parallel embedding generation for {part_hash_id}")
            embedding_orchestrator = EmbeddingOrchestrator(
                embedding_provider=self.provider.embedding_provider,
                image_embedding_provider=self.provider.image_embedding_provider,
            )
            await embedding_orchestrator.generate_all_embeddings(part_hash_id)
            self.logger.info(f"[PHASE 2] All embeddings generated successfully")

            # ============================================================
            # PHASE 3: BULK UPLOAD AND INDEXING
            # ============================================================
            self.logger.info(f"[PHASE 3] Starting parallel uploads and indexing for {part_hash_id}")
            upload_orchestrator = UploadOrchestrator(
                search_provider={"chapter":self.provider.vectordb_chapter, "object_collection":self.provider.vectordb_object_registry, "keyframe":self.provider.vectordb_keyframes},
                blob_manager=blob_manager,
            )
            await upload_orchestrator.upload_all(
                video_id=part_hash_id,
                url=self.url,
                keyframe_blob_url=context.keyframes_blob_folder_url,
            )
            self.logger.info(f"[PHASE 3] All uploads and indexing completed successfully")

            # ============================================================
            # PHASE 4: CLEANUP
            # ============================================================
            self.logger.info(f"[PHASE 4] Starting cleanup for {part_hash_id}")
            cleanup_manager = CleanupManager(keep_keyframes=False)
            await cleanup_manager.cleanup(part_hash_id)
            self.logger.info(f"[PHASE 4] Cleanup completed successfully")

            self.logger.info(f"Successfully processed video part: {part_hash_id}")

        except Exception as e:
            self.logger.exception(
                f"Exception occurred while processing video part {part_hash_id}: {e}"
            )
            raise

    async def _get_transcription(
        self, context: ProcessingContext, transcription_provider: BaseTranscriptionProvider, time_offset: float = 0.0, 
    ) -> ProcessingContext:
        """
        Generate transcription for video.

        Args:
            context: Processing context for the video
            time_offset: Time offset in seconds to add to all timestamps (for Part B videos)
        """
        try:
            self.logger.info(
                f"Using hash ID for video path: {context.video_path}\nHash Id: {context.hash_id}"
            )

            # Copy video file to hash_id.extension (keep original)
            video_dir = os.path.dirname(context.video_path)
            new_video_path = os.path.join(video_dir, f"{context.hash_id}{context.video_extension}")

            if context.video_path != new_video_path:
                shutil.copy2(context.video_path, new_video_path)
                context.video_path = new_video_path
                context.local_resources.append(new_video_path)  # Track renamed copy for cleanup
                self.logger.info(f"Video file copied to: {context.video_path}")

        
            self.logger.info(f"Using transcription provider: {transcription_provider.__class__.__name__}")
            output_dir = await get_media_folder()

            # Extract language name for translation (used by Azure Speech Service if needed)
            source_language_name = None
            if self.language:
                source_language_name = self.language.name.split("_")[0].capitalize()

            # Call provider's transcribe_video method with appropriate parameters
            context.transcript, local_paths = await transcription_provider.transcribe_video(
                video_path=context.video_path,
                hash_id=context.hash_id,
                output_dir=output_dir,
                language=self.language.value if self.language else None,
                translate_to_english=True,
                source_language_name=source_language_name,
                response_format="srt"  # Used by Whisper, ignored by Azure Speech Service
            )
            self.logger.info("Successfully generated the transcript using transcription provider.")

            # Adjust transcript timestamps if offset is provided (for Part B videos)
            if time_offset > 0:
                self.logger.info(f"Adjusting transcript timestamps by offset: {time_offset}s")
                context.transcript = adjust_transcript_timestamps(
                    context.transcript, time_offset
                )
                # Save adjusted transcript
                adjusted_transcript_path = os.path.join(
                    await get_media_folder(), f"transcript_{context.hash_id}.srt"
                )
                async with aiofiles.open(adjusted_transcript_path, "w", encoding="utf-8") as f:
                    await f.write(context.transcript)
                self.logger.info(f"Saved adjusted transcript to {adjusted_transcript_path}")

            context.local_resources.extend(local_paths)
            del local_paths
            gc.collect()
            return context
        except Exception as e:
            self.logger.exception(f"Exception occured while performing transcription: {e}")
            raise

    async def _generate_semantic_chapters(
        self, context: ProcessingContext,keyframe_index_name:str, url: Optional[str] = None
    ) -> ProcessingContext:
        """
        Generate semantic chapters from transcript using the chapter ingestion pipeline.
        Creates chapters and indexes them for search and retrieval.
        """
        try:
            self.logger.info(
                "Creating an instance of ChapterIngestionPipeline to orchestrate chapter generation"
            )
            chapter_pipeline = ChapterIngestionPipeline(
                hash_id=context.hash_id,
                keyframe_index_name = keyframe_index_name,
                transcript=context.transcript,
                keyframe_blob_url=context.keyframes_blob_folder_url,
                frame_stacking_grid_size=self.frame_stacking_grid_size,
                parent_id=context.parent_id,
                parent_duration=context.parent_duration,
                video_duration=context.video_duration,
                llm_provider=self.provider.llm_provider,
                embedding_provider=self.provider.embedding_provider,
            )
            self.logger.info("Successfully created an instance of ChapterIngestionPipeline!")

            context.chapter_responses, context.chapter_transcripts, context.is_already_ingested = (
                await chapter_pipeline.run(url=url)
            )

            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while creating an instance of ChapterIngestionPipeline: {e}"
            )
            raise

    async def _validate_audio_stream_exists(self, video_path: str) -> bool:
        """
        Validate that the video file contains an audio stream.

        Args:
            video_path: Path to the video file

        Returns:
            bool: True if audio stream exists, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()
            return stdout and stdout.strip() == b"audio"
        except Exception as e:
            self.logger.warning(f"Could not check for audio stream: {e}")
            return False

    async def run(self):
        """Main ingestion pipeline method - now supports video splitting and parallel processing."""
        try:
            # Early ingestion check - exit immediately if already processed
            should_continue = await self._perform_early_ingestion_check()
            if not should_continue:
                return

            # Validate audio stream exists (only if transcript_path not provided)
            if not self.transcript_path:
                self.logger.info("Validating video has audio stream...")
                has_audio = await self._validate_audio_stream_exists(self.video_path)
                if not has_audio:
                    error_msg = (
                        "ERROR: Video does not have an audio stream!\n"
                        "Please provide either:\n"
                        "  1. A video file with audio, OR\n"
                        "  2. A transcript file using the transcript_path parameter"
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                self.logger.info("Video has audio stream - proceeding with transcription")

            # Calculate parent video metadata (original video before any splitting)
            parent_video_id = await get_file_hash(file_path=self.original_video_path)
            parent_video_duration = await get_video_duration(self.video_path)
            self.logger.info(
                f"Parent video ID: {parent_video_id}, Duration: {parent_video_duration:.2f}s"
            )

            # Split video if needed based on duration (>= 30 minutes)
            video_paths, hash_suffixes = await split_video_if_needed(self.video_path)
            self.logger.info(f"Processing {len(video_paths)} video part(s)")

            # Track split video files for cleanup
            split_video_cleanup_paths = []
            if len(video_paths) > 1:
                split_video_cleanup_paths.extend(video_paths)

            # Use parent_video_id as base hash ID
            base_hash_id = parent_video_id

            # Calculate video split time (only needed if video was split into 2 parts)
            video_split_time = parent_video_duration / 2 if len(video_paths) == 2 else None

            # Create tasks for parallel processing (compression, keyframe extraction, transcription per part)
            tasks = []
            for idx, (video_path, hash_suffix) in enumerate(zip(video_paths, hash_suffixes)):
                part_name = "Part A" if hash_suffix == "" else f"Part {hash_suffix}"
                part_hash_id = base_hash_id + hash_suffix

                self.logger.info(f"Creating task for {part_name}: {os.path.basename(video_path)}")
                self.logger.info(f"  Hash ID: {part_hash_id}")

                # Create asyncio task for processing this video part
                task = asyncio.create_task(
                    self._process_single_video_part(
                        video_path=video_path,
                        part_hash_id=part_hash_id,
                        part_index=idx,
                        parent_id=parent_video_id,
                        parent_duration=parent_video_duration,
                        video_split_time=video_split_time,
                    )
                )
                tasks.append(task)

            # Execute all video parts (single or multiple) in parallel
            processing_mode = "parallel" if len(video_paths) > 1 else "single"
            self.logger.info(
                f"Starting {processing_mode} processing of {len(tasks)} video part(s)..."
            )
            await asyncio.gather(*tasks)

            self.logger.info("All video parts processed successfully!")

            # Clean up split video files if any were created
            for split_video_path in split_video_cleanup_paths:
                try:
                    if os.path.exists(split_video_path):
                        os.remove(split_video_path)
                        self.logger.info(f"Removed split video file: {split_video_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove split video file {split_video_path}: {e}"
                    )

            self.logger.info("Local files cleaned up successfully!")

            self.logger.info("Ingestion pipeline ran successfully!")

        except Exception as e:
            self.logger.exception(f"Exception occurred while running Ingestion pipeline: {e}")
            raise


if __name__ == "__main__":
    # Example usage - replace with your actual values
    video_path = "/home/v-amanpatkar/work/demo/What Makes People Engage With Math  Grant Sanderson  TEDxBerkeley.mp4"  # "video-path"
    index = "test"
    url = "video-url-2"
    source_language = Languages.ENGLISH_UNITED_STATES
    transcript_path = "transcript.srt"  # Optional: path to existing transcript file
    keyframe_config = {"motion_threshold": 1.5, "sample_fps": 2}
    ingestion = IngestionPipeline(
        video_path=video_path,
        index_name=index,
        url=url,
        transcription_service=TranscriptionServices.WHISPER,
        language=source_language,
        # transcript_path=transcript_path,
        keyframe_config=keyframe_config,
    )
    asyncio.run(ingestion.run())
