import asyncio
import aiofiles
from typing import Optional, Annotated, Dict, List, Tuple, Any
import os
import shutil
from loguru import logger
import gc
from mmct.video_pipeline.core.ingestion.transcription.cloud_transcription import CloudTranscription
from mmct.video_pipeline.core.ingestion.transcription.whisper_transcription import (
    WhisperTranscription,
)
from mmct.video_pipeline.utils.helper import (
    get_file_hash,
    extract_frames,
    save_frames_as_png,
    file_upload_to_blob,
    encode_frames_to_base64,
    chunked,
    remove_file,
    check_video_already_ingested,
    split_video_if_needed,
    load_srt,
    split_transcript_by_segments,
    get_video_duration
)

from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.core.ingestion.transcription.transcription_services import (
    TranscriptionServices,
)

from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import (
    KeyframeExtractor,
    KeyframeExtractionConfig,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import (
    CLIPEmbeddingsGenerator,
    EmbeddingConfig,
    FrameEmbedding,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_search_index import (
    KeyframeSearchIndex,
)

from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic import (
    SemanticChunking,
)
from mmct.video_pipeline.core.ingestion.merge_summary_n_transcript.merge_visual_summay_with_transcript import (
    MergeVisualSummaryWithTranscript,
)
from mmct.video_pipeline.core.ingestion.video_compression.video_compression import VideoCompressor
from mmct.blob_store_manager import BlobStorageManager
from mmct.video_pipeline.utils.helper import get_media_folder
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
    pending_upload_tasks: Optional[List] = None
    local_resources: Optional[List[str]] = None
    video_url: Optional[str] = None
    chapter_responses: Optional[Any] = None
    chapter_transcripts: Optional[Any] = None
    is_already_ingested: Optional[bool] = None
    keyframe_metadata: Optional[List] = None
    frame_embeddings: Optional[List[FrameEmbedding]] = None
    parent_id: Optional[str] = None  # Original video ID (for both split and non-split cases)
    parent_duration: Optional[float] = None  # Original video duration in seconds
    video_duration: Optional[float] = None  # Duration of this specific video part in seconds

    def __post_init__(self):
        if self.blob_urls is None:
            self.blob_urls = {}
        if self.pending_upload_tasks is None:
            self.pending_upload_tasks = []
        if self.local_resources is None:
            self.local_resources = []

    async def cleanup_pending_uploads(self):
        """Clean up any pending upload tasks that were never awaited."""
        if self.pending_upload_tasks:
            try:
                # Close/cancel any pending coroutines
                for task in self.pending_upload_tasks:
                    if asyncio.iscoroutine(task):
                        task.close()
                self.pending_upload_tasks.clear()
            except Exception as e:
                pass  # Silently ignore cleanup errors


class IngestionPipeline:
    """
    IngestionPipeline handles the ingestion, processing, and indexing of a video to prepare it
    for use with the VideoAgent system.

    This pipeline supports transcription using Speech-to-Text ("azure-stt") or OpenAI Whisper,
    and it stores the resulting transcripts, frames, audio, metaData and optionally created index corresponsing to the video in computer vision.

    It also uploads all required video-related files (e.g., original video, transcripts, metadata)
    to an Storage account as part of the ingestion process.

    Attributes:
        video_path (str): Path to the video file to be ingested.
        index_name (str): Name of the Azure AI Search index where video data will be stored.
        language (Languages, optional): Language of the video (only Languages Enum), used for transcription.
            Required only when transcript_path is not provided. Defaults to None.
        transcription_service (str, optional): Transcription service to use ("azure-stt" or "whisper"). Defaults to "azure-stt".
            Only used when transcript_path is not provided.
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
    >>> from mmct.video_pipeline.core.ingestion.transcription.transcription_services import
    TranscriptionServices
    >>> import asyncio
    >>>
    >>> async def run_ingestion():
    >>>     ingestion = IngestionPipeline(
    >>>         video_path="<valid-video-path>",
    >>>         index_name="<ai-search-index-name>",
    >>>         language=Languages.TELUGU_INDIA,
    >>>         transcription_service=TranscriptionServices.AZURE_STT",
    >>>         url=None
    >>>     )
    >>>     await ingestion()
    >>>
    >>> asyncio.run(run_ingestion())

    """

    def __init__(
        self,
        video_path: str,
        index_name: str,
        language: Optional[Languages] = None,
        transcription_service: Optional[str] = None,
        url: Optional[str] = None,
        transcript_path: Optional[str] = None,
        disable_console_log: Annotated[
            bool, "boolean flag to disable console logs"
        ] = False,
        hash_video_id: Annotated[str, "unique Hash Video Id"] = None,
        frame_stacking_grid_size: Annotated[int, "Grid size for frame stacking (>1 enables stacking, 1 disables)"] = 4,
        keyframe_config: Annotated[Dict[str, float], "Keyframe extraction configuration"] = None,
    ):
        if disable_console_log == False:
            log_manager.enable_console()
        else:
            log_manager.disable_console()
        self.logger = log_manager.get_logger()

        # Validate that language is provided if transcript_path is not provided
        if not transcript_path and not language:
            raise ValueError("language parameter is required when transcript_path is not provided")

        self.hash_video_id = hash_video_id
        self.video_container = os.getenv("VIDEO_CONTAINER_NAME")
        self.audio_container = os.getenv("AUDIO_CONTAINER_NAME")
        self.transcript_container = os.getenv("TRANSCRIPT_CONTAINER_NAME")
        self.frames_container = os.getenv("FRAMES_CONTAINER_NAME")
        self.keyframe_container = os.getenv("KEYFRAME_CONTAINER_NAME", "keyframes")  # Default to "keyframes" if not set
        self.timestamps_container = os.getenv("TIMESTAMPS_CONTAINER_NAME")
        self.video_description_container_name = os.getenv("VIDEO_DESCRIPTION_CONTAINER_NAME")
        self.video_path = video_path
        self.transcript_path = transcript_path
        _, self.video_extension = os.path.splitext(self.video_path)
        self.transcription_service = transcription_service
        self.url = url
        self.index_name = index_name
        self.language = language
        self.frame_stacking_grid_size = frame_stacking_grid_size
        # Set default keyframe config if not provided
        if keyframe_config is None:
            self.keyframe_config = {
                "motion_threshold": 1.5,
                "sample_fps": 2
            }
        else:
            self.keyframe_config = keyframe_config
        self.blob_manager = None  # Will be initialized async when first needed
        self.original_video_path = video_path

    async def _get_blob_manager(self):
        """Initialize blob manager if not already initialized."""
        if self.blob_manager is None:
            self.blob_manager = await BlobStorageManager.create()
        return self.blob_manager

    
    async def _check_and_compress_video(self):
        """
        Check if video file size exceeds 500 MB and compress if needed.
        Note: This method still modifies self.video_path for backward compatibility.
        """
        try:
            file_size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
            self.logger.info(f"Video file size: {file_size_mb:.2f} MB")

            if file_size_mb > 500:
                self.logger.info(f"Video file size ({file_size_mb:.2f} MB) exceeds 500 MB threshold. Starting compression...")

                # Create compressed directory in media folder
                media_folder = await get_media_folder()
                compressed_dir = os.path.join(media_folder, "compressed")
                os.makedirs(compressed_dir, exist_ok=True)

                # Initialize video compressor
                compressor = VideoCompressor(
                    input_path=self.video_path,
                    target_size_mb=500,
                    output_dir=compressed_dir
                )

                # Compress the video
                compressor.compress()

                # Update video path to compressed version
                compressed_path = compressor.output_path
                if os.path.exists(compressed_path):
                    self.video_path = compressed_path
                    # Note: compressed path will be tracked in context later
                    compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                    self.logger.info(f"Video compressed successfully. New size: {compressed_size_mb:.2f} MB")
                    self.logger.info(f"Using compressed video: {compressed_path}")
                    return True
                else:
                    self.logger.warning("Compression failed, compressed file not found")
                    return False
            else:
                self.logger.info("Video file size is within acceptable limits, no compression needed")
                return True
        except Exception as e:
            self.logger.warning(f"Exception occurred during video compression check: {e}")
            return False

    async def _extract_keyframes(self):
        """
        Extract keyframes from video using motion detection early in pipeline.
        """
        try:
            self.logger.info("Starting keyframe extraction...")

            # Generate hash ID for consistent naming
            video_hash_id = await get_file_hash(self.video_path)

            # Configure keyframe extraction
            keyframe_config = KeyframeExtractionConfig(
                motion_threshold=self.keyframe_config["motion_threshold"],
                sample_fps=self.keyframe_config["sample_fps"]
            )

            # Initialize keyframe extractor
            keyframe_extractor = KeyframeExtractor(keyframe_config)

            # Extract keyframes
            keyframe_metadata = await keyframe_extractor.extract_keyframes(
                video_path=self.video_path,
                video_id=video_hash_id
            )

            self.logger.info(f"Successfully extracted {len(keyframe_metadata)} keyframes")
            for frame in keyframe_metadata:
                self.logger.debug(f"  Frame {frame.frame_number}: {frame.timestamp_seconds:.2f}s (motion: {frame.motion_score:.3f})")

            return keyframe_metadata

        except Exception as e:
            self.logger.exception(f"Exception occurred during keyframe extraction: {e}")
            raise

    async def _generate_embeddings_for_keyframes(self, context: ProcessingContext) -> ProcessingContext:
        """
        Generate CLIP embeddings for extracted keyframes.
        """
        try:
            if not context.keyframe_metadata:
                self.logger.warning("No keyframe metadata available for embedding generation")
                return context

            self.logger.info(f"Starting embedding generation for {len(context.keyframe_metadata)} keyframes...")

            # Configure embedding generation
            embedding_config = EmbeddingConfig(
                clip_model_name="openai/clip-vit-base-patch32",
                batch_size=8
            )

            # Initialize embeddings generator
            embeddings_generator = CLIPEmbeddingsGenerator(embedding_config)

            try:
                # Generate embeddings
                context.frame_embeddings = await embeddings_generator.process_frames(
                    context.keyframe_metadata,
                    context.hash_id
                )

                self.logger.info(f"Successfully generated {len(context.frame_embeddings)} frame embeddings")

            finally:
                # Clean up embeddings generator resources
                embeddings_generator.cleanup()

            return context

        except Exception as e:
            self.logger.exception(f"Exception occurred during embedding generation: {e}")
            raise

    async def _store_frame_embeddings_to_search_index(self, context: ProcessingContext) -> ProcessingContext:
        """
        Store frame embeddings to Azure AI Search index.
        """
        try:
            if not context.frame_embeddings:
                self.logger.warning("No frame embeddings available for storage")
                return context

            self.logger.info(f"Storing {len(context.frame_embeddings)} frame embeddings to search index...")

            # Get Azure Search endpoint
            search_endpoint = os.getenv("SEARCH_ENDPOINT")
            if not search_endpoint:
                self.logger.error("SEARCH_ENDPOINT environment variable not set")
                return context

            # Create keyframe search index
            keyframe_index_name = f"keyframes-{self.index_name}"
            keyframe_search_index = KeyframeSearchIndex(
                search_endpoint=search_endpoint,
                index_name=keyframe_index_name
            )

            try:
                # Upload frame embeddings to search index
                success = await keyframe_search_index.upload_frame_embeddings(
                    frame_embeddings=context.frame_embeddings,
                    video_id=context.hash_id,
                    video_path=context.video_path,
                    parent_id=context.parent_id,
                    parent_duration=context.parent_duration,
                    video_duration=context.video_duration
                )

                if success:
                    self.logger.info("Successfully stored frame embeddings to search index")
                else:
                    self.logger.error("Failed to store frame embeddings to search index")

            finally:
                await keyframe_search_index.close()

            return context

        except Exception as e:
            self.logger.exception(f"Exception occurred during frame embeddings storage: {e}")
            raise


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
                hash_id=video_hash_id,
                index_name=self.index_name
            )

            if is_already_ingested:
                
                self.logger.info(f"Video with hash_id {video_hash_id} already exists in index {self.index_name}. Skipping pipeline - no processing needed.")
                return False

            self.logger.info("Video not found in index. Proceeding with full ingestion pipeline...")
            return True

        except Exception as e:
            self.logger.exception(f"Exception occurred during early ingestion check: {e}")
            raise

    async def _perform_keyframe_extraction(self, video_paths: list, hash_suffixes: list):
        """
        Orchestrate keyframe extraction based on video configuration.
        Handles both single video and multi-part video scenarios.
        Returns keyframe metadata (dict for split videos, list for single video, None otherwise).
        """
        try:
            if len(video_paths) > 1:
                # Video was split - extract keyframes from each part
                self.logger.info(f"Video was split into {len(video_paths)} parts. Extracting keyframes from each part...")
                keyframe_metadata_map = await self._extract_keyframes_from_video_parts(video_paths, hash_suffixes)
                return keyframe_metadata_map
            else:
                # Single video - extract keyframes from original
                self.logger.info("Single video detected. Extracting keyframes from original video...")
                keyframe_metadata = await self._extract_keyframes()
                return keyframe_metadata

        except Exception as e:
            self.logger.exception(f"Exception occurred during keyframe extraction orchestration: {e}")
            raise

    async def _extract_keyframes_from_video_parts(self, video_paths: list, hash_suffixes: list):
        """
        Extract keyframes from multiple video parts with consistent naming.
        Returns a dictionary mapping hash_id to keyframe metadata list.
        """
        try:
            self.logger.info(f"Starting keyframe extraction for {len(video_paths)} video parts...")

            # Generate base hash ID from Part A video for consistent hash IDs
            part_a_path = video_paths[0]  # Part A is always first
            base_hash_id = await get_file_hash(part_a_path)

            # Store metadata for each part
            keyframe_metadata_map = {}

            for video_path, hash_suffix in zip(video_paths, hash_suffixes):
                part_name = "Part A" if hash_suffix == "" else f"Part {hash_suffix}"
                part_hash_id = base_hash_id + hash_suffix

                self.logger.info(f"Extracting keyframes for {part_name}: {os.path.basename(video_path)}")
                self.logger.info(f"  Hash ID: {part_hash_id}")

                # Configure keyframe extraction
                keyframe_config = KeyframeExtractionConfig(
                    motion_threshold=self.keyframe_config["motion_threshold"],
                    sample_fps=self.keyframe_config["sample_fps"]
                )

                # Initialize keyframe extractor
                keyframe_extractor = KeyframeExtractor(keyframe_config)

                # Extract keyframes for this part
                keyframe_metadata = await keyframe_extractor.extract_keyframes(
                    video_path=video_path,
                    video_id=part_hash_id
                )

                # Store metadata for this part
                keyframe_metadata_map[part_hash_id] = keyframe_metadata

                self.logger.info(f"Successfully extracted {len(keyframe_metadata)} keyframes for {part_name}")
                for frame in keyframe_metadata:
                    self.logger.debug(f"  Frame {frame.frame_number}: {frame.timestamp_seconds:.2f}s (motion: {frame.motion_score:.3f})")

            return keyframe_metadata_map

        except Exception as e:
            self.logger.exception(f"Exception occurred during keyframe extraction from video parts: {e}")
            raise

    async def _add_keyframes_to_upload_tasks(self, context: ProcessingContext, blob_manager):
        """
        Add keyframes to pending upload tasks instead of all frames.
        """
        try:
            # Get media folder and keyframes directory
            media_folder = await get_media_folder()
            keyframes_dir = os.path.join(media_folder, "keyframes", context.hash_id)

            if not os.path.exists(keyframes_dir):
                self.logger.warning(f"Keyframes directory not found: {keyframes_dir}")
                return

            # Get all keyframe files from the directory
            keyframe_files = []
            for filename in os.listdir(keyframes_dir):
                if filename.endswith('.jpg'):
                    keyframe_path = os.path.join(keyframes_dir, filename)
                    keyframe_files.append((filename, keyframe_path))

            if not keyframe_files:
                self.logger.warning(f"No keyframes found in directory: {keyframes_dir}")
                return

            self.logger.info(f"Found {len(keyframe_files)} keyframes to upload")

            # Add upload tasks for each keyframe
            for filename, keyframe_path in keyframe_files:
                context.pending_upload_tasks.append(
                    blob_manager.upload_file(
                        container=self.keyframe_container,
                        blob_name=f"{context.hash_id}/{filename}",
                        file_path=keyframe_path,
                    )
                )

            # Add keyframes directory to local resources for cleanup
            context.local_resources.append(keyframes_dir)

            # Update blob URLs to point to keyframes instead of frames
            context.blob_urls["keyframes_blob_folder_url"] = blob_manager.get_blob_url(
                container=self.keyframe_container, blob_name=f"keyframes/{context.hash_id}"
            )

        except Exception as e:
            self.logger.exception(f"Exception occurred during keyframe upload: {e}")
            raise
    
    async def _process_video_part_parallel(self, video_path: str, part_hash_id: str, transcript_path: Optional[str] = None, parent_id: Optional[str] = None, parent_duration: Optional[float] = None, keyframe_metadata: Optional[List] = None) -> None:
        """
        Process a single video part in parallel - used for split videos.

        Args:
            video_path: Path to the video part file
            part_hash_id: Hash ID for this specific video part
            transcript_path: Optional path to the transcript file for this part
            parent_id: Hash ID of the original video (before splitting)
            parent_duration: Duration of the original video in seconds
        """
        try:
            self.logger.info(f"Starting processing of video part: {os.path.basename(video_path)}")
            self.logger.info(f"Part Hash ID: {part_hash_id}")
            if transcript_path:
                self.logger.info(f"Using transcript: {os.path.basename(transcript_path)}")

            # Check if this video part already exists in the index
            is_already_ingested = await check_video_already_ingested(
                hash_id=part_hash_id,
                index_name=self.index_name
            )

            if is_already_ingested:
                self.logger.info(f"Video part with hash_id {part_hash_id} already exists in index {self.index_name}. Skipping.")
                return

            # Create processing context for this video part
            _, video_extension = os.path.splitext(video_path)
            # Calculate duration of this video part
            part_duration = await get_video_duration(video_path)
            context = ProcessingContext(
                hash_id=part_hash_id,
                video_path=video_path,
                video_extension=video_extension,
                transcript_path=transcript_path,
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=part_duration
            )
            
            # Get blob manager
            blob_manager = await self._get_blob_manager()

            # Set keyframes blob URL
            context.blob_urls["keyframes_blob_folder_url"] = blob_manager.get_blob_url(
                container=self.keyframe_container, blob_name=f"keyframes/{context.hash_id}"
            )

            # Use the keyframe metadata that was extracted earlier
            if keyframe_metadata:
                context.keyframe_metadata = keyframe_metadata
                self.logger.info(f"Using {len(keyframe_metadata)} keyframes for part {part_hash_id}")

                # Generate embeddings for these keyframes
                context = await self._generate_embeddings_for_keyframes(context)
                self.logger.info(f"Generated embeddings for {len(context.frame_embeddings)} keyframes for part {part_hash_id}")

                # Store embeddings to AI Search index
                context = await self._store_frame_embeddings_to_search_index(context)
                self.logger.info(f"Stored frame embeddings to AI Search for part {part_hash_id}")

                # Upload keyframes to blob storage
                await self._add_keyframes_to_upload_tasks(context, blob_manager)
                self.logger.info(f"Added keyframes to upload tasks for part {part_hash_id}")
            else:
                self.logger.warning(f"No keyframe metadata provided for part {part_hash_id}")

            # Run functional pipeline methods
            context = await self.get_transcription(context, blob_manager)
            self.logger.info(f"Transcript generated for part {part_hash_id}")

            # Upload video part to blob
            context.video_url = await file_upload_to_blob(
                file_path=context.video_path,
                blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                container_name=self.video_container,
            )
            self.logger.info(f"Uploaded video part to blob: {part_hash_id}")

            # Run semantic chunking and chapter generation
            context = await self._semantic_chunking_chapter_generation(
                context, context.video_url, self.url
            )
    
            if not context.is_already_ingested:
                self.logger.info(f"Chapter generated for part {part_hash_id}")

            # Upload files in batches
            for batch in chunked(context.pending_upload_tasks, 20):
                upload_results = await asyncio.gather(*batch)
                del upload_results
                gc.collect()

            self.logger.info(f"Files uploaded for part {part_hash_id}")

            await blob_manager.close()

            # Clean up local files for this part
            await remove_file(context.hash_id)
            
            # Clean up local resources for this part
            for resource_path in context.local_resources:
                try:
                    if os.path.exists(resource_path):
                        if os.path.isfile(resource_path):
                            os.remove(resource_path)
                            self.logger.info(f"Removed local file: {resource_path}")
                        elif os.path.isdir(resource_path):
                            shutil.rmtree(resource_path)
                            self.logger.info(f"Removed local directory: {resource_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove local resource {resource_path}: {e}")
            
            # Clean up context variables
            del context.video_url
            gc.collect()
            
            self.logger.info(f"Successfully processed video part: {part_hash_id}")
            
        except Exception as e:
            self.logger.exception(f"Exception occurred while processing video part {part_hash_id}: {e}")
            raise

    async def get_transcription(self,context: ProcessingContext, blob_manager) -> ProcessingContext:
        """Generate transcription for video - functional version."""
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
            
            # Handle transcript_path case - no transcription needed, just use provided transcript
            transcript_path_to_use = context.transcript_path or self.transcript_path
            if transcript_path_to_use:
                self.logger.info(f"Using provided transcript path: {transcript_path_to_use}")

                # Copy provided transcript to target location
                target_transcript_path = os.path.join(
                    await get_media_folder(), f"transcript_{context.hash_id}.srt"
                )
                shutil.copy2(transcript_path_to_use, target_transcript_path)

                # Load transcript content
                context.transcript = await load_srt(target_transcript_path)
                self.logger.info("Successfully loaded provided transcript")

                # Track transcript file for cleanup
                local_paths = [target_transcript_path]

                # Set audio_blob_url to None or empty string since no audio was generated
                context.blob_urls["audio_blob_url"] = "None"

            else:
                # Normal transcription flow
                if self.transcription_service == TranscriptionServices.AZURE_STT:
                    transcriber = CloudTranscription(
                        video_path=context.video_path,
                        hash_id=context.hash_id,
                        language=self.language,
                    )
                elif self.transcription_service is None:
                    transcriber = CloudTranscription(
                        video_path=context.video_path,
                        hash_id=context.hash_id,
                        language=self.language,
                    )
                else:
                    transcriber = WhisperTranscription(
                        video_path=context.video_path,
                        hash_id=context.hash_id
                    )

                self.logger.info("Initialized the transcriber instance")
                context.transcript, local_paths = await transcriber()
                self.logger.info("Successfully generated the transcript for the video.")

            # Only upload audio if transcription was performed (not when transcript_path is provided)
            if not transcript_path_to_use:
                audio_extension = (
                ".wav"
                if self.transcription_service in [TranscriptionServices.AZURE_STT, None]
                else ".mp3")

                context.blob_urls["audio_blob_url"] = blob_manager.get_blob_url(
                    container=self.audio_container,
                    blob_name=f"{context.hash_id}" + audio_extension,
                )

                context.pending_upload_tasks.append(
                    blob_manager.upload_file(
                        container=self.audio_container,
                        blob_name=f"{context.hash_id}" + audio_extension,
                        file_path=os.path.join(
                            await get_media_folder(),
                            (f"{context.hash_id}" + audio_extension),
                        ),
                    )
                )
                self.logger.info("Added Audio File to pending upload tasks list")

            context.pending_upload_tasks.append(
                blob_manager.upload_file(
                    container=self.transcript_container,
                    blob_name=f"transcript_{context.hash_id}.srt",
                    file_path=os.path.join(
                        await get_media_folder(),
                        f"transcript_{context.hash_id}.srt",
                    ),
                )
            )
            self.logger.info("Added Transcript File to pending upload tasks list")

            context.blob_urls["transcript_blob_url"] = blob_manager.get_blob_url(
                container=self.transcript_container,
                blob_name=f"transcript_{context.hash_id}.srt",
            )
            self.logger.info(
                "Logged the transcript blob url to the blob urls mapping dictionary"
            )

            context.local_resources.extend(local_paths)
            del local_paths
            gc.collect()
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while performing transcription: {e}"
            )
            raise


    async def _merge_visual_summary_with_transcript(self, context: ProcessingContext, blob_manager) -> ProcessingContext:
        """
        This method merge summaries from chapters and transcript - functional version.
        """
        try:
            self.logger.info(
                "Creating an instance of class MergeVisualSummaryWithTranscript"
            )
            merge_summary_transcript = MergeVisualSummaryWithTranscript(
                chapter_responses=context.chapter_responses,
                video_id=context.hash_id,
                full_transcript_string=context.transcript,
                transcripts=context.chapter_transcripts,
            )
            self.logger.info("Merging the visual summary and transcript")
            await merge_summary_transcript()
            self.logger.info("Successfully merged the visual summary and transcript")
            context.pending_upload_tasks.append(
                blob_manager.upload_file(
                    container=self.video_description_container_name,
                    blob_name=f"{context.hash_id}.json",
                    file_path=os.path.join(
                        await get_media_folder(), f"{context.hash_id}.json"
                    ),
                )
            )
            self.logger.info(
                "Logged the transcript and summary file url to the blob urls mapping dictionary"
            )
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while merging visual summary with transcript: {e}"
            )
            raise

    async def _semantic_chunking_chapter_generation(self, context: ProcessingContext, video_url: str, url: Optional[str] = None) -> ProcessingContext:
        """
        This method initializes the semantic chunker and runs chapter generation - functional version.
        """
        try:
            self.logger.info(
                "Creating an instance of SemanticChunking class to perform operations related to semantic chunking"
            )
            semantic_chunker = SemanticChunking(
                hash_id=context.hash_id,
                index_name=self.index_name,
                transcript=context.transcript,
                blob_urls=context.blob_urls,
                frame_stacking_grid_size=self.frame_stacking_grid_size,
                parent_id=context.parent_id,
                parent_duration=context.parent_duration,
                video_duration=context.video_duration,
            )
            self.logger.info(
                "Successfully created an instance of SemanticChunking class!"
            )
            
            context.chapter_responses, context.chapter_transcripts, context.is_already_ingested = (
                await semantic_chunker.run(
                    video_blob_url=video_url, url=url
                )
            )
            
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while creating an instance of SemanticChunking class: {e}"
            )
            raise


    async def _check_video_has_audio(self, video_path: str) -> bool:
        """
        Check if video file has an audio stream.
        Returns True if audio exists, False otherwise.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            return stdout and stdout.strip() == b"audio"
        except Exception as e:
            self.logger.warning(f"Could not check for audio stream: {e}")
            return False

    async def __call__(self):
        """Main ingestion pipeline method - now supports video splitting and parallel processing."""
        try:
            # Early ingestion check - exit immediately if already processed
            should_continue = await self._perform_early_ingestion_check()
            if not should_continue:
                return

            # Check if video has audio stream BEFORE processing (only if transcript_path not provided)
            if not self.transcript_path:
                self.logger.info("Checking if video has audio stream...")
                has_audio = await self._check_video_has_audio(self.video_path)
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

            compression_flag = await self._check_and_compress_video()  # Check file size and compress if needed
            if not compression_flag:
                self.logger.warning("Video Compression could not be performed. Proceeding with further steps.")
            else:
                self.logger.info("Video compression check completed!")

            # Calculate parent video metadata (original video before any splitting)
            parent_video_id = await get_file_hash(file_path=self.original_video_path)
            parent_video_duration = await get_video_duration(self.video_path)
            self.logger.info(f"Parent video ID: {parent_video_id}, Duration: {parent_video_duration:.2f}s")

            # Split video if needed based on duration (>= 30 minutes)
            video_paths, hash_suffixes = await split_video_if_needed(self.video_path)

            # Extract keyframes after video splitting check
            keyframe_metadata_result = await self._perform_keyframe_extraction(video_paths, hash_suffixes)
            self.logger.info("Keyframes extraction completed!")
            self.logger.info(f"Processing {len(video_paths)} video part(s)")

            # Track split video files for cleanup
            split_video_cleanup_paths = []
            if len(video_paths) > 1:
                split_video_cleanup_paths.extend(video_paths)

            # If transcript_path is provided and video was split, split the transcript
            transcript_paths = []
            if self.transcript_path and len(video_paths) > 1:
                self.logger.info("Video was split and transcript_path provided. Splitting transcript into two parts...")
                transcript_content = await load_srt(self.transcript_path)
                transcript_chunks = split_transcript_by_segments(transcript_content, len(video_paths))

                # Save transcript chunks to temporary files
                media_folder = await get_media_folder()
                base_hash_id = await get_file_hash(file_path=video_paths[0])

                for transcript_chunk, hash_suffix in zip(transcript_chunks, hash_suffixes):
                    part_hash_id = base_hash_id + hash_suffix
                    transcript_chunk_path = os.path.join(media_folder, f"transcript_{part_hash_id}.srt")
                    async with aiofiles.open(transcript_chunk_path, 'w', encoding='utf-8') as f:
                        await f.write(transcript_chunk)
                    transcript_paths.append(transcript_chunk_path)
                    self.logger.info(f"Created transcript chunk: {transcript_chunk_path}")
            elif self.transcript_path:
                # Single video with transcript
                transcript_paths = [self.transcript_path]

            # Process video parts in parallel if split, otherwise process single video
            if len(video_paths) > 1:
                self.logger.info("Processing video parts in parallel for faster execution...")

                # Generate base hash ID from Part A video for consistent hash IDs
                part_a_path = video_paths[0]  # Part A is always first
                base_hash_id = await get_file_hash(file_path=part_a_path)
                self.logger.info(f"Generated base hash ID from Part A: {base_hash_id}")

                # Create tasks for parallel processing with consistent hash IDs
                tasks = []
                for idx, (video_path, hash_suffix) in enumerate(zip(video_paths, hash_suffixes)):
                    part_name = "Part A" if hash_suffix == "" else f"Part {hash_suffix}"

                    # Use base hash ID + suffix for consistent naming
                    part_hash_id = base_hash_id + hash_suffix

                    # Get corresponding transcript path if available
                    part_transcript_path = transcript_paths[idx] if transcript_paths else None

                    # Get keyframe metadata for this part
                    part_keyframe_metadata = keyframe_metadata_result.get(part_hash_id) if isinstance(keyframe_metadata_result, dict) else None

                    self.logger.info(f"Creating task for {part_name}: {os.path.basename(video_path)}")
                    self.logger.info(f"  Hash ID: {part_hash_id}")
                    if part_transcript_path:
                        self.logger.info(f"  Transcript: {os.path.basename(part_transcript_path)}")
                    if part_keyframe_metadata:
                        self.logger.info(f"  Keyframes: {len(part_keyframe_metadata)}")

                    # Create asyncio task for processing this video part
                    task = asyncio.create_task(
                        self._process_video_part_parallel(video_path, part_hash_id, part_transcript_path, parent_video_id, parent_video_duration, part_keyframe_metadata)
                    )
                    tasks.append(task)

                # Execute all video parts in parallel
                self.logger.info(f"Starting parallel processing of {len(tasks)} video parts...")
                await asyncio.gather(*tasks)

                # Clean up transcript chunk files
                for transcript_path in transcript_paths:
                    try:
                        if transcript_path != self.transcript_path and os.path.exists(transcript_path):
                            os.remove(transcript_path)
                            self.logger.info(f"Removed transcript chunk: {transcript_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove transcript chunk {transcript_path}: {e}")

                self.logger.info("All video parts processed successfully!")
                
            else:
                # Single video processing (no split) - original logic
                self.logger.info("Processing single video (no split required)")

                # Generate hash ID
                hash_id = await get_file_hash(file_path=self.video_path)
                self.logger.info(f"Generated hash ID: {hash_id}")

                # Get transcript path if available
                single_transcript_path = transcript_paths[0] if transcript_paths else None
                if single_transcript_path:
                    self.logger.info(f"Using transcript: {os.path.basename(single_transcript_path)}")

                # Create processing context
                context = ProcessingContext(
                    hash_id=hash_id,
                    video_path=self.video_path,
                    video_extension=self.video_extension,
                    transcript_path=single_transcript_path,
                    parent_id=parent_video_id,  # Same as hash_id for non-split videos
                    parent_duration=parent_video_duration,
                    video_duration=parent_video_duration  # Same as parent for non-split videos
                )

                # Extract keyframes and generate embeddings early in pipeline
                context.keyframe_metadata = await self._extract_keyframes()
                self.logger.info("Keyframes extracted!")

                context = await self._generate_embeddings_for_keyframes(context)
                self.logger.info("Embeddings generated for keyframes!")

                # Store frame embeddings to search index
                context = await self._store_frame_embeddings_to_search_index(context)
                self.logger.info("Frame embeddings stored to search index!")

                # Get blob manager
                blob_manager = await self._get_blob_manager()

                # Upload keyframes to blob storage
                await self._add_keyframes_to_upload_tasks(context, blob_manager)
                self.logger.info("Added keyframes to upload tasks!")

                # Set keyframes blob URL
                context.blob_urls["keyframes_blob_folder_url"] = blob_manager.get_blob_url(
                    container=self.keyframe_container, blob_name=f"keyframes/{context.hash_id}"
                )

                # Run functional pipeline methods
                context = await self.get_transcription(context, blob_manager)
                self.logger.info("Transcript Generated!")


                # Upload video to blob
                context.video_url = await file_upload_to_blob(
                    file_path=context.video_path,
                    blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                    container_name=self.video_container,
                )
                self.logger.info("Uploaded local video to blob")

                # Run semantic chunking and chapter generation
                context = await self._semantic_chunking_chapter_generation(
                    context, context.video_url, self.url
                )

                if not context.is_already_ingested:
                    self.logger.info(
                    "Chapter generated for the visual summary and successfully ingested to index!"
                    )

                # Upload files in batches
                for batch in chunked(context.pending_upload_tasks, 5):
                    upload_results = await asyncio.gather(*batch)
                    # Free completed upload tasks to prevent memory buildup
                    del upload_results
                    gc.collect()
                self.logger.info(
                    "Successfully uploaded the files to blob present in the pending upload tasks list!"
                )

                await blob_manager.close()
                
                # Clean up local files after successful ingestion
                await remove_file(context.hash_id)
                
                # Clean up local resources (including copied video file)
                for resource_path in context.local_resources:
                    try:
                        if os.path.exists(resource_path):
                            if os.path.isfile(resource_path):
                                os.remove(resource_path)
                                self.logger.info(f"Removed local file: {resource_path}")
                            elif os.path.isdir(resource_path):
                                shutil.rmtree(resource_path)
                                self.logger.info(f"Removed local directory: {resource_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove local resource {resource_path}: {e}")
                
                # Clean up context variables
                del context.video_url
                gc.collect()
            
            # Clean up split video files if any were created
            for split_video_path in split_video_cleanup_paths:
                try:
                    if os.path.exists(split_video_path):
                        os.remove(split_video_path)
                        self.logger.info(f"Removed split video file: {split_video_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove split video file {split_video_path}: {e}")
            
            self.logger.info("Local files cleaned up successfully!")
            self.logger.info("Ingestion pipeline ran successfully!")
            
        except Exception as e:
            self.logger.exception(
                f"Exception occurred while running Ingestion pipeline: {e}"
            )
            raise
    
    @classmethod
    async def process_videos_parallel(
        cls,
        video_paths: List[str],
        index_name: str,
        language: Optional[Languages] = None,
        transcription_service: Optional[str] = TranscriptionServices.AZURE_STT.value,
        urls: Optional[List[str]] = None,
        transcript_paths: Optional[List[str]] = None,
        disable_console_log: bool = False,
        frame_stacking_grid_size: int = 4,
        max_concurrent: int = 3
    ) -> List[str]:
        """
        Process multiple videos in parallel using separate pipeline instances.

        Args:
            video_paths: List of video file paths to process
            index_name: Azure AI Search index name
            language: Language for transcription (optional if transcript_paths is provided)
            transcription_service: Transcription service to use (only used if transcript_paths not provided)
            urls: Optional list of URLs (must match video_paths length)
            transcript_paths: Optional list of transcript file paths (must match video_paths length)
            disable_console_log: Whether to disable console logging
            frame_stacking_grid_size: Grid size for frame stacking
            max_concurrent: Maximum number of concurrent video processing tasks

        Returns:
            List of hash IDs for successfully processed videos
        """
        if urls and len(urls) != len(video_paths):
            raise ValueError("urls length must match video_paths length")

        if transcript_paths and len(transcript_paths) != len(video_paths):
            raise ValueError("transcript_paths length must match video_paths length")

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_video(video_path: str, url: Optional[str] = None, transcript_path: Optional[str] = None) -> Optional[str]:
            """Process a single video with semaphore control."""
            async with semaphore:
                try:
                    # Create separate instance for each video
                    pipeline = cls(
                        video_path=video_path,
                        index_name=index_name,
                        language=language,
                        transcription_service=transcription_service,
                        url=url,
                        transcript_path=transcript_path,
                        disable_console_log=disable_console_log,
                        frame_stacking_grid_size=frame_stacking_grid_size
                    )

                    # Process the video
                    await pipeline()

                    # Return hash ID for tracking
                    hash_id = await get_file_hash(file_path=video_path)
                    return hash_id

                except Exception as e:
                    pipeline.logger.error(f"Failed to process video {video_path}: {e}")
                    return None

        # Create tasks for all videos
        tasks = []
        for i, video_path in enumerate(video_paths):
            url = urls[i] if urls else None
            transcript_path = transcript_paths[i] if transcript_paths else None
            task = process_single_video(video_path, url, transcript_path)
            tasks.append(task)

        # Execute all tasks in parallel (with semaphore limiting concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_hash_ids = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing {video_paths[i]}: {result}")
            elif result is not None:
                successful_hash_ids.append(result)

        return successful_hash_ids
    
    async def process_with_context(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process a video using an existing context - useful for advanced parallel scenarios.
        
        Args:
            context: Pre-initialized ProcessingContext
            
        Returns:
            Updated ProcessingContext with processing results
        """
        try:
            # Get blob manager
            blob_manager = await self._get_blob_manager()

            # Set keyframes blob URL
            context.blob_urls["keyframes_blob_folder_url"] = blob_manager.get_blob_url(
                container=self.keyframe_container, blob_name=f"keyframes/{context.hash_id}"
            )

            # Run functional pipeline methods
            context = await self.get_transcription(context, blob_manager)
            self.logger.info("Transcript Generated!")

            # Upload video to blob
            context.video_url = await file_upload_to_blob(
                file_path=context.video_path,
                blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                container_name=self.video_container,
            )
            self.logger.info("Uploaded local video to blob")

            # Run semantic chunking and chapter generation
            context = await self._semantic_chunking_chapter_generation(
                context, context.video_url, self.url
            )
    
            if not context.is_already_ingested:
                self.logger.info(
                "Chapter generated for the visual summary and successfully ingested to index!"
                )

            # Upload files in batches
            for batch in chunked(context.pending_upload_tasks, 5):
                upload_results = await asyncio.gather(*batch)
                del upload_results
                gc.collect()
            
            await blob_manager.close()
            
            # Clean up local files
            await remove_file(context.hash_id)
            
            # Clean up local resources
            for resource_path in context.local_resources:
                try:
                    if os.path.exists(resource_path):
                        if os.path.isfile(resource_path):
                            os.remove(resource_path)
                        elif os.path.isdir(resource_path):
                            shutil.rmtree(resource_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove local resource {resource_path}: {e}")
            
            return context
            
        except Exception as e:
            self.logger.exception(f"Exception occurred while processing with context: {e}")
            raise


if __name__ == "__main__":
    # Example usage - replace with your actual values
    video_path = "video-path"
    index = "index-name"
    url = "video-url"
    source_language = Languages.ENGLISH_UNITED_STATES
    transcript_path = "transcript.srt"  # Optional: path to existing transcript file
    keyframe_config = {
        "motion_threshold": 1.5,
        "sample_fps": 2
    }
    ingestion = IngestionPipeline(
        video_path=video_path,
        index_name=index,
        url=url,
        transcription_service=TranscriptionServices.AZURE_STT,
        language=source_language,
        transcript_path=transcript_path,
        keyframe_config=keyframe_config
    )
    asyncio.run(ingestion())
