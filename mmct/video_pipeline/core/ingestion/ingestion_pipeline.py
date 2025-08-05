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
)

from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.core.ingestion.transcription.transcription_services import (
    TranscriptionServices,
)
from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic import (
    SemanticChunking,
)
from mmct.video_pipeline.core.ingestion.computer_vision.computer_vision_services import (
    ComputerVisionService,
)
from mmct.video_pipeline.core.ingestion.merge_summary_n_transcript.merge_visual_summay_with_transcript import (
    MergeVisualSummaryWithTranscript,
)
from mmct.video_pipeline.core.ingestion.video_compression.video_compression import VideoCompressor
from mmct.blob_store_manager import BlobStorageManager
from mmct.video_pipeline.utils.helper import get_media_folder
from dotenv import load_dotenv, find_dotenv
from mmct.custom_logger import log_manager
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
    
    def __post_init__(self):
        if self.blob_urls is None:
            self.blob_urls = {}
        if self.pending_upload_tasks is None:
            self.pending_upload_tasks = []
        if self.local_resources is None:
            self.local_resources = []


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
        language (Languages): Language of the video (only Languages Enum), used for transcription and search indexing.
        transcription_service (str): Transcription service to use ("azure-stt" or "whisper"). Defaults to "azure-stt".
        youtube_url (str, optional): Optional YouTube URL associated with the video.
        use_computer_vision (bool): Whether to use Computer Vision for content analysis. Defaults to True.
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
    >>>         youtube_url=None,
    >>>         use_computer_vision_tool=True
    >>>     )
    >>>     await ingestion()
    >>>
    >>> asyncio.run(run_ingestion())

    """

    def __init__(
        self,
        video_path: str,
        index_name: str,
        language: Languages,
        transcription_service: Optional[str] = TranscriptionServices.AZURE_STT.value,
        youtube_url: Optional[str] = None,
        use_computer_vision_tool: Optional[bool] = False,
        disable_console_log: Annotated[
            bool, "boolean flag to disable console logs"
        ] = False,
        hash_video_id: Annotated[str, "unique Hash Video Id"] = None,
        frame_stacking_grid_size: Annotated[int, "Grid size for frame stacking (>1 enables stacking, 1 disables)"] = 4,
    ):
        if disable_console_log == False:
            log_manager.enable_console()
        else:
            log_manager.disable_console()
        self.logger = log_manager.get_logger()
        self.hash_video_id = hash_video_id
        self.video_container = os.getenv("VIDEO_CONTAINER_NAME")
        self.audio_container = os.getenv("AUDIO_CONTAINER_NAME")
        self.transcript_container = os.getenv("TRANSCRIPT_CONTAINER_NAME")
        self.frames_container = os.getenv("FRAMES_CONTAINER_NAME")
        self.timestamps_container = os.getenv("TIMESTAMPS_CONTAINER_NAME")
        self.video_description_container_name = os.getenv("VIDEO_DESCRIPTION_CONTAINER_NAME")
        self.video_path = video_path
        _, self.video_extension = os.path.splitext(self.video_path)
        self.transcription_service = transcription_service
        self.youtube_url = youtube_url
        self.index_name = index_name
        self.use_computer_vision_tool = use_computer_vision_tool
        self.language = language
        self.frame_stacking_grid_size = frame_stacking_grid_size
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
                else:
                    self.logger.error("Compression failed, compressed file not found")
                    raise RuntimeError("Video compression failed")
            else:
                self.logger.info("Video file size is within acceptable limits, no compression needed")
                
        except Exception as e:
            self.logger.exception(f"Exception occurred during video compression check: {e}")
            raise
    
    async def _process_video_part_parallel(self, video_path: str, part_hash_id: str) -> None:
        """
        Process a single video part in parallel - used for split videos.
        
        Args:
            video_path: Path to the video part file
            part_hash_id: Hash ID for this specific video part
        """
        try:
            self.logger.info(f"Starting processing of video part: {os.path.basename(video_path)}")
            self.logger.info(f"Part Hash ID: {part_hash_id}")
            
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
            context = ProcessingContext(
                hash_id=part_hash_id,
                video_path=video_path,
                video_extension=video_extension
            )
            
            # Get blob manager
            blob_manager = await self._get_blob_manager()
            
            # Run functional pipeline methods
            context = await self.get_transcription(context, blob_manager)
            self.logger.info(f"Transcript generated for part {part_hash_id}")
            
            context = await self._get_frames_timestamps(context, blob_manager)
            self.logger.info(f"Frames and timestamps generated for part {part_hash_id}")
            
            # Upload video part to blob
            context.video_url = await file_upload_to_blob(
                file_path=context.video_path,
                blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                container_name=self.video_container,
            )
            self.logger.info(f"Uploaded video part to blob: {part_hash_id}")

            # Run semantic chunking and chapter generation
            context = await self._semantic_chunking_chapter_generation(
                context, context.video_url, self.youtube_url
            )
    
            if not context.is_already_ingested:
                self.logger.info(f"Chapter generated for part {part_hash_id}")
                
                if self.use_computer_vision_tool:
                    context = await self._create_ingest_azurecv_index(context)
                    self.logger.info(f"Computer Vision index created for part {part_hash_id}")

                context = await self._merge_visual_summary_with_transcript(context, blob_manager)
                self.logger.info(f"Visual summary merged for part {part_hash_id}")

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
            del context.frames, context.timestamps, context.base64_frames, context.video_url
            gc.collect()
            
            self.logger.info(f"Successfully processed video part: {part_hash_id}")
            
        except Exception as e:
            self.logger.exception(f"Exception occurred while processing video part {part_hash_id}: {e}")
            raise

    async def get_transcription(self, context: ProcessingContext, blob_manager) -> ProcessingContext:
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
                
            transcriber = (
                CloudTranscription(
                    video_path=context.video_path,
                    hash_id=context.hash_id,
                    language=self.language,
                )
                if self.transcription_service == TranscriptionServices.AZURE_STT
                else WhisperTranscription(
                    video_path=context.video_path, hash_id=context.hash_id
                )
            )
            self.logger.info("Initialized the transcriber instance")
            audio_extension = (
                ".wav"
                if self.transcription_service == TranscriptionServices.AZURE_STT
                else ".mp3"
            )
            
            context.blob_urls["audio_blob_url"] = blob_manager.get_blob_url(
                container=self.audio_container,
                blob_name=f"{context.hash_id}" + audio_extension,
            )
            self.logger.info("Generating Transcript for the input video...")
            context.transcript, local_paths = await transcriber()
            self.logger.info("Successfully generated the transcript for the video.")
            
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

            context.blob_urls["transcript_and_summary_file_url"] = (
                blob_manager.get_blob_url(
                    container=self.video_description_container_name,
                    blob_name=f"{context.hash_id}.json",
                )
            )
            self.logger.info(
                "Logged the transcript and summary file url to the blob urls mapping dictionary"
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

    async def _create_ingest_azurecv_index(self, context: ProcessingContext) -> ProcessingContext:
        """
        This method created and ingest the video into the computer vision indexes - functional version.
        """
        try:
            self.logger.info("Ingesting the video the Computer Vision Index")
            cv_services = ComputerVisionService(video_id=context.hash_id)
            await cv_services.create_index()
            self.logger.info("Successfully created the index!")
            await cv_services.add_video_to_index(blob_url=context.video_url)
            self.logger.info("Successfully added video to the Computer Vision Index")
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while ingesting Video to Computer Vision Index: {e}"
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

    async def _semantic_chunking_chapter_generation(self, context: ProcessingContext, video_url: str, youtube_url: Optional[str] = None) -> ProcessingContext:
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
                base64Frames=context.base64_frames,
                blob_urls=context.blob_urls,
                frame_stacking_grid_size=self.frame_stacking_grid_size,
            )
            self.logger.info(
                "Successfully created an instance of SemanticChunking class!"
            )
            
            context.chapter_responses, context.chapter_transcripts, context.is_already_ingested = (
                await semantic_chunker.run(
                    video_blob_url=video_url, youtube_url=youtube_url
                )
            )
            
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while creating an instance of SemanticChunking class: {e}"
            )
            raise

    async def _get_frames_timestamps(self, context: ProcessingContext, blob_manager) -> ProcessingContext:
        """Extract frames and timestamps from video - functional version."""
        try:
            self.logger.info("Extracting frames and timestamps from the video...")
            context.frames, context.timestamps = await extract_frames(
                video_path=context.video_path
            )
            self.logger.info(
                "Successfully extracted the frames and timestamps from the video!"
            )

            self.logger.info("Encoding frames to base64...")
            context.base64_frames = await encode_frames_to_base64(context.frames)
            self.logger.info("Successfully converted frames to base64 strings!")

            self.logger.info("Saving frames as png files")
            png_paths = await save_frames_as_png(
                context.frames,
                os.path.join(await get_media_folder(), f"Frames", context.hash_id),
            )
            context.local_resources.extend(png_paths)  # Track for cleanup
            self.logger.info("Successfully saved the frames")

            for i, png_path in enumerate(png_paths):
                context.pending_upload_tasks.append(
                    blob_manager.upload_file(
                        container=self.frames_container,
                        blob_name=f"frames/{context.hash_id}/frame_{i}.png",
                        file_path=png_path,
                    )
                )
            self.logger.info("Logged the frames to pending upload tasks list")

            self.logger.info("Saving the timestamps file to the local directory")
            async with aiofiles.open(
                os.path.join(
                    await get_media_folder(), f"timestamps_{context.hash_id}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                await f.write("\n".join(map(str, context.timestamps)))
            self.logger.info(
                f"Successfully saved the timestamps file to the local directory: timestamps_{context.hash_id}.txt"
            )

            context.pending_upload_tasks.append(
                blob_manager.upload_file(
                    container=self.timestamps_container,
                    blob_name=f"timestamps_{context.hash_id}.txt",
                    file_path=os.path.join(
                        await get_media_folder(), f"timestamps_{context.hash_id}.txt"
                    ),
                )
            )
            self.logger.info("Logged the timestamps file to pending upload tasks list")

            context.blob_urls["frames_blob_folder_url"] = blob_manager.get_blob_url(
                container=self.frames_container, blob_name=f"frames/{context.hash_id}"
            )
            self.logger.info(
                "Successfully loggeed the frames blob folder path url to the blob urls mapping dictionary!"
            )
            context.blob_urls["timestamps_blob_url"] = blob_manager.get_blob_url(
                container=self.timestamps_container,
                blob_name=f"timestamps_{context.hash_id}.txt",
            )
            self.logger.info(
                "Successfully logged the timestamps blob url to the blob urls mapping dictionary!"
            )

            del png_paths
            gc.collect()
            return context
        except Exception as e:
            self.logger.exception(
                f"Exception occured while generating the frames and timestamps from video: {e}"
            )
            raise

    async def __call__(self):
        """Main ingestion pipeline method - now supports video splitting and parallel processing."""
        try:
            await self._check_and_compress_video()  # Check file size and compress if needed
            self.logger.info("Video compression check completed!")
            
            # Split video if needed based on conditions
            video_paths, hash_suffixes = await split_video_if_needed(self.video_path)
            self.logger.info(f"Processing {len(video_paths)} video part(s)")
            
            # Track split video files for cleanup
            split_video_cleanup_paths = []
            if len(video_paths) > 1:
                split_video_cleanup_paths.extend(video_paths)
            
            # Process video parts in parallel if split, otherwise process single video
            if len(video_paths) > 1:
                self.logger.info("Processing video parts in parallel for faster execution...")
                
                # Generate base hash ID from Part A video for consistent hash IDs
                part_a_path = video_paths[0]  # Part A is always first
                base_hash_id = await get_file_hash(file_path=part_a_path)
                self.logger.info(f"Generated base hash ID from Part A: {base_hash_id}")
                
                # Create tasks for parallel processing with consistent hash IDs
                tasks = []
                for video_path, hash_suffix in zip(video_paths, hash_suffixes):
                    part_name = "Part A" if hash_suffix == "" else f"Part {hash_suffix}"
                    
                    # Use base hash ID + suffix for consistent naming
                    part_hash_id = base_hash_id + hash_suffix
                    
                    self.logger.info(f"Creating task for {part_name}: {os.path.basename(video_path)}")
                    self.logger.info(f"  Hash ID: {part_hash_id}")
                    
                    # Create asyncio task for processing this video part
                    task = asyncio.create_task(
                        self._process_video_part_parallel(video_path, part_hash_id)
                    )
                    tasks.append(task)
                
                # Execute all video parts in parallel
                self.logger.info(f"Starting parallel processing of {len(tasks)} video parts...")
                await asyncio.gather(*tasks)
                
                self.logger.info("All video parts processed successfully!")
                
            else:
                # Single video processing (no split) - original logic
                self.logger.info("Processing single video (no split required)")
                
                # Generate hash ID early to check for duplicates
                hash_id = await get_file_hash(file_path=self.video_path)
                self.logger.info(f"Generated hash ID: {hash_id}")
                
                # Check if video already exists in the index (early duplicate check)
                is_already_ingested = await check_video_already_ingested(
                    hash_id=hash_id, 
                    index_name=self.index_name
                )
                
                if is_already_ingested:
                    self.logger.info(f"Video with hash_id {hash_id} already exists in index {self.index_name}. Skipping ingestion.")
                    return
                
                self.logger.info("Video not found in index. Proceeding with ingestion...")
                
                # Create processing context
                context = ProcessingContext(
                    hash_id=hash_id,
                    video_path=self.video_path,
                    video_extension=self.video_extension
                )
                
                # Get blob manager
                blob_manager = await self._get_blob_manager()
                
                # Run functional pipeline methods
                context = await self.get_transcription(context, blob_manager)
                self.logger.info("Transcript Generated!")
                
                context = await self._get_frames_timestamps(context, blob_manager)
                self.logger.info("Frames and Timestamps Generated!")
                
                # Upload video to blob
                context.video_url = await file_upload_to_blob(
                    file_path=context.video_path,
                    blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                    container_name=self.video_container,
                )
                self.logger.info("Uploaded local video to blob")

                # Run semantic chunking and chapter generation
                context = await self._semantic_chunking_chapter_generation(
                    context, context.video_url, self.youtube_url
                )
        
                print(context.is_already_ingested, type(context.is_already_ingested))
                if not context.is_already_ingested:
                    self.logger.info(
                    "Chapter generated for the visual summary and successfully ingested to index!"
                    )
                    if self.use_computer_vision_tool:
                        context = await self._create_ingest_azurecv_index(context)
                        self.logger.info(
                            "Computer Vision Index Created and Ingested to computer vision index"
                        )

                    context = await self._merge_visual_summary_with_transcript(context, blob_manager)
                    self.logger.info("Successfully merged Summary & transcript file!")

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
                del context.frames, context.timestamps, context.base64_frames, context.video_url
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
        language: Languages,
        transcription_service: Optional[str] = TranscriptionServices.AZURE_STT.value,
        youtube_urls: Optional[List[str]] = None,
        use_computer_vision_tool: Optional[bool] = False,
        disable_console_log: bool = False,
        frame_stacking_grid_size: int = 4,
        max_concurrent: int = 3
    ) -> List[str]:
        """
        Process multiple videos in parallel using separate pipeline instances.
        
        Args:
            video_paths: List of video file paths to process
            index_name: Azure AI Search index name
            language: Language for transcription
            transcription_service: Transcription service to use
            youtube_urls: Optional list of YouTube URLs (must match video_paths length)
            use_computer_vision_tool: Whether to use computer vision
            disable_console_log: Whether to disable console logging
            frame_stacking_grid_size: Grid size for frame stacking
            max_concurrent: Maximum number of concurrent video processing tasks
            
        Returns:
            List of hash IDs for successfully processed videos
        """
        if youtube_urls and len(youtube_urls) != len(video_paths):
            raise ValueError("youtube_urls length must match video_paths length")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_video(video_path: str, youtube_url: Optional[str] = None) -> Optional[str]:
            """Process a single video with semaphore control."""
            async with semaphore:
                try:
                    # Create separate instance for each video
                    pipeline = cls(
                        video_path=video_path,
                        index_name=index_name,
                        language=language,
                        transcription_service=transcription_service,
                        youtube_url=youtube_url,
                        use_computer_vision_tool=use_computer_vision_tool,
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
            youtube_url = youtube_urls[i] if youtube_urls else None
            task = process_single_video(video_path, youtube_url)
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
            
            # Run functional pipeline methods
            context = await self.get_transcription(context, blob_manager)
            self.logger.info("Transcript Generated!")
            
            context = await self._get_frames_timestamps(context, blob_manager)
            self.logger.info("Frames and Timestamps Generated!")
            
            # Upload video to blob
            context.video_url = await file_upload_to_blob(
                file_path=context.video_path,
                blob_file_name=f"{context.hash_id}" + f"{context.video_extension}",
                container_name=self.video_container,
            )
            self.logger.info("Uploaded local video to blob")

            # Run semantic chunking and chapter generation
            context = await self._semantic_chunking_chapter_generation(
                context, context.video_url, self.youtube_url
            )
    
            if not context.is_already_ingested:
                self.logger.info(
                "Chapter generated for the visual summary and successfully ingested to index!"
                )
                if self.use_computer_vision_tool:
                    context = await self._create_ingest_azurecv_index(context)
                    self.logger.info(
                        "Computer Vision Index Created and Ingested to computer vision index"
                    )

                context = await self._merge_visual_summary_with_transcript(context, blob_manager)
                self.logger.info("Successfully merged Summary & transcript file!")

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
    video_path = "/home/v-amanpatkar/work/chapter_generation_opt/MMCTAgent/mmct/video_pipeline/core/ingestion/ingestion_pipeline.py"
    index = "test_index_a"
    source_language = Languages.ENGLISH_UNITED_STATES
    ingestion = IngestionPipeline(
        video_path=video_path,
        index_name=index,
        transcription_service=TranscriptionServices.AZURE_STT,
        language=source_language,
    )
    asyncio.run(ingestion())
