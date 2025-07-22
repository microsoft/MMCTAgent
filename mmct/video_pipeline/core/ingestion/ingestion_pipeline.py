import asyncio
import aiofiles
from typing import Optional, Annotated
import os
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

# Load environment variables
load_dotenv(find_dotenv(), override=True)


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
        self.local_resources = []
        self.pending_upload_tasks = []
        self.blob_urls = {}
        self.blob_manager = BlobStorageManager()
        self.original_video_path = video_path

    async def _check_and_compress_video(self):
        """
        Check if video file size exceeds 500 MB and compress if needed.
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
                    self.local_resources.append(compressed_path)  # Track for cleanup
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

    async def get_transcription(self):
        try:
            self.hash_id = await get_file_hash(file_path=self.video_path)
            self.logger.info(
                f"Successfully generated the file hash for the video path: {self.video_path}\nHash Id: {self.hash_id}"
            )
            
            # Rename video file to hash_id.extension
            _, self.video_extension = os.path.splitext(self.video_path)
            video_dir = os.path.dirname(self.video_path)
            new_video_path = os.path.join(video_dir, f"{self.hash_id}{self.video_extension}")
            
            if self.video_path != new_video_path:
                os.rename(self.video_path, new_video_path)
                self.video_path = new_video_path
                self.logger.info(f"Video file renamed to: {self.video_path}")
            transcriber = (
                CloudTranscription(
                    video_path=self.video_path,
                    hash_id=self.hash_id,
                    language=self.language,
                )
                if self.transcription_service == TranscriptionServices.AZURE_STT
                else WhisperTranscription(
                    video_path=self.video_path, hash_id=self.hash_id
                )
            )
            self.logger.info("Initialized the transcriber instance")
            audio_extension = (
                ".wav"
                if self.transcription_service == TranscriptionServices.AZURE_STT
                else ".mp3"
            )
            self.blob_urls["audio_blob_url"] = self.blob_manager.get_blob_url(
                container=self.audio_container,
                blob_name=f"{self.hash_id}" + audio_extension,
            )
            self.logger.info("Generating Transcript for the input video...")
            self.transcript, local_paths = await transcriber()
            self.logger.info("Successfully generated the transcript for the video.")
            self.pending_upload_tasks.append(
                self.blob_manager.upload_file(
                    container=self.audio_container,
                    blob_name=f"{self.hash_id}" + audio_extension,
                    file_path=os.path.join(
                        await get_media_folder(),
                        (f"{self.hash_id}" + audio_extension),
                    ),
                )
            )
            self.logger.info("Added Audio File to pending upload tasks list")

            self.pending_upload_tasks.append(
                self.blob_manager.upload_file(
                    container=self.transcript_container,
                    blob_name=f"transcript_{self.hash_id}.srt",
                    file_path=os.path.join(
                        await get_media_folder(),
                        f"transcript_{self.hash_id}.srt",
                    ),
                )
            )
            self.logger.info("Added Transcript File to pending upload tasks list")

            self.blob_urls["transcript_blob_url"] = self.blob_manager.get_blob_url(
                container=self.transcript_container,
                blob_name=f"transcript_{self.hash_id}.srt",
            )
            self.logger.info(
                "Logged the transcript blob url to the blob urls mapping dictionary"
            )

            self.blob_urls["transcript_and_summary_file_url"] = (
                self.blob_manager.get_blob_url(
                    container=self.video_description_container_name,
                    blob_name=f"{self.hash_id}.json",
                )
            )
            self.logger.info(
                "Logged the transcript and summary file url to the blob urls mapping dictionary"
            )

            self.local_resources.extend(local_paths)
            del local_paths
            gc.collect()
        except Exception as e:
            self.logger.exception(
                f"Exception occured while performing transcription: {e}"
            )
            raise

    async def _create_ingest_azurecv_index(self):
        """
        This method created and ingest the video into the computer vision indexes.
        """
        try:
            self.logger.info("Ingesting the video the Computer Vision Index")
            cv_services = ComputerVisionService(video_id=self.hash_id)
            await cv_services.create_index()
            self.logger.info("Successfully created the index!")
            await cv_services.add_video_to_index(blob_url=self.video_url)
            self.logger.info("Successfully added video to the Computer Vision Index")
        except Exception as e:
            self.logger.exception(
                f"Exception occured while ingesting Video to Computer Vision Index: {e}"
            )
            raise

    async def _merge_visual_summary_with_transcript(self):
        """
        This method merge summaries from chapters and transcript.
        """
        try:
            self.logger.info(
                "Creating an instance of class MergeVisualSummaryWithTranscript"
            )
            merge_summary_transcript = MergeVisualSummaryWithTranscript(
                chapter_responses=self.chapter_responses,
                video_id=self.hash_id,
                full_transcript_string=self.transcript,
                transcripts=self.chapter_transcripts,
            )
            self.logger.info("Merging the visual summary and transcript")
            await merge_summary_transcript()
            self.logger.info("Successfully merged the visual summary and transcript")
            self.pending_upload_tasks.append(
                self.blob_manager.upload_file(
                    container=self.video_description_container_name,
                    blob_name=f"{self.hash_id}.json",
                    file_path=os.path.join(
                        await get_media_folder(), f"{self.hash_id}.json"
                    ),
                )
            )
            self.logger.info(
                "Logged the transcript and summary file url to the blob urls mapping dictionary"
            )
        except Exception as e:
            self.logger.exception(
                f"Exception occured while merging visual summary with transcript: {e}"
            )
            raise

    async def _semantic_chunking_chapter_generation(self):
        """
        This method intitalise the semantic chunker
        """
        try:
            self.logger.info(
                "Creating an instance of SemanticChunking class to perform operations related to semantic chunking"
            )
            self.semantic_chunker = SemanticChunking(
                hash_id=self.hash_id,
                index_name=self.index_name,
                transcript=self.transcript,
                base64Frames=self.base64Frames,
                blob_urls=self.blob_urls,
            )
            self.logger.info(
                "Successfully created an instance of SemanticChunking class!"
            )
        except Exception as e:
            self.logger.exception(
                f"Exception occured while creating an instance of SemanticChunking class: {e}"
            )
            raise

    async def _get_frames_timestamps(self):
        try:
            self.logger.info("Extracting frames and timestamps from the video...")
            self.frames, self.timestamps = await extract_frames(
                video_path=self.video_path
            )
            self.logger.info(
                "Successfully extracted the frames and timestamps from the video!"
            )

            self.logger.info("Encoding frames to base64...")
            self.base64Frames = await encode_frames_to_base64(self.frames)
            self.logger.info("Successfully converted frames to base64 strings!")

            self.logger.info("Saving frames as png files")
            png_paths = await save_frames_as_png(
                self.frames,
                os.path.join(await get_media_folder(), f"Frames", self.hash_id),
            )
            self.local_resources.extend(png_paths)  # Track for cleanup
            self.logger.info("Successfully saved the frames")

            for i, png_path in enumerate(png_paths):
                self.pending_upload_tasks.append(
                    self.blob_manager.upload_file(
                        container=self.frames_container,
                        blob_name=f"frames/{self.hash_id}/frame_{i}.png",
                        file_path=png_path,
                    )
                )
            self.logger.info("Logged the frames to pending upload tasks list")

            self.logger.info("Saving the timestamps file to the local directory")
            async with aiofiles.open(
                os.path.join(
                    await get_media_folder(), f"timestamps_{self.hash_id}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                await f.write("\n".join(map(str, self.timestamps)))
            self.logger.info(
                f"Successfully saved the timestamps file to the local directory: timestamps_{self.hash_id}.txt"
            )

            self.pending_upload_tasks.append(
                self.blob_manager.upload_file(
                    container=self.timestamps_container,
                    blob_name=f"timestamps_{self.hash_id}.txt",
                    file_path=os.path.join(
                        await get_media_folder(), f"timestamps_{self.hash_id}.txt"
                    ),
                )
            )
            self.logger.info("Logged the timestamps file to pending upload tasks list")

            self.blob_urls["frames_blob_folder_url"] = self.blob_manager.get_blob_url(
                container=self.frames_container, blob_name=f"frames/{self.hash_id}"
            )
            self.logger.info(
                "Successfully loggeed the frames blob folder path url to the blob urls mapping dictionary!"
            )
            self.blob_urls["timestamps_blob_url"] = self.blob_manager.get_blob_url(
                container=self.timestamps_container,
                blob_name=f"timestamps_{self.hash_id}.txt",
            )
            self.logger.info(
                "Successfully logged the timestamps blob url to the blob urls mapping dictionary!"
            )

            del png_paths
            gc.collect()
        except Exception as e:
            self.logger.exception(
                f"Exception occured while generating the frames and timestamps from video: {e}"
            )
            raise

    async def __call__(self):
        try:
            await self._check_and_compress_video()  # Check file size and compress if needed
            self.logger.info("Video compression check completed!")
            await self.get_transcription()  # transcribing the audio from video
            self.logger.info("Transcript Generated!")
            await self._get_frames_timestamps()  # extact, encoding & saving the frames
            self.logger.info("Frames and Timestamps Generated!")
            
            self.video_url = await file_upload_to_blob(
                file_path=self.video_path,
                blob_file_name=f"{self.hash_id}" + f"{self.video_extension}",
                container_name=self.video_container,
            )
            self.logger.info("Uploaded local video to blob")

            await self._semantic_chunking_chapter_generation()
            self.chapter_responses, self.chapter_transcripts, self.is_already_ingested = (
                await self.semantic_chunker.run(
                    video_blob_url=self.video_url, youtube_url=self.youtube_url
                )
            )
    
            print(self.is_already_ingested, type(self.is_already_ingested))
            if not self.is_already_ingested:
                self.logger.info(
                "Chapter generated for the visual summary and successfully ingested to index!"
                )
                if self.use_computer_vision_tool:
                    await self._create_ingest_azurecv_index()
                    self.logger.info(
                        "Computer Vision Index Created and Ingested to computer vision index"
                    )

                await self._merge_visual_summary_with_transcript()
                self.logger.info("Successfully merged Summary & transcript file!")

            for batch in chunked(self.pending_upload_tasks, 5):
                upload_results = await asyncio.gather(*batch)
                # Free completed upload tasks to prevent memory buildup
                del upload_results
                gc.collect()
            self.logger.info(
                "Successfully uploaded the files to blob present in the pending upload tasks list!"
            )

            await self.blob_manager.close()
            
            # Clean up local files after successful ingestion
            await remove_file(self.hash_id)
            self.logger.info("Local files cleaned up successfully!")
            
            del self.frames, self.timestamps, self.base64Frames, self.video_url
            gc.collect()
            self.logger.info("Ingestion pipeline ran succesfully!")
        except Exception as e:
            self.logger.exception(
                f"Exception occured while running Ingestion pipeline: {e}"
            )
            raise


if __name__ == "__main__":
    # Example usage - replace with your actual values
    video_path = "path/to/your/video.mp4"
    index = "your-index-name"
    source_language = Languages.ENGLISH_US
    ingestion = IngestionPipeline(
        video_path=video_path,
        index_name=index,
        transcription_service=TranscriptionServices.AZURE_STT,
        language=source_language,
    )
    asyncio.run(ingestion())

