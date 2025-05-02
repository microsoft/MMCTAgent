import asyncio
import aiofiles
from typing import Optional
import os
from loguru import logger
import gc
from mmct.video_pipeline.core.ingestion.transcription.azure_transcription import (
    AzureTranscription,
)
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
)

from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.core.ingestion.transcription.transcription_services import TranscriptionServices

from mmct.blob_store_manager import BlobStorageManager

from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic import (
    SemanticChunking,
)
from mmct.video_pipeline.core.ingestion.azure_cv.azure_computer_vision import (
    AzureComputerVision,
)
from mmct.video_pipeline.core.ingestion.merge_summary_n_transcript.merge_visual_summay_with_transcript import (
    MergeVisualSummaryWithTranscript,
)
from mmct.blob_store_manager import BlobStorageManager
from mmct.video_pipeline.utils.helper import get_media_folder
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class IngestionPipeline:
    """
    IngestionPipeline handles the ingestion, processing, and indexing of a video to prepare it
    for use with the VideoAgent system.

    This pipeline supports transcription using Azure Speech-to-Text ("azure-stt") or OpenAI Whisper,
    and it stores the resulting transcripts, frames, audio, metaData and optionally created index corresponsing to the video in azure computer vision.

    It also uploads all required video-related files (e.g., original video, transcripts, metadata)
    to an Azure Storage account as part of the ingestion process.

    Attributes:
        video_path (str): Path to the video file to be ingested.
        index_name (str): Name of the Azure AI Search index where video data will be stored.
        language (Languages): Language of the video (only Languages Enum), used for transcription and search indexing.
        transcription_service (str): Transcription service to use ("azure-stt" or "whisper"). Defaults to "azure-stt".
        youtube_url (str, optional): Optional YouTube URL associated with the video.
        use_azure_computer_vision (bool): Whether to use Azure Computer Vision for content analysis. Defaults to True.

    Example Usage:
    ---------------
    >>> from mmct.video_pipeline.ingestion import IngestionPipeline
    >>> from mmct.video_pipeline.language import Languages
    >>> import asyncio
    >>> 
    >>> async def run_ingestion():
    >>>     ingestion = IngestionPipeline(
    >>>         video_path="C:/Users/v-amanpatkar/Downloads/Preparation of Neemasthram - Telugu - SERP- Mahabubnagar - Andhra Pradesh.mp4",
    >>>         index_name="general-video-index-v2",
    >>>         language=Languages.TELUGU_INDIA,
    >>>         transcription_service="azure-stt",
    >>>         youtube_url=None,
    >>>         use_azure_computer_vision=True
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
        use_azure_computer_vision: Optional[str] = True,
    ):
        self.video_container = os.getenv("VIDEO_CONTAINER")
        self.audio_container = os.getenv("AUDIO_CONTAINER")
        self.transcript_container = os.getenv("TRANSCRIPT_CONTAINER")
        self.frames_container = os.getenv("FRAMES_CONTAINER")
        self.timestamps_container = os.getenv("TIMESTAMPS_CONTAINER")
        self.summary_n_transcript = os.getenv("SUMMARY_CONTAINER_NAME")
        self.blob_manager = BlobStorageManager()
        self.video_path = video_path
        _, self.video_extension = os.path.splitext(self.video_path)
        self.transcription_service = transcription_service
        self.youtube_url = youtube_url
        self.index_name = index_name
        self.use_azure_computer_vision = use_azure_computer_vision
        self.language = language
        self.local_resources = []
        self.pending_upload_tasks = []

    async def get_transcription(self):
        self.hash_id = await get_file_hash(file_path=self.video_path)
        transcriber = (
            AzureTranscription(
                video_path=self.video_path, hash_id=self.hash_id, language=self.language
            )
            if self.transcription_service == TranscriptionServices.AZURE_STT
            else WhisperTranscription(video_path=self.video_path, hash_id=self.hash_id)
        )
        audio_extension = (
            ".wav" if self.transcription_service == TranscriptionServices.AZURE_STT else ".mp3"
        )
        self.transcript, local_paths = await transcriber()
        logger.info(f"ingestion first transcript:{self.transcript}")
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

        self.local_resources.extend(local_paths)
        del local_paths
        gc.collect()

    async def _create_ingest_azurecv_index(self):
        """
        This method created and ingest the video into the azure computer vision indexes.
        """
        azure_cv = AzureComputerVision(video_id=self.hash_id)
        await azure_cv.create_index()
        await azure_cv.add_video_to_index(blob_url=self.video_url)

    async def _merge_visual_summary_with_transcript(self):
        """
        This method merge summaries from chapters and transcript.
        """
        merge_summary_transcript = MergeVisualSummaryWithTranscript(
            chapter_responses=self.chapter_responses,
            video_id=self.hash_id,
            full_transcript_string=self.transcript,
            transcripts=self.chapter_transcripts,
        )
        await merge_summary_transcript()
        self.pending_upload_tasks.append(
            self.blob_manager.upload_file(
                container=self.summary_n_transcript,
                blob_name=f"{self.hash_id}.json",
                file_path=os.path.join(await get_media_folder(),f"{self.hash_id}.json"),
            )
        )

    async def _semantic_chunking_chapter_generation(self):
        """
        This method intitalise the semantic chunker
        """
        self.semantic_chunker = SemanticChunking(
            hash_id=self.hash_id,
            index_name=self.index_name,
            transcript=self.transcript,
            base64Frames=self.base64Frames,
        )

    async def _get_frames_timestamps(self):
        self.frames, self.timestamps = await extract_frames(video_path=self.video_path)
        self.base64Frames = await encode_frames_to_base64(self.frames)
        png_paths = await save_frames_as_png(
            self.frames, os.path.join(await get_media_folder(),f"Frames",self.hash_id)
        )
        self.local_resources.extend(png_paths)  # Track for cleanup

        for i, png_path in enumerate(png_paths):
            self.pending_upload_tasks.append(
                self.blob_manager.upload_file(
                    container=self.frames_container,
                    blob_name=f"frames/{self.hash_id}/frame_{i}.png",
                    file_path=png_path,
                )
            )

        async with aiofiles.open(
           os.path.join(await get_media_folder(), f"timestamps_{self.hash_id}.txt"), "w", encoding="utf-8"
        ) as f:
            await f.write("\n".join(map(str, self.timestamps)))

        self.pending_upload_tasks.append(
            self.blob_manager.upload_file(
                container=self.timestamps_container,
                blob_name=f"timestamps_{self.hash_id}.txt",
                file_path= os.path.join(await get_media_folder(), f"timestamps_{self.hash_id}.txt"),
            )
        )

        del png_paths
        gc.collect()

    async def __call__(self):
        try:
            await self.get_transcription()  # transcribing the audio from video
            print("Got transcript!")

            await self._get_frames_timestamps()  # extact, encoding & saving the frames
            print("Got frames!")
            self.video_url = await file_upload_to_blob(
                file_path=self.video_path,
                blob_file_name=f"{self.hash_id}" + f"{self.video_extension}",
                container_name=self.video_container,
            )  # uploading local video to blob
            print("Uploaded local video to blob")
            await self._semantic_chunking_chapter_generation()
            self.chapter_responses, self.chapter_transcripts = (
                await self.semantic_chunker.run(
                    video_blob_url=self.video_url, youtube_url=self.youtube_url
                )
            )
            print("chapter generated and ingestion to index(AI Search)!")
            if self.use_azure_computer_vision:
                await self._create_ingest_azurecv_index()
                print("Created and Ingested to azure computer vision index")

            await self._merge_visual_summary_with_transcript()
            print("Got Summary & transcript merged file!")

            for batch in chunked(self.pending_upload_tasks, 5):
                upload_results = await asyncio.gather(*batch)
                # Free completed upload tasks to prevent memory buildup
                del upload_results
                gc.collect()

            await self.blob_manager.close()
            del self.frames, self.timestamps, self.base64Frames, self.video_url
            gc.collect()
            print("Ingestion pipeline ran succesfully!")
        except Exception as e:
            raise Exception(f"Error while ingestion:{e}")
        finally:
            pass


if __name__ == "__main__":
    video_path = "C:/Users/v-amanpatkar/Downloads/Preparation of Neemasthram - Telugu - SERP- Mahabubnagar - Andhra Pradesh.mp4"
    index = "general-video-index-v2"
    source_language = Languages.TELUGU_INDIA
    ingestion = IngestionPipeline(
        video_path=video_path,
        index_name=index,
        transcription_service=TranscriptionServices.WHISPER,
        language=source_language,
    )
    asyncio.run(ingestion())
