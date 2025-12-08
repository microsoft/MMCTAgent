from loguru import logger
from typing import Union, Optional, List, Tuple
import os
import tempfile
import asyncio
import aiofiles
from pydub import AudioSegment
from mmct.utils.error_handler import (
    ProviderException,
    ConfigurationException,
    handle_exceptions,
    convert_exceptions,
)
from mmct.providers.base import BaseTranscriptionProvider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider


class WhisperTranscriptionProvider(BaseTranscriptionProvider):
    """Azure OpenAI Whisper transcription provider implementation."""

    def __init__(
        self,
        endpoint: str,
        api_version: str,
        deployment_name: str,
        speech_timeout: Optional[int] = 200,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        api_key: Optional[str] = None,
    ):
        if not endpoint:
            raise ConfigurationException(
                "Azure OpenAI endpoint is required for Whisper Transcription Provider!"
            )

        if not deployment_name:
            raise ConfigurationException(
                "Azure OpenAI deployment name is required for Whisper Transcription Provider!"
            )

        if not api_version:
            raise ConfigurationException(
                "Azure OpenAI api version is required for Whisper Transcription Provider!"
            )

        # Validate that exactly one of credentials or api_key is provided
        if credentials is None and api_key is None:
            raise ConfigurationException("Either credentials or api_key must be provided!")

        self.credentials = credentials
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.api_key = api_key
        self.timeout = speech_timeout
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client for Whisper."""
        try:
            if self.credentials is not None:
                token_provider = get_bearer_token_provider(
                    self.credentials, "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=self.timeout,
                )
            else:
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
        except Exception as e:
            raise ProviderException(
                f"Failed to initialize Azure OpenAI client for transcription: {e}"
            )

    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes to text using Whisper."""
        raise NotImplementedError(
            "Whisper API requires file-based transcription. Use transcribe_file() instead."
        )

    def _split_audio_into_chunks(
        self, audio_path: str, chunk_duration_ms: int = 10 * 60 * 1000
    ) -> List[str]:
        """Split audio file into fixed-duration chunks (default 10 minutes).

        Args:
            audio_path: Path to the audio file to split
            chunk_duration_ms: Duration of each chunk in milliseconds (default 10 mins)

        Returns:
            List of paths to temporary chunk files
        """
        chunk_paths = []
        try:
            logger.info(f"Splitting audio file into chunks of {chunk_duration_ms}ms: {audio_path}")

            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)

            # Determine export format
            file_ext = os.path.splitext(audio_path)[1].lower()
            format_map = {
                ".mp3": "mp3",
                ".wav": "wav",
                ".m4a": "mp4",
                ".mp4": "mp4",
                ".flac": "flac",
                ".ogg": "ogg",
                ".webm": "webm",
            }
            export_format = format_map.get(file_ext, "mp3")

            # Split into chunks
            for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
                end_ms = min(start_ms + chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]

                # Create temp file for chunk
                chunk_fd, chunk_path = tempfile.mkstemp(
                    suffix=f"_{i}{file_ext}", prefix="whisper_chunk_"
                )
                os.close(chunk_fd)

                logger.debug(f"Exporting chunk {i} ({start_ms}-{end_ms}ms) to {chunk_path}")
                chunk.export(chunk_path, format=export_format, parameters=["-q:a", "0"])
                chunk_paths.append(chunk_path)

            logger.info(f"Successfully split into {len(chunk_paths)} chunks")
            return chunk_paths

        except Exception as e:
            # Cleanup any chunks created so far
            for path in chunk_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass
            logger.error(f"Failed to split audio file: {e}")
            raise ProviderException(f"Failed to split audio file: {e}")

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _merge_chunks(self, chunks_results: List[any], chunk_duration_ms: int) -> List[dict]:
        """Merge segment results from multiple chunks, shifting timestamps."""
        # This implementation expects chunks_results to be list of objects with a 'segments' attribute
        # commonly returned by Azure OpenAI when response_format="verbose_json"

        all_segments = []
        time_offset_s = 0.0

        for i, result in enumerate(chunks_results):
            # The result object from client.audio.translations.create(..., response_format="verbose_json")
            # is a Transcription object which has a segments list.
            # Each segment has start, end, text, etc.

            if hasattr(result, "segments"):
                segments = result.segments
            elif isinstance(result, dict) and "segments" in result:
                segments = result["segments"]
            else:
                # Fallback if raw text or unexpected format
                logger.warning(
                    f"Chunk {i} result does not have segments. Appending raw text as single segment."
                )
                all_segments.append(
                    {
                        "start": time_offset_s,
                        "end": time_offset_s + (chunk_duration_ms / 1000),
                        "text": str(result),
                    }
                )
                time_offset_s += chunk_duration_ms / 1000
                continue

            for segment in segments:
                # Handle both object access and dict access
                if isinstance(segment, dict):
                    seg_start = segment.get("start", 0.0)
                    seg_end = segment.get("end", 0.0)
                    text = segment.get("text", "")
                else:
                    seg_start = getattr(segment, "start", 0.0)
                    seg_end = getattr(segment, "end", 0.0)
                    text = getattr(segment, "text", "")

                # Create new segment with shifted timestamps
                new_segment = {
                    "start": seg_start + time_offset_s,
                    "end": seg_end + time_offset_s,
                    "text": text,
                }
                all_segments.append(new_segment)

            # Increment offset by the chunk duration (or strict duration of this chunk?)
            # Using fixed chunk duration is safer to align with the splitting logic,
            # effectively assuming the split was perfect.
            time_offset_s += chunk_duration_ms / 1000

        return all_segments

    def _convert_to_output_format(self, segments: List[dict], output_format: str) -> str:
        """Convert merged segments to the requested output format."""
        if output_format == "srt":
            output = []
            for i, seg in enumerate(segments, 1):
                start = self._format_timestamp(seg["start"])
                end = self._format_timestamp(seg["end"])
                text = seg["text"].strip()
                output.append(f"{i}\n{start} --> {end}\n{text}\n")
            return "\n".join(output)

        elif output_format == "vtt":
            output = ["WEBVTT\n"]
            for seg in segments:
                # VTT uses dot for millis
                start = self._format_timestamp(seg["start"]).replace(",", ".")
                end = self._format_timestamp(seg["end"]).replace(",", ".")
                text = seg["text"].strip()
                output.append(f"{start} --> {end}\n{text}\n")
            return "\n".join(output)

        elif output_format == "json":
            import json

            return json.dumps({"segments": segments}, indent=2)

        else:  # "text" or default
            return " ".join([seg["text"].strip() for seg in segments])

    async def _transcribe_single_file(
        self, audio_path: str, deployment_name: str, response_format: str
    ) -> str:
        """Transcribe a single audio file without retry logic (used internally).

        Args:
            audio_path: Path to audio file
            deployment_name: Azure OpenAI deployment name
            response_format: Response format (text, json, etc.)

        Returns:
            Transcription result
        """
        with open(audio_path, "rb") as audio_file:
            result = await self.client.audio.translations.create(
                file=audio_file, model=deployment_name, response_format=response_format
            )
        return result

    async def _process_large_file(
        self, audio_path: str, output_format: str, deployment_name: str
    ) -> str:
        """Process a large audio file by splitting, transcribing chunks, and merging results."""
        chunk_paths = []
        try:
            # Step 1: Split into 10-minute chunks
            chunk_paths = self._split_audio_into_chunks(audio_path)

            chunk_results = []

            # Step 2: Transcribe each chunk
            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")

                # We FORCE verbose_json for chunks to get timestamps for accurate merging
                # independent of what the user asked for.
                # We will convert to user format at the end.
                result = await self._transcribe_single_file(
                    chunk_path, deployment_name, response_format="verbose_json"
                )
                chunk_results.append(result)

            # Step 3: Merge results
            logger.info("Merging chunk results...")
            merged_segments = self._merge_chunks(chunk_results, chunk_duration_ms=10 * 60 * 1000)

            # Step 4: Convert to requested format
            logger.info(f"Converting merged results to format: {output_format}")
            final_output = self._convert_to_output_format(merged_segments, output_format)

            return final_output

        finally:
            # Cleanup chunks
            for path in chunk_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary chunk {path}: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file to text using Azure OpenAI Whisper with automatic retry logic.

        Handles large files automatically by chunking.
        """
        try:
            response_format = kwargs.get("response_format", "text")

            # Check file size (approx 24MB limit gives some buffer for API overhead)
            file_size = os.path.getsize(audio_path)
            limit_bytes = 24 * 1024 * 1024  # 24 MB

            if file_size > limit_bytes:
                logger.info(
                    f"File size {file_size} bytes exceeds limit {limit_bytes} bytes. Using chunked processing."
                )
                return await self._process_large_file(
                    audio_path, response_format, self.deployment_name
                )

            # Try transcribing the whole file directly
            try:
                result = await self._transcribe_single_file(
                    audio_path, self.deployment_name, response_format
                )
                return result

            except Exception as e:
                error_msg = str(e)
                # Fallback to chunking if API rejects it despite size check (e.g. size vs duration limits)
                if "Maximum content size limit" in error_msg:
                    logger.warning(
                        f"API rejected file size. Falling back to chunked processing: {error_msg}"
                    )
                    return await self._process_large_file(
                        audio_path, response_format, self.deployment_name
                    )
                raise

        except Exception as e:
            logger.error(f"Azure Whisper transcription failed: {e}")
            raise ProviderException(f"Azure Whisper transcription failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct audio API access."""
        return self.client

    async def _extract_audio_from_video(self, video_path: str, output_path: str) -> None:
        """
        Extract audio from video file using FFmpeg.

        Args:
            video_path: Path to the video file
            output_path: Path where audio file should be saved

        Raises:
            ProviderException: If audio extraction fails
        """
        try:
            logger.info(f"Extracting audio from video: {video_path}")

            # Run FFmpeg in a subprocess to extract audio
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                video_path,
                "-q:a",
                "0",  # Best quality
                "-map",
                "a",  # Extract audio stream
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Wait for the process to complete
            returncode = await process.wait()

            if returncode != 0:
                raise ProviderException(f"FFmpeg failed with return code {returncode}")

            logger.info(f"Successfully extracted audio to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            raise ProviderException(f"Failed to extract audio from video: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_video(
        self,
        video_path: str,
        hash_id: str,
        output_dir: Optional[str] = None,
        language: str = None,
        **kwargs,
    ) -> Tuple[str, List[str]]:
        """
        Transcribe video file by extracting audio and transcribing it.

        Args:
            video_path: Path to the video file
            hash_id: Unique identifier for the video
            output_dir: Directory to save audio and transcript files (defaults to current working directory's media folder)
            language: Language code for transcription
            **kwargs: Additional parameters (e.g., response_format)

        Returns:
            Tuple of (transcript_content, list_of_local_file_paths)
            - transcript_content: The transcribed text
            - list_of_local_file_paths: List of paths to temporary files created (audio, transcript)

        Raises:
            ProviderException: If transcription fails
        """
        local_files = []

        try:
            # Determine output directory
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "media")
                os.makedirs(output_dir, exist_ok=True)

            # Step 1: Extract audio from video
            audio_path = os.path.join(output_dir, f"{hash_id}.mp3")
            await self._extract_audio_from_video(video_path, audio_path)
            local_files.append(audio_path)
            logger.info(f"Audio extracted and saved to: {audio_path}")

            # Step 2: Transcribe the audio file
            response_format = kwargs.get("response_format", "srt")
            logger.info(f"Starting transcription with format: {response_format}")

            transcript = await self.transcribe_file(
                audio_path=audio_path, language=language, response_format=response_format
            )
            logger.info("Successfully generated transcript")

            # Step 3: Save transcript to file
            transcript_path = os.path.join(output_dir, f"transcript_{hash_id}.srt")
            async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                await f.write(transcript)
            local_files.append(transcript_path)
            logger.info(f"Transcript saved to: {transcript_path}")

            return transcript, local_files

        except Exception as e:
            # Clean up any files created before the error
            for file_path in local_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up file after error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")

            logger.error(f"Video transcription failed: {e}")
            raise ProviderException(f"Video transcription failed: {e}")

    async def close(self):
        """Close the transcription client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI transcription client")
            await self.client.close()
