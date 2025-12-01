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
        deployment_name:str,
        speech_timeout: Optional[int] = 200,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        api_key: Optional[str] = None,
    ):
        if not endpoint:
            raise ConfigurationException("Azure OpenAI endpoint is required for Whisper Transcription Provider!")

        if not deployment_name:
            raise ConfigurationException("Azure OpenAI deployment name is required for Whisper Transcription Provider!")
        
        if not api_version:
            raise ConfigurationException("Azure OpenAI api version is required for Whisper Transcription Provider!")
    
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

    def _split_audio_file(self, audio_path: str) -> tuple[str, str]:
        """Split audio file into two equal halves.

        Args:
            audio_path: Path to the audio file to split

        Returns:
            Tuple of (first_half_path, second_half_path)
        """
        try:
            logger.info(f"Splitting audio file: {audio_path}")

            # Load audio file
            audio = AudioSegment.from_file(audio_path)

            # Calculate midpoint
            duration_ms = len(audio)
            midpoint_ms = duration_ms // 2

            logger.info(f"Audio duration: {duration_ms}ms, splitting at {midpoint_ms}ms")

            # Split into two halves
            first_half = audio[:midpoint_ms]
            second_half = audio[midpoint_ms:]

            # Determine the original format
            file_ext = os.path.splitext(audio_path)[1].lower()
            # Map common extensions to pydub format names
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

            # Use named temporary files to avoid conflicts
            first_half_fd, first_half_path = tempfile.mkstemp(
                suffix=file_ext, prefix="whisper_split_1_"
            )
            second_half_fd, second_half_path = tempfile.mkstemp(
                suffix=file_ext, prefix="whisper_split_2_"
            )

            # Close file descriptors as pydub will write directly
            os.close(first_half_fd)
            os.close(second_half_fd)

            logger.info(f"Exporting first half to {first_half_path} as {export_format}")
            # Export the splits with explicit parameters to avoid conversion issues
            first_half.export(
                first_half_path,
                format=export_format,
                parameters=["-q:a", "0"],  # High quality, fast encoding
            )

            logger.info(f"Exporting second half to {second_half_path} as {export_format}")
            second_half.export(second_half_path, format=export_format, parameters=["-q:a", "0"])

            logger.info(f"Successfully split audio into: {first_half_path} and {second_half_path}")

            return first_half_path, second_half_path

        except Exception as e:
            logger.error(f"Failed to split audio file: {e}")
            raise ProviderException(f"Failed to split audio file: {e}")

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

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file to text using Azure OpenAI Whisper with automatic retry logic.

        Handles content size limit errors by splitting audio into halves and transcribing separately.
        """
        try:
            response_format = kwargs.get("response_format", "text")

            try:
                # Try transcribing the whole file first
                result = await self._transcribe_single_file(
                    audio_path, self.deployment_name, response_format
                )
                return result

            except Exception as e:
                error_msg = str(e)

                # Check if it's a content size limit error
                if "Maximum content size limit" in error_msg and "exceeded" in error_msg:
                    logger.warning(
                        f"File too large for Whisper API. Splitting into halves: {error_msg}"
                    )

                    # Split the audio file
                    first_half_path, second_half_path = self._split_audio_file(audio_path)

                    try:
                        # Transcribe both halves
                        logger.info("Transcribing first half...")
                        first_result = await self._transcribe_single_file(
                            first_half_path, self.deployment_name, response_format
                        )

                        logger.info("Transcribing second half...")
                        second_result = await self._transcribe_single_file(
                            second_half_path, self.deployment_name, response_format
                        )

                        # Combine results
                        combined_result = f"{first_result} {second_result}"
                        logger.info("Successfully transcribed both halves and combined results")

                        return combined_result

                    finally:
                        # Clean up temporary split files
                        for temp_file in [first_half_path, second_half_path]:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    logger.debug(f"Removed temporary file: {temp_file}")
                            except Exception as cleanup_error:
                                logger.warning(
                                    f"Failed to remove temporary file {temp_file}: {cleanup_error}"
                                )
                else:
                    # Re-raise if it's not a size limit error
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
        **kwargs
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
                audio_path=audio_path,
                language=language,
                response_format=response_format
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
