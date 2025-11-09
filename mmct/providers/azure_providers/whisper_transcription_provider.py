from loguru import logger
from typing import Dict, Any
import os
import tempfile
from pydub import AudioSegment
from mmct.utils.error_handler import ProviderException, ConfigurationException, handle_exceptions, convert_exceptions
from mmct.providers.base import TranscriptionProvider
from mmct.providers.credentials import AzureCredentials
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider

class WhisperTranscriptionProvider(TranscriptionProvider):
    """Azure OpenAI Whisper transcription provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client for Whisper."""
        try:
            logger.info(f"Transcription config received: {self.config}")
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)

            logger.info(f"Endpoint from config: {endpoint}")
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required for Whisper API")

            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential,
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")

                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client for transcription: {e}")

    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes to text using Whisper."""
        raise NotImplementedError("Whisper API requires file-based transcription. Use transcribe_file() instead.")

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
                '.mp3': 'mp3',
                '.wav': 'wav',
                '.m4a': 'mp4',
                '.mp4': 'mp4',
                '.flac': 'flac',
                '.ogg': 'ogg',
                '.webm': 'webm'
            }
            export_format = format_map.get(file_ext, 'mp3')
            
            # Use named temporary files to avoid conflicts
            first_half_fd, first_half_path = tempfile.mkstemp(suffix=file_ext, prefix='whisper_split_1_')
            second_half_fd, second_half_path = tempfile.mkstemp(suffix=file_ext, prefix='whisper_split_2_')
            
            # Close file descriptors as pydub will write directly
            os.close(first_half_fd)
            os.close(second_half_fd)
            
            logger.info(f"Exporting first half to {first_half_path} as {export_format}")
            # Export the splits with explicit parameters to avoid conversion issues
            first_half.export(
                first_half_path, 
                format=export_format,
                parameters=["-q:a", "0"]  # High quality, fast encoding
            )
            
            logger.info(f"Exporting second half to {second_half_path} as {export_format}")
            second_half.export(
                second_half_path, 
                format=export_format,
                parameters=["-q:a", "0"]
            )
            
            logger.info(f"Successfully split audio into: {first_half_path} and {second_half_path}")
            
            return first_half_path, second_half_path
            
        except Exception as e:
            logger.error(f"Failed to split audio file: {e}")
            raise ProviderException(f"Failed to split audio file: {e}")

    async def _transcribe_single_file(self, audio_path: str, deployment_name: str, response_format: str) -> str:
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
                file=audio_file,
                model=deployment_name,
                response_format=response_format
            )
        return result

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file to text using Azure OpenAI Whisper with automatic retry logic.
        
        Handles content size limit errors by splitting audio into halves and transcribing separately.
        """
        try:
            deployment_name = self.config.get("deployment_name")
            if not deployment_name:
                raise ConfigurationException("Azure OpenAI Whisper deployment name is required")

            response_format = kwargs.get("response_format", "text")

            try:
                # Try transcribing the whole file first
                result = await self._transcribe_single_file(audio_path, deployment_name, response_format)
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a content size limit error
                if "Maximum content size limit" in error_msg and "exceeded" in error_msg:
                    logger.warning(f"File too large for Whisper API. Splitting into halves: {error_msg}")
                    
                    # Split the audio file
                    first_half_path, second_half_path = self._split_audio_file(audio_path)
                    
                    try:
                        # Transcribe both halves
                        logger.info("Transcribing first half...")
                        first_result = await self._transcribe_single_file(first_half_path, deployment_name, response_format)
                        
                        logger.info("Transcribing second half...")
                        second_result = await self._transcribe_single_file(second_half_path, deployment_name, response_format)
                        
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
                                logger.warning(f"Failed to remove temporary file {temp_file}: {cleanup_error}")
                else:
                    # Re-raise if it's not a size limit error
                    raise
                    
        except Exception as e:
            logger.error(f"Azure Whisper transcription failed: {e}")
            raise ProviderException(f"Azure Whisper transcription failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct audio API access."""
        return self.client

    async def close(self):
        """Close the transcription client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI transcription client")
            await self.client.close()
