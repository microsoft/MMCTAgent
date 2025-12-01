from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class BaseTranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_data: Audio data as bytes
            language: Language code for transcription
            **kwargs: Additional provider-specific parameters

        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> Any:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            **kwargs: Additional provider-specific parameters

        Returns:
            Transcribed text (format varies by provider: str, List[Dict], etc.)
        """
        pass

    async def transcribe_video(
        self,
        video_path: str,
        hash_id: str,
        output_dir: Optional[str] = None,
        language: str = None,
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Transcribe video file to text (optional method for providers that support video).

        Args:
            video_path: Path to the video file
            hash_id: Unique identifier for the video
            output_dir: Directory to save audio and transcript files
            language: Language code for transcription
            **kwargs: Additional provider-specific parameters

        Returns:
            Tuple of (transcript_content, list_of_local_file_paths)
            - transcript_content: The transcribed text
            - list_of_local_file_paths: List of temporary file paths created
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct video transcription. "
            "Extract audio first and use transcribe_file() instead."
        )

    async def close(self):
        """
        Close the transcription provider and cleanup resources (optional method).

        Override this method if your provider needs cleanup (e.g., closing connections, clients).
        Default implementation does nothing.
        """
        pass