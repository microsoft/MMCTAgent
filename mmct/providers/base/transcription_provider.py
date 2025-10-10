from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file to text."""
        pass