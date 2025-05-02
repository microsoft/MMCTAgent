""""
This module provides transcription services for video ingestion.
"""
from enum import Enum
class TranscriptionServices(Enum):
    
    AZURE_STT = "azure-stt"
    WHISPER = "whisper"