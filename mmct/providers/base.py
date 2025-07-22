from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion response."""
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate text embedding."""
        pass
    
    @abstractmethod
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, index_name: str, **kwargs) -> List[Dict]:
        """Search for documents."""
        pass
    
    @abstractmethod
    async def index_document(self, document: Dict, index_name: str) -> bool:
        """Index a document."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str, index_name: str) -> bool:
        """Delete a document."""
        pass


class VisionProvider(ABC):
    """Abstract base class for vision providers."""
    
    @abstractmethod
    async def analyze_image(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Analyze image content."""
        pass
    
    @abstractmethod
    async def extract_text(self, image_data: bytes, **kwargs) -> str:
        """Extract text from image."""
        pass


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