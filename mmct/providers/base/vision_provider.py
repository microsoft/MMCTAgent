from abc import ABC, abstractmethod
from typing import Dict, Any

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

    @abstractmethod
    async def close(self):
        """Close the provider and cleanup resources."""
        pass