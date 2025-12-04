from abc import ABC, abstractmethod

class BaseStorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def get_file_url(self, file_name: str, **kwargs) -> str:
        """Generate a URL for a file."""
        pass

    @abstractmethod
    async def upload_file(self, file_name: str, src_file_path: str, **kwargs) -> str:
        """Upload a local file to storage."""
        pass


    @abstractmethod
    async def load_file_to_memory(self, folder: str, file_name: str) -> bytes:
        """Load a file into memory as bytes."""
        pass

    @abstractmethod
    async def close(self):
        """Close the underlying client and cleanup."""
        pass
