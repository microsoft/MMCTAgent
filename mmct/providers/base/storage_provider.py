from abc import ABC, abstractmethod

class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def get_file_url(self, file_name: str, **kwargs) -> str:
        """Generate a URL for a file."""
        pass

    @abstractmethod
    async def save_file(self, file_name: str, file_path: str, **kwargs) -> str:
        """Save a local file to storage."""
        pass

    @abstractmethod
    async def save_base64(self, file_name: str, b64_str: str, **kwargs) -> str:
        """Save base64-encoded data to storage."""
        pass

    @abstractmethod
    async def save_string(self, file_name: str, content: str, **kwargs) -> str:
        """Save a string directly to storage."""
        pass

    @abstractmethod
    async def download_to_file(self, file_name: str, download_path: str, **kwargs) -> str:
        """Download a file to a local file path."""
        pass

    @abstractmethod
    async def download_from_url(self, file_url: str, save_folder: str) -> str:
        """Download a file from its URL to a local folder."""
        pass

    @abstractmethod
    async def close(self):
        """Close the underlying client and cleanup."""
        pass
