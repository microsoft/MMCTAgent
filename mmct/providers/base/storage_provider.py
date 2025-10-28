from abc import ABC, abstractmethod

class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def get_blob_url(self, container: str, blob_name: str) -> str:
        """Generate a URL for a blob."""
        pass

    @abstractmethod
    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        """Upload a local file to storage."""
        pass

    @abstractmethod
    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        """Upload base64-encoded data to storage."""
        pass

    @abstractmethod
    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        """Upload a string directly to storage."""
        pass

    @abstractmethod
    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        """Download a blob to a local file path."""
        pass

    @abstractmethod
    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        """Download a blob from its URL to a local folder."""
        pass

    @abstractmethod
    async def close(self):
        """Close the underlying client and cleanup."""
        pass
