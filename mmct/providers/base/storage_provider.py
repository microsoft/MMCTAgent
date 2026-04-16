"""Base interface for blob storage providers.

This module defines the abstract interface for storing and retrieving files 
(e.g., keyframes, videos, logs) in the MMCT system, supporting both local 
and cloud storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseStorageProvider(ABC):
    """Abstract base class for storage providers.

    All storage provider implementations must inherit from this class to handle 
    file lifecycle operations such as uploading, URL generation, and cleanup.
    """

    @abstractmethod
    async def get_file_url(self, file_name: str, **kwargs: Any) -> str:
        """Generates a transient or permanent URL for accessing a stored file.

        Args:
            file_name: The name or relative path of the file in storage.
            **kwargs: Additional provider-specific parameters (e.g., expiration).

        Returns:
            str: The public or pre-signed URL for the file.
        """
        pass

    @abstractmethod
    async def upload_file(self, file_name: str, src_file_path: str, **kwargs: Any) -> str:
        """Uploads a local file to the storage backend.

        Args:
            file_name: The destination name/path in storage.
            src_file_path: The local filesystem path to the source file.
            **kwargs: Additional provider-specific parameters (e.g., tags).

        Returns:
            str: The reference or URL of the uploaded file.
        """
        pass

    @abstractmethod
    async def load_file_to_memory(self, folder: str, file_name: str) -> bytes:
        """Downloads a file and loads its contents into memory as bytes.

        Args:
            folder: The storage container or directory name.
            file_name: The specific file to download.

        Returns:
            bytes: The raw data of the file.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes the underlying storage client and releases resources."""
        pass
