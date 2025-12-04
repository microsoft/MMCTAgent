import os
import base64
import aiofiles
from pathlib import Path
from urllib.parse import urlparse
from loguru import logger
from typing import Dict, Any, Optional
from mmct.providers.base import BaseStorageProvider
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException


class LocalStorageProvider(BaseStorageProvider):
    """Local filesystem-based storage provider."""

    def __init__(self, base_path: Optional[str] = "./local_storage"):
        """
        Initialize Local Storage Provider.

        Args:
              "base_path": str -> Root directory for local storage (default: ./local_storage)
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageProvider initialized at {self.base_path}")

    def _get_file_path(self, folder: str, file_name: str) -> Path:
        """Return full path to file, creating parent directories if needed."""
        file_path = self.base_path / folder / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    async def get_file_url(self, file_name: str, **kwargs) -> str:
        """
        Generate file:// URL for a local file.
        Ensures consistent format across OS (handles Windows drive letters).
        """
        folder_name = kwargs.pop("folder_name")
        file_path = self._get_file_path(folder=folder_name, file_name=file_name)
        abs_path = file_path.resolve()

        # Proper file:// handling on Windows (e.g., file:///C:/path/to/file)
        if os.name == "nt":
            url = f"file:///{abs_path.as_posix()}"
        else:
            url = abs_path.as_uri()

        return url

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def save_file(self, file_name: str, src_file_path: str, **kwargs) -> str:
        """Copy a local file into the local storage directory."""
        try:
            folder_name = kwargs.pop("folder_name")
            dest_path = self._get_file_path(folder=folder_name,file_name=file_name)
            async with aiofiles.open(src_file_path, "rb") as src, aiofiles.open(dest_path, "wb") as dst:
                while chunk := await src.read(1024 * 1024):
                    await dst.write(chunk)
            logger.info(f"File uploaded to {dest_path}")
            return await self.get_file_url(file_name=file_name, folder_name=folder_name)
        except Exception as e:
            logger.error(f"Error uploading file locally: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def load_file_to_memory(self, folder: str, file_name: str) -> bytes:
        """Load a local file) into memory as bytes."""
        try:
            file_path = self._get_file_path(folder=folder, file_name=file_name)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()

            logger.info(f"Loaded file {file_name} ({len(data)} bytes) into memory")
            return data
        except Exception as e:
            logger.error(f"Error loading file into memory: {e}")
            raise ProviderException(str(e))

    async def close(self):
        """No-op for local provider (for interface consistency)."""
        logger.debug("LocalStorageProvider closed (no-op).")
        pass
