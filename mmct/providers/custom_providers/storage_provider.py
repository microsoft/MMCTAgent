import os
import base64
import aiofiles
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from loguru import logger
from typing import Dict, Any
from mmct.providers.base import StorageProvider
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException


class LocalStorageProvider(StorageProvider):
    """Local filesystem-based storage provider."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Local Storage Provider.

        Args:
            config: {
                        "base_path": str -> Root directory for local storage (default: ./local_storage)
                    }
        """
        self.config = config
        self.base_path = Path(config.get("base_path", "./local_storage")).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageProvider initialized at {self.base_path}")

    def _get_blob_path(self, container: str, blob_name: str) -> Path:
        """Return full path to blob file, creating parent directories if needed."""
        blob_path = self.base_path / container / blob_name
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        return blob_path

    async def get_blob_url(self, container: str, blob_name: str) -> str:
        """
        Generate file:// URL for a local blob.
        Ensures consistent format across OS (handles Windows drive letters).
        """
        blob_path = self._get_blob_path(container, blob_name)
        abs_path = blob_path.resolve()

        # Proper file:// handling on Windows (e.g., file:///C:/path/to/file)
        if os.name == "nt":
            url = f"file:///{abs_path.as_posix()}"
        else:
            url = abs_path.as_uri()

        return url

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        """Copy a local file into the local storage directory."""
        try:
            dest_path = self._get_blob_path(container, blob_name)
            async with aiofiles.open(file_path, "rb") as src, aiofiles.open(dest_path, "wb") as dst:
                while chunk := await src.read(1024 * 1024):
                    await dst.write(chunk)
            logger.info(f"File uploaded to {dest_path}")
            return await self.get_blob_url(container, blob_name)
        except Exception as e:
            logger.error(f"Error uploading file locally: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        """Upload base64-encoded content."""
        try:
            dest_path = self._get_blob_path(container, blob_name)
            data = base64.b64decode(b64_str)
            async with aiofiles.open(dest_path, "wb") as f:
                await f.write(data)
            logger.info(f"Base64 data saved to {dest_path}")
            return await self.get_blob_url(container, blob_name)
        except Exception as e:
            logger.error(f"Error uploading base64 content: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        """Upload plain text/string content."""
        try:
            dest_path = self._get_blob_path(container, blob_name)
            async with aiofiles.open(dest_path, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.info(f"String uploaded to {dest_path}")
            return await self.get_blob_url(container, blob_name)
        except Exception as e:
            logger.error(f"Error uploading string content: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        """Copy blob from local storage to a specified path."""
        try:
            src_path = self._get_blob_path(container, blob_name)
            if not src_path.exists():
                raise FileNotFoundError(f"Blob not found: {src_path}")

            dst_path = Path(download_path)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(src_path, "rb") as src, aiofiles.open(dst_path, "wb") as dst:
                while chunk := await src.read(1024 * 1024):
                    await dst.write(chunk)

            logger.info(f"Downloaded {src_path} to {dst_path}")
            return str(dst_path)
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        """
        Handle file:// URLs for local blobs.
        Since local, only file:// URLs are supported.
        """
        try:
            parsed = urlparse(blob_url)
            if parsed.scheme != "file":
                raise ProviderException("Only file:// URLs supported in LocalStorageProvider")

            blob_path = Path(parsed.path)
            if not blob_path.exists():
                raise FileNotFoundError(f"File not found: {blob_path}")

            save_folder = Path(save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
            local_path = save_folder / blob_path.name

            async with aiofiles.open(blob_path, "rb") as src, aiofiles.open(local_path, "wb") as dst:
                while chunk := await src.read(1024 * 1024):
                    await dst.write(chunk)

            logger.info(f"Copied from {blob_path} to {local_path}")
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading from URL: {e}")
            raise ProviderException(str(e))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def load_blob_to_memory(self, container: str, blob_name: str) -> bytes:
        """Load a local blob (file) into memory as bytes."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            if not blob_path.exists():
                raise FileNotFoundError(f"Blob not found: {blob_path}")

            async with aiofiles.open(blob_path, "rb") as f:
                data = await f.read()

            logger.info(f"Loaded blob {blob_name} ({len(data)} bytes) into memory")
            return data
        except Exception as e:
            logger.error(f"Error loading blob into memory: {e}")
            raise ProviderException(str(e))

    async def close(self):
        """No-op for local provider (for interface consistency)."""
        logger.debug("LocalStorageProvider closed (no-op).")
        pass
