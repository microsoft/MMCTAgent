import os
import base64
from pathlib import Path
import aiofiles
from urllib.parse import urlparse
from azure.storage.blob.aio import BlobServiceClient
from loguru import logger
from typing import Dict, Any
from mmct.providers.base import StorageProvider
from mmct.providers.credentials import AzureCredentials
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException


class AzureStorageProvider(StorageProvider):
    """Azure Blob Storage provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure Storage Provider.

        Args:
            config: Configuration dictionary with:
                - account_url: Azure Storage account URL (or use BLOB_ACCOUNT_URL env var)
                - use_managed_identity: Whether to use managed identity (default: True)
        """
        self.config = config
        self.credential = None
        self.service_client = None

    def _initialize(self):
        """Initialize credential and service client asynchronously."""
        if self.service_client is None:
            try:
                use_managed_identity = self.config.get("use_managed_identity") or os.getenv("STORAGE_USE_MANAGED_IDENTITY", "true").lower() == "true"

                if use_managed_identity:
                    self.credential = AzureCredentials.get_async_credentials()
                else:
                    # For non-managed identity, you'd use connection string or SAS token
                    raise ConfigurationException("Non-managed identity auth not yet implemented for blob storage")

                account_url = self.config.get("account_url") or os.getenv("STORAGE_ACCOUNT_URL")
                if not account_url:
                    raise ConfigurationException("Azure Storage account_url is required")

                self.service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.credential,
                )
                logger.info("Successfully initialized Azure Blob Storage client")
            except Exception as e:
                logger.exception(f"Failed to initialize Azure Blob Storage client: {e}")
                raise ProviderException(f"Failed to initialize Azure Blob Storage client: {e}")

    def _ensure_initialized(self):
        """Ensure the client is initialized before operations."""
        if self.service_client is None:
            self._initialize()

    async def get_blob_url(self, container: str, blob_name: str) -> str:
        """
        Generate a URL for a blob that doesn't yet exist in storage.

        Args:
            container: The container name where the blob would be stored
            blob_name: The name for the potential future blob

        Returns:
            str: The unencoded URL that the blob would have when uploaded
        """
        await self._ensure_initialized()

        try:
            # Use service client URL if available, otherwise fall back to config
            if self.service_client:
                url = f"{self.service_client.url}/{container}/{blob_name}"
            else:
                account_url = self.config.get("account_url") or os.getenv("STORAGE_ACCOUNT_URL")
                if not account_url:
                    raise ConfigurationException("Azure Storage account_url is required")
                url = f"{account_url.rstrip('/')}/{container}/{blob_name}"

            logger.info(f"Generated blob URL: {url}")
            return url
        except ConfigurationException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate blob URL: {e}")
            raise ProviderException(f"Failed to generate blob URL: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        """Upload a local file to blob storage."""
        await self._ensure_initialized()

        try:
            logger.info(f"Uploading file: {file_path}")
            logger.info(f"Container: {container}, Blob: {blob_name}")

            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
            await client.upload_blob(data, overwrite=True)

            logger.info(f"Successfully uploaded file: {file_path}")
            url = f"{self.service_client.url}/{container}/{blob_name}"
            await client.close()
            return url
        except Exception as e:
            logger.exception(f"Error uploading file {file_path}: {e}")
            raise ProviderException(f"Error uploading file {file_path}: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        """Upload base64-encoded data to blob storage."""
        await self._ensure_initialized()

        try:
            logger.info(f"Uploading base64 data to Container: {container}, Blob: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            data = base64.b64decode(b64_str)
            await client.upload_blob(data, overwrite=True)

            url = f"{self.service_client.url}/{container}/{blob_name}"
            await client.close()
            return url
        except Exception as e:
            logger.exception(f"Error uploading base64 data: {e}")
            raise ProviderException(f"Error uploading base64 data: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        """Upload a string directly to blob storage."""
        await self._ensure_initialized()

        try:
            logger.info(f"Uploading string content to Container: {container}, Blob: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            await client.upload_blob(content, overwrite=True)

            logger.info(f"Successfully uploaded content to blob: {blob_name}")
            url = f"{self.service_client.url}/{container}/{blob_name}"
            await client.close()
            return url
        except Exception as e:
            logger.exception(f"Error uploading string content: {e}")
            raise ProviderException(f"Error uploading string content: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        """Download a blob to a local file path."""
        await self._ensure_initialized()

        try:
            logger.info(f"Downloading blob {blob_name} to {download_path}")
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)

            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            stream = await client.download_blob()
            data = await stream.readall()

            async with aiofiles.open(download_path, "wb") as f:
                await f.write(data)

            await client.close()
            logger.info(f"Successfully downloaded blob to {download_path}")
            return download_path
        except Exception as e:
            logger.exception(f"Error downloading blob: {e}")
            raise ProviderException(f"Error downloading blob: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        """Download a blob from its URL to a local folder."""
        try:
            logger.info(f"Downloading blob from URL: {blob_url}")
            parsed = urlparse(blob_url)
            container, blob_name = parsed.path.lstrip("/").split("/", 1)
            local_path = os.path.join(save_folder, blob_name)
            return await self.download_to_file(container, blob_name, local_path)
        except Exception as e:
            logger.exception(f"Error downloading blob from URL: {e}")
            raise ProviderException(f"Error downloading blob from URL: {e}")

    async def close(self):
        """Close the underlying service client and cleanup."""
        if self.service_client:
            logger.info("Closing Azure Blob Storage client")
            await self.service_client.close()
            if self.credential:
                await self.credential.close()
