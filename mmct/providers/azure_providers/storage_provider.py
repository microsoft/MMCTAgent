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

    async def load_blob_to_memory(self, container: str, blob_name: str) -> bytes:
        """Load a blob's content into memory as bytes."""
        self._ensure_initialized()

        client = None
        try:
            logger.info(f"Loading blob {blob_name} from container {container} into memory")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            stream = await client.download_blob()
            data = await stream.readall()
            logger.info(f"Successfully loaded blob {blob_name} into memory")
            return data
        except Exception as e:
            logger.exception(f"Error loading blob {blob_name} into memory: {e}")
            raise ProviderException(f"Error loading blob {blob_name} into memory: {e}")
        finally:
            if client:
                await client.close()

    async def get_file_url(self, file_name: str, **kwargs) -> str:
        """
        Generate a URL for a blob that doesn't yet exist in storage.

        Args:
            container: The container name where the blob would be stored
            blob_name: The name for the potential future blob

        Returns:
            str: The unencoded URL that the blob would have when uploaded
        """
        self._ensure_initialized()

        try:
            container = kwargs.pop('container')
            if not container:
                raise ProviderException("Container name is required while utilizing azure provider")
            # Use service client URL if available, otherwise fall back to config
            if self.service_client:
                url = f"{self.service_client.url}/{container}/{file_name}"
            else:
                account_url = self.config.get("account_url") or os.getenv("STORAGE_ACCOUNT_URL")
                if not account_url:
                    raise ConfigurationException("Azure Storage account_url is required")
                url = f"{account_url.rstrip('/')}/{container}/{file_name}"

            logger.info(f"Generated blob URL: {url}")
            return url
        except ConfigurationException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate blob URL: {e}")
            raise ProviderException(f"Failed to generate blob URL: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def save_file(self, file_name: str, file_path: str, **kwargs) -> str:
        """Upload a local file to blob storage."""
        self._ensure_initialized()

        client = None
        try:
            container = kwargs.pop('container')
            if not container:
                raise ProviderException("Container name is required while utilizing azure provider")
            logger.info(f"Uploading file: {file_path}")

            client = self.service_client.get_blob_client(container=container, blob=file_name)
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
            await client.upload_blob(data, overwrite=True)

            logger.info(f"Successfully uploaded file: {file_path}")
            url = f"{self.service_client.url}/{container}/{file_name}"
            return url
        except Exception as e:
            logger.exception(f"Error uploading file {file_path}: {e}")
            raise ProviderException(f"Error uploading file {file_path}: {e}")
        finally:
            if client:
                await client.close()

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def save_base64(self, file_name: str, b64_str: str, **kwargs) -> str:
        """Upload base64-encoded data to blob storage."""
        self._ensure_initialized()

        client = None
        try:
            container = kwargs.pop('container')
            logger.info(f"Uploading base64 data to Container: {container}, Blob: {file_name}")
            client = self.service_client.get_blob_client(container=container, blob=file_name)
            data = base64.b64decode(b64_str)
            await client.upload_blob(data, overwrite=True)

            url = f"{self.service_client.url}/{container}/{file_name}"
            return url
        except Exception as e:
            logger.exception(f"Error uploading base64 data: {e}")
            raise ProviderException(f"Error uploading base64 data: {e}")
        finally:
            if client:
                await client.close()

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def save_string(self, file_name: str, content: str, **kwargs) -> str:
        """Upload a string directly to blob storage."""
        self._ensure_initialized()

        client = None
        try:
            container = kwargs.pop('container')
            logger.info(f"Uploading string content to Container: {container}, Blob: {file_name}")
            client = self.service_client.get_blob_client(container=container, blob=file_name)
            await client.upload_blob(content, overwrite=True)

            logger.info(f"Successfully uploaded content to blob: {file_name}")
            url = f"{self.service_client.url}/{container}/{file_name}"
            return url
        except Exception as e:
            logger.exception(f"Error uploading string content: {e}")
            raise ProviderException(f"Error uploading string content: {e}")
        finally:
            if client:
                await client.close()

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_to_file(self, file_name: str, download_path: str, **kwargs) -> str:
        """Download a blob to a local file path."""
        self._ensure_initialized()

        client = None
        try:
            container = kwargs.pop('container')
            logger.info(f"Downloading blob {file_name} to {download_path}")
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)

            client = self.service_client.get_blob_client(container=container, blob=file_name)
            stream = await client.download_blob()
            data = await stream.readall()

            async with aiofiles.open(download_path, "wb") as f:
                await f.write(data)

            logger.info(f"Successfully downloaded blob to {download_path}")
            return download_path
        except Exception as e:
            logger.exception(f"Error downloading blob: {e}")
            raise ProviderException(f"Error downloading blob: {e}")
        finally:
            if client:
                await client.close()

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def download_from_url(self, file_url: str, save_folder: str) -> str:
        """Download a blob from its URL to a local folder."""
        try:
            logger.info(f"Downloading blob from URL: {file_url}")
            parsed = urlparse(file_url)
            container, blob_name = parsed.path.lstrip("/").split("/", 1)
            local_path = os.path.join(save_folder, blob_name)
            return await self.download_to_file(container=container,file_name=blob_name,download_path=local_path)
        except Exception as e:
            logger.exception(f"Error downloading blob from URL: {e}")
            raise ProviderException(f"Error downloading blob from URL: {e}")

    async def close(self):
        """Close the underlying service client and cleanup."""
        if self.service_client:
            logger.info("Closing Azure Blob Storage client")
            await self.service_client.close()
