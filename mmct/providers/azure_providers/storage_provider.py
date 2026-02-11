import os
import base64
from pathlib import Path
import aiofiles
from urllib.parse import urlparse
from azure.storage.blob.aio import BlobServiceClient
from loguru import logger
from typing import Dict, Any, Union, Optional
from mmct.providers.base import BaseStorageProvider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import ResourceExistsError
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException


class AzureStorageProvider(BaseStorageProvider):
    """Azure Blob Storage provider implementation."""

    def __init__(
        self,
        storage_account_name: str,
        keyframe_container_name: str,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        blob_connection_string: Optional[str] = None,
    ):
        """
        Initialize Azure Storage Provider.

        Args:
            storage_account_name: Azure Storage account name
            credentials: Azure credentials for token-based authentication (mutually exclusive with blob_connection_string)
            blob_connection_string: Connection string for connection string-based authentication (mutually exclusive with credentials)

        Raises:
            ConfigurationException: If storage_account_name is missing, or if neither credentials nor
                                   blob_connection_string is provided, or if both are provided
        """
        if not storage_account_name:
            raise ConfigurationException("Storage account name is required!")

        if not keyframe_container_name:
            raise ConfigurationException("Keyframe container name is required!")

        # Validate that exactly one of credentials or blob_connection_string is provided
        if credentials is None and blob_connection_string is None:
            raise ConfigurationException(
                "Either credentials or blob_connection_string must be provided!"
            )

        if credentials is not None and blob_connection_string is not None:
            raise ConfigurationException(
                "Only one of credentials or blob_connection_string should be provided, not both!"
            )

        self.credentials = credentials
        self.blob_connection_string = blob_connection_string
        self.storage_account_name = storage_account_name
        self.keyframe_container_name = keyframe_container_name
        self.storage_account_url = f"https://{self.storage_account_name}.blob.core.windows.net/"
        self._keyframe_container_ensured = False
        self.service_client = self._initialize()

    def _initialize(self):
        """Initialize BlobServiceClient with either credentials or connection string."""
        try:
            if self.credentials is not None:
                # Use credentials with token-based authentication
                self.service_client = BlobServiceClient(
                    account_url=self.storage_account_url,
                    credential=self.credentials,
                )
                logger.info("Successfully initialized Azure Blob Storage client with credentials")
            else:
                # Use connection string authentication
                self.service_client = BlobServiceClient.from_connection_string(
                    conn_str=self.blob_connection_string
                )
                logger.info(
                    "Successfully initialized Azure Blob Storage client with connection string"
                )

            return self.service_client
        except Exception as e:
            logger.exception(f"Failed to initialize Azure Blob Storage client: {e}")
            raise ProviderException(f"Failed to initialize Azure Blob Storage client: {e}")

    async def _ensure_keyframe_container_exists(self):
        """
        Ensure the keyframe container exists in Azure Blob Storage.
        Creates it if it doesn't exist. Only runs once per provider instance.
        """
        if self._keyframe_container_ensured:
            return

        container_client = None
        try:
            container_client = self.service_client.get_container_client(
                self.keyframe_container_name
            )
            # Attempt to get container properties to check existence
            await container_client.get_container_properties()
            logger.info(f"Keyframe container '{self.keyframe_container_name}' already exists.")
        except Exception as ex:
            if "ContainerNotFound" in str(ex):
                try:
                    await container_client.create_container()
                    logger.info(
                        f"Keyframe container '{self.keyframe_container_name}' created successfully."
                    )
                except ResourceExistsError:
                    # Race condition safety — another process may have created it
                    logger.info(
                        f"Keyframe container '{self.keyframe_container_name}' was created by another process."
                    )
            else:
                logger.exception(
                    f"Failed to verify keyframe container '{self.keyframe_container_name}': {ex}"
                )
                raise ProviderException(
                    f"Failed to verify keyframe container '{self.keyframe_container_name}': {ex}"
                )
        finally:
            if container_client:
                await container_client.close()

        self._keyframe_container_ensured = True

    async def load_file_to_memory(self, folder: str, file_name: str) -> bytes:
        """Load a file's content into memory as bytes."""
        await self._ensure_keyframe_container_exists()

        client = None
        try:
            logger.info(f"Loading file {file_name} from container {folder} into memory")
            client = self.service_client.get_blob_client(container=folder, blob=file_name)
            stream = await client.download_blob()
            data = await stream.readall()
            logger.info(f"Successfully loaded file {file_name} into memory")
            return data
        except Exception as e:
            logger.exception(f"Error loading file {file_name} into memory: {e}")
            raise ProviderException(f"Error loading file {file_name} into memory: {e}")
        finally:
            if client:
                await client.close()

    async def get_file_url(self, file_name: str, **kwargs) -> str:
        """
        Generate a URL for a file that doesn't yet exist in storage.
        """
        await self._ensure_keyframe_container_exists()
        try:
            folder_name = self.keyframe_container_name
            # Use service client URL if available, otherwise fall back to config
            if self.service_client:
                # Remove trailing slash to avoid double slashes in URL
                base_url = self.service_client.url.rstrip("/")
                url = f"{base_url}/{folder_name}/{file_name}"
            else:
                if not self.storage_account_url:
                    raise ConfigurationException("Azure Storage account_url is required")
                url = f"{self.storage_account_url.rstrip('/')}/{folder_name}/{file_name}"

            logger.info(f"Generated file URL: {url}")
            return url
        except ConfigurationException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate URL: {e}")
            raise ProviderException(f"Failed to generate URL: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_file(self, file_name: str, src_file_path: str, **kwargs) -> str:
        """Upload a local file to blob storage."""
        await self._ensure_keyframe_container_exists()
        client = None
        container_client = None
        try:
            logger.debug(f"Uploading file: {src_file_path}")
            folder_name = kwargs.pop("folder_name")

            # Check if container exists, create if it doesn't
            container_client = self.service_client.get_container_client(folder_name)
            if not await container_client.exists():
                logger.info(f"Container {folder_name} does not exist. Creating it...")
                try:
                    await container_client.create_container()
                    logger.info(f"Successfully created container: {folder_name}")
                except ResourceExistsError:
                    logger.info(f"Container {folder_name} already exists.")

            client = self.service_client.get_blob_client(container=folder_name, blob=file_name)
            async with aiofiles.open(src_file_path, "rb") as f:
                data = await f.read()
            await client.upload_blob(data, overwrite=True)

            logger.debug(f"Successfully uploaded file: {src_file_path}")
            url = f"{self.storage_account_url}/{folder_name}/{file_name}"
            return url
        except Exception as e:
            logger.exception(f"Error uploading file {src_file_path}: {e}")
            raise ProviderException(f"Error uploading file {src_file_path}: {e}")
        finally:
            if client:
                await client.close()
            if container_client:
                await container_client.close()

    async def close(self):
        """Close the underlying service client and cleanup."""
        if self.service_client:
            logger.info("Closing Azure Blob Storage client")
            await self.service_client.close()
