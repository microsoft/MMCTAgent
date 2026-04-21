"""Azure Blob Storage provider implementation.

This module provides the AzureStorageProvider class, which implements the 
BaseStorageProvider interface for managing file assets in Azure Blob Storage.
"""

import os
import aiofiles
from azure.storage.blob.aio import BlobServiceClient
from loguru import logger
from typing import Dict, Any, Union, Optional
from mmct.providers.base import BaseStorageProvider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import ResourceExistsError
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException, ConfigurationException


class AzureStorageProvider(BaseStorageProvider):
    """Azure Blob Storage provider implementation.

    This provider handles authentication and client management for storing and 
    retrieving video-related assets (e.g., keyframes) in Azure Blob Storage. 
    It supports both SAS/Connection String and AAD-based authentication.

    Attributes:
        credentials (Union[AzureKeyCredential, AsyncTokenCredential], optional): 
            Identity-based credentials.
        blob_connection_string (str, optional): Connection string for the storage account.
        storage_account_name (str): The name of the Azure Storage account.
        keyframe_container_name (str): The default container for keyframes.
        storage_account_url (str): The base URL for the blob service.
        service_client (BlobServiceClient): The initialized async storage client.
        _verified_containers (set): Containers already confirmed to exist.
    """

    def __init__(
        self,
        storage_account_name: str,
        keyframe_container_name: str,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        blob_connection_string: Optional[str] = None,
    ):
        """Initializes the AzureStorageProvider.

        Args:
            storage_account_name: Azure Storage account name.
            keyframe_container_name: Default container name for keyframe storage.
            credentials: Azure credentials for token-based authentication.
                Mutually exclusive with `blob_connection_string`.
            blob_connection_string: Connection string for the storage account.
                Mutually exclusive with `credentials`.

        Raises:
            ConfigurationException: If required fields are missing or if both
                `credentials` and `blob_connection_string` are provided.
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
        self.storage_account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
        self._verified_containers: set = set()
        self.service_client = self._initialize()

    def _initialize(self) -> BlobServiceClient:
        """Initializes the BlobServiceClient with either credentials or connection string.

        Returns:
            BlobServiceClient: The initialized asynchronous service client.

        Raises:
            ProviderException: If client initialization fails.
        """
        try:
            if self.credentials is not None:
                # Use credentials with token-based authentication
                client = BlobServiceClient(
                    account_url=self.storage_account_url,
                    credential=self.credentials,
                )
                logger.info("Successfully initialized Azure Blob Storage client with credentials")
            else:
                # Use connection string authentication
                client = BlobServiceClient.from_connection_string(
                    conn_str=self.blob_connection_string
                )
                logger.info(
                    "Successfully initialized Azure Blob Storage client with connection string"
                )

            return client
        except Exception as e:
            logger.exception(f"Failed to initialize Azure Blob Storage client: {e}")
            raise ProviderException(f"Failed to initialize Azure Blob Storage client: {e}")

    async def load_file_to_memory(self, folder: str, file_name: str) -> bytes:
        """Downloads a blob's content and loads it into memory as bytes.

        Args:
            folder: The name of the container.
            file_name: The name/path of the blob within the container.

        Returns:
            bytes: The raw data of the file.

        Raises:
            ProviderException: If the download fails.
        """

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

    async def get_file_url(self, file_name: str, **kwargs: Any) -> str:
        """Generates a static URL for a blob.

        Args:
            file_name: The name/path of the blob.
            **kwargs: Reserved for future parameter expansion.

        Returns:
            str: The constructed URL for the file.

        Raises:
            ProviderException: If the URL cannot be constructed.
        """
        try:
            folder_name = self.keyframe_container_name
            # Construct the URL based on the storage account base URL
            base_url = self.storage_account_url.rstrip("/")
            url = f"{base_url}/{folder_name}/{file_name}"

            logger.info(f"Generated file URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate URL: {e}")
            raise ProviderException(f"Failed to generate URL: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_file(self, file_name: str, src_file_path: str, **kwargs: Any) -> str:
        """Uploads a local file to a blob in the specified container.

        Args:
            file_name: The destination name/path in storage.
            src_file_path: The local filesystem path to the file.
            **kwargs: Expected to contain 'folder_name' for the container.

        Returns:
            str: The URL of the uploaded file.

        Raises:
            ProviderException: If the upload fails.
        """
        client = None
        try:
            logger.debug(f"Uploading file: {src_file_path}")
            folder_name = kwargs.pop("folder_name", self.keyframe_container_name)

            # Only check container existence once per container
            if folder_name not in self._verified_containers:
                container_client = self.service_client.get_container_client(folder_name)
                if not await container_client.exists():
                    logger.info(f"Container {folder_name} does not exist. Creating it...")
                    try:
                        await container_client.create_container()
                        logger.info(f"Successfully created container: {folder_name}")
                    except ResourceExistsError:
                        logger.info(f"Container {folder_name} already exists.")
                await container_client.close()
                self._verified_containers.add(folder_name)

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

    async def check_health(self) -> Dict[str, Any]:
        """Verify Azure Blob Storage connectivity.

        Lists the default container to confirm the service client can
        authenticate and reach the storage account.
        """
        try:
            container = self.service_client.get_container_client(
                self.keyframe_container_name
            )
            exists = await container.exists()
            await container.close()
            return {
                "status": "ok",
                "container": self.keyframe_container_name,
                "container_exists": exists,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def close(self) -> None:
        """Closes the Azure Blob Storage service client."""
        if self.service_client:
            logger.info("Closing Azure Blob Storage client")
            await self.service_client.close()
