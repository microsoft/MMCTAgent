"""
blob_manager.py
 
BlobStorageManager
------------------
This class centralizes blob operations and optimizes resource usage:
 
Steps taken:
1. Single shared BlobServiceClient with limited HTTP connection pool.
2. Reuse `DefaultAzureCredential` to avoid repeated instantiation.
3. Async methods for upload and download, using `aiofiles`.
4. Explicitly close `BlobClient`s and the service client to release file descriptors.
"""
 
import os
import base64
import asyncio
from pathlib import Path
import aiofiles
from urllib.parse import urlparse, unquote
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential, AzureCliCredential
from loguru import logger

from MMCTAgent.mmct.custom_token_credential import CustomTokenCredential
 
class BlobStorageManager:
    def __init__(self):
        # Private constructor - use create() class method instead
        pass
    
    @classmethod
    async def create(cls, account_url: str = None):
        """Factory method to create BlobStorageManager with async initialization."""
        instance = cls()
        await instance._initialize(account_url)
        return instance
    
    async def _initialize(self, account_url: str = None):
        """Initialize credential and transport with limited connection pool."""
        try:
            # Use Azure CLI credential if available, fallback to DefaultAzureCredential
            self.credential = await self._get_credential()
            self.service_client = BlobServiceClient(
                account_url or os.getenv("BLOB_ACCOUNT_URL"),
                credential=self.credential,
            )
            logger.info("Successfully initialized the blob service client")
        except Exception as e:
            logger.exception(f"Exception occured while creating the blob service client: {e}")
            raise
    
    async def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        return CustomTokenCredential()
        # try:
        #     # Try Azure CLI credential first
        #     credential = AzureCliCredential()
        #     await credential.get_token("https://storage.azure.com/.default")
            
        #     logger.info("Using Azure CLI credential for Blob Storage")
        #     return credential
        # except Exception as e:
        #     logger.warning(f"Azure CLI credential failed: {e}")
        #     # Fall back to DefaultAzureCredential
        #     credential = DefaultAzureCredential()
        #     logger.info("Using DefaultAzureCredential for Blob Storage")
        #     return credential
        
    def get_blob_url(self, container: str, blob_name: str) -> str:
        """
        Generate a URL for a blob that doesn't yet exist in storage.
        This method creates a URL pointing to where the blob would be located
        without actually creating or checking for the blob's existence.
        Useful for pre-generating URLs for future blob uploads.
        Args:
            container: The container name where the blob would be stored
            blob_name: The name for the potential future blob
        Returns:
            str: The unencoded URL that the blob would have when uploaded
        """
        try:
            logger.info(f"Fetching the blob url for the input blob name: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            url = unquote(client.url)
            logger.info(f"Successfully retrieved the blob url: {url}")
            return url
        except Exception as e:
            raise
 
    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        """Upload a local file to blob storage."""
        try:
            logger.info(f"Uploading the local file: {file_path}")
            logger.info(f"Container Name: {container}\nBlob Name: {blob_name}")
            logger.info(f"Storage Account URL: {self.service_client.url}")
            logger.info(f"Credential type: {type(self.credential).__name__}")
            
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
            await client.upload_blob(data, overwrite=True)
            logger.info(f"Successfully upload the local file: {file_path}")
            url = f"{self.service_client.url}/{container}/{blob_name}"
            return url
        except Exception as e:
            logger.exception(f"Error occured while uploading file : {file_path}\nError: {e}")
            logger.error(f"Storage account: {self.service_client.url}")
            logger.error(f"Container: {container}")
            logger.error(f"Blob name: {blob_name}")
            logger.error(f"Credential type: {type(self.credential).__name__}")
            raise
 
    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        """Upload base64-encoded data to blob storage."""
        try:
            logger.info(f"Container Name: {container}\nBlob Name: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            data = base64.b64decode(b64_str)
            await client.upload_blob(data, overwrite=True)
            url = f"{self.service_client.url}/{container}/{blob_name}"
            return url
        except Exception as e:
            raise
    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        """Upload a string directly to blob storage without saving to a local file."""
        try:
            logger.info(f"Uploading the file with string content: {content}")
            logger.info(f"Container Name: {container}\nBlob Name: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            await client.upload_blob(content, overwrite=True)
            logger.info(f"Successfully upload the content to blob name: {blob_name}")
            url = f"{self.service_client.url}/{container}/{blob_name}"
            return url
        except Exception as e:
            logger.exception(f"Exception occured while uploading string content to blob: {e}")
            raise
    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        """Download a blob to a local file path."""
        try:
            logger.info(f"Downloading the blob {blob_name} to the path: {download_path}")
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            stream = await client.download_blob()
            data = await stream.readall()
            async with aiofiles.open(download_path, "wb") as f:
                await f.write(data)
            logger.info(f"Successfully downloaded the blob")
            return download_path
        except Exception as e:
            logger.exception(f"Exception occured while downloading the blob: {e}")
            raise
 
    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        try:
            logger.info(f"Downloading blob from url: {blob_url}")
            """Download a blob from its URL to a local folder."""
            parsed = urlparse(blob_url)
            container, blob_name = parsed.path.lstrip("/").split("/", 1)
            local_path = os.path.join(save_folder, blob_name)
            return await self.download_to_file(container, blob_name, local_path)
        except Exception as e:
            logger.info(f"Exception while downloading the blob from url: {e}")
            raise
 
    async def close(self):
        """Close the underlying service client and cleanup."""
        logger.info("Closing the blob service client!")
        await self.service_client.close()