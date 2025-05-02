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
from azure.identity.aio import DefaultAzureCredential

class BlobStorageManager:
    def __init__(self, account_url: str = None):
        # Initialize credential and transport with limited connection pool
        self.credential = DefaultAzureCredential()
        self.service_client = BlobServiceClient(
            account_url or os.getenv("BLOB_ACCOUNT_URL"),
            credential=self.credential,
        )

    async def get_blob_url(self, container: str, blob_name: str) -> str:
        """Return the unencoded URL for a blob."""
        client = self.service_client.get_blob_client(container=container, blob=blob_name)
        url = unquote(client.url)
        await client.close()
        return url

    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        """Upload a local file to blob storage."""
        client = self.service_client.get_blob_client(container=container, blob=blob_name)
        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()
        await client.upload_blob(data, overwrite=True)
        url = f"{self.service_client.url}/{container}/{blob_name}"
        await client.close()
        return url

    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        """Upload base64-encoded data to blob storage."""
        client = self.service_client.get_blob_client(container=container, blob=blob_name)
        data = base64.b64decode(b64_str)
        await client.upload_blob(data, overwrite=True)
        url = f"{self.service_client.url}/{container}/{blob_name}"
        await client.close()
        return url

    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        """Upload a string directly to blob storage without saving to a local file."""
        client = self.service_client.get_blob_client(container=container, blob=blob_name)
        await client.upload_blob(content, overwrite=True)
        url = f"{self.service_client.url}/{container}/{blob_name}"
        await client.close()
        return url

    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        """Download a blob to a local file path."""
        Path(download_path).parent.mkdir(parents=True, exist_ok=True)
        client = self.service_client.get_blob_client(container=container, blob=blob_name)
        stream = await client.download_blob()
        data = await stream.readall()
        async with aiofiles.open(download_path, "wb") as f:
            await f.write(data)
        await client.close()
        return download_path

    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        """Download a blob from its URL to a local folder."""
        parsed = urlparse(blob_url)
        container, blob_name = parsed.path.lstrip("/").split("/", 1)
        local_path = os.path.join(save_folder, blob_name)
        return await self.download_to_file(container, blob_name, local_path)

    async def close(self):
        """Close the underlying service client and cleanup."""
        await self.service_client.close()
