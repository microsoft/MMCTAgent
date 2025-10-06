"""
Azure Blob Storage management for keyframe uploads.
"""

import os
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceExistsError
from ..core import BlobStorageConfig, FrameEmbedding
from ..auth import get_storage_credential

logger = logging.getLogger(__name__)


class BlobStorageManager:
    """Manages blob storage operations for keyframes."""

    def __init__(
        self,
        storage_account_url: Optional[str] = None,
        connection_string: Optional[str] = None,
        config: Optional[BlobStorageConfig] = None
    ):
        """
        Initialize the blob storage manager.

        Authentication priority:
        1. Azure CLI credentials
        2. DefaultAzureCredential (Managed Identity, etc.)
        3. Connection string (if provided)

        Args:
            storage_account_url: Azure Storage account URL (e.g., https://myaccount.blob.core.windows.net)
            connection_string: Optional connection string (fallback)
            config: Optional BlobStorageConfig
        """
        self.config = config or BlobStorageConfig()
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.storage_account_url = storage_account_url or os.getenv("AZURE_STORAGE_ACCOUNT_URL")

        # Override container name from environment if set
        env_container_name = os.getenv("KEYFRAME_CONTAINER_NAME")
        if env_container_name:
            self.config.container_name = env_container_name

        # Initialize blob service client with prioritized authentication
        if self.connection_string:
            logger.info("Initializing Blob Storage with connection string")
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        elif self.storage_account_url:
            logger.info("Initializing Blob Storage with Azure credentials (CLI -> Default)")
            credential = get_storage_credential()
            self.blob_service_client = BlobServiceClient(
                account_url=self.storage_account_url,
                credential=credential
            )
        else:
            raise ValueError(
                "Azure Storage configuration required. Provide either:\n"
                "1. storage_account_url (recommended - uses Azure CLI/Managed Identity)\n"
                "2. connection_string (fallback)\n"
                "Or set AZURE_STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING environment variable"
            )

        self.container_name = self.config.container_name
        logger.info(f"Using blob container: {self.container_name}")

        # Ensure container exists
        self._create_container_if_not_exists()

    def _create_container_if_not_exists(self):
        """Create the container if it doesn't exist."""
        try:
            self.blob_service_client.create_container(self.container_name)
            logger.info(f"Created container: {self.container_name}")
        except ResourceExistsError:
            logger.info(f"Container already exists: {self.container_name}")
        except Exception as e:
            logger.error(f"Error creating container: {e}")
            raise

    async def upload_keyframe(
        self,
        file_path: str,
        blob_name: str,
        overwrite: bool = True
    ) -> str:
        """
        Upload a single keyframe to blob storage.

        Args:
            file_path: Local path to the keyframe file
            blob_name: Name for the blob in storage
            overwrite: Whether to overwrite existing blob

        Returns:
            Blob URL
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)

            blob_url = blob_client.url
            logger.info(f"Uploaded keyframe: {blob_name}")

            # Generate SAS URL if configured
            if self.config.generate_sas_token:
                blob_url = self._generate_sas_url(blob_name)

            return blob_url

        except Exception as e:
            logger.error(f"Failed to upload keyframe {file_path}: {e}")
            raise

    async def upload_keyframes_batch(
        self,
        frame_embeddings: List[FrameEmbedding],
        video_id: str
    ) -> List[FrameEmbedding]:
        """
        Upload a batch of keyframes to blob storage.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Video ID for organizing blobs

        Returns:
            Updated list of FrameEmbedding objects with blob_url set
        """
        try:
            logger.info(f"Uploading {len(frame_embeddings)} keyframes for video {video_id}")

            # Import here to avoid circular dependency
            from ..core.utils import get_media_folder

            media_folder = await get_media_folder()
            keyframes_dir = os.path.join(media_folder, "keyframes", video_id)

            updated_embeddings = []

            for i, frame_embedding in enumerate(frame_embeddings):
                # Construct the file path from frame metadata
                frame_filename = f"{video_id}_{frame_embedding.frame_metadata.frame_number}.jpg"
                file_path = os.path.join(keyframes_dir, frame_filename)
                blob_name = f"{video_id}/{frame_filename}"

                # Upload the frame
                blob_url = await self.upload_keyframe(
                    file_path=file_path,
                    blob_name=blob_name
                )

                # Update the frame embedding with blob URL
                frame_embedding.blob_url = blob_url
                updated_embeddings.append(frame_embedding)

                # Log progress
                if (i + 1) % self.config.upload_batch_size == 0:
                    logger.info(f"Uploaded {i + 1}/{len(frame_embeddings)} keyframes")

            logger.info(f"Successfully uploaded all {len(frame_embeddings)} keyframes")
            return updated_embeddings

        except Exception as e:
            logger.error(f"Failed to upload keyframes batch: {e}")
            raise

    def _generate_sas_url(self, blob_name: str) -> str:
        """
        Generate a SAS URL for a blob.

        Args:
            blob_name: Name of the blob

        Returns:
            SAS URL
        """
        try:
            # Extract account name and key from connection string
            conn_parts = dict(item.split("=", 1) for item in self.connection_string.split(";") if "=" in item)
            account_name = conn_parts.get("AccountName")
            account_key = conn_parts.get("AccountKey")

            if not account_name or not account_key:
                logger.warning("Cannot generate SAS token: missing account credentials")
                return self.get_blob_url(blob_name)

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=self.config.sas_expiry_hours)
            )

            # Construct SAS URL
            blob_url = f"https://{account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            return blob_url

        except Exception as e:
            logger.warning(f"Failed to generate SAS URL: {e}")
            return self.get_blob_url(blob_name)

    def get_blob_url(self, blob_name: str) -> str:
        """
        Get the blob URL without SAS token.

        Args:
            blob_name: Name of the blob

        Returns:
            Blob URL
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        return blob_client.url

    async def delete_keyframe(self, blob_name: str) -> bool:
        """
        Delete a keyframe from blob storage.

        Args:
            blob_name: Name of the blob to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted keyframe: {blob_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete keyframe {blob_name}: {e}")
            return False

    async def list_keyframes(self, video_id: str) -> List[str]:
        """
        List all keyframes for a specific video.

        Args:
            video_id: Video ID to filter blobs

        Returns:
            List of blob names
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with=f"{video_id}/")

            return [blob.name for blob in blob_list]

        except Exception as e:
            logger.error(f"Failed to list keyframes for video {video_id}: {e}")
            return []

    def close(self):
        """Close the blob service client."""
        try:
            self.blob_service_client.close()
            logger.info("Blob storage manager closed")
        except Exception as e:
            logger.warning(f"Error closing blob storage manager: {e}")
