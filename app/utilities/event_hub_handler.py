from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore
from azure.storage.blob.aio import ContainerClient
from azure.identity.aio import DefaultAzureCredential, AzureCliCredential
from azure.eventhub.aio import EventHubProducerClient, EventHubConsumerClient
from azure.eventhub import EventData
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from typing import Annotated, Dict
import asyncio, json, os

load_dotenv(find_dotenv(), override=True)

class EventHubHandler:
    def __init__(self, hub_name: Annotated[str, "Event Hub Name"]):
        try:
            if not hub_name:
                raise Exception("Invalid event hub name. Cannot be empty!")

            self.credential = self._get_credential()
            self.host_name = os.getenv("EVENT_HUB_HOSTNAME")
            self.hub_name = hub_name

            # Checkpoint store setup (MUST HAVE)
            storage_account_url = os.getenv("BLOB_ACCOUNT_URL")  # example: https://mystorageaccount.blob.core.windows.net
            container_name = os.getenv("CHECKPOINT_CONTAINER_NAME", "eventhub-checkpoints")

            self.checkpoint_store = BlobCheckpointStore(blob_account_url = storage_account_url, container_name = container_name, credential=self.credential)

            self.producer = EventHubProducerClient(
                fully_qualified_namespace=self.host_name,
                eventhub_name=self.hub_name,
                credential=self.credential,
            )

            self.consumer = EventHubConsumerClient(
                fully_qualified_namespace=self.host_name,
                eventhub_name=self.hub_name,
                consumer_group="$Default",
                credential=self.credential,
                checkpoint_store=self.checkpoint_store,  # ✅ This enables persistent checkpointing!
            )
        except Exception as e:
            logger.exception(f"Exception while creating Event Hub handler: {e}")
            raise

    def _get_credential(self):
        try:
            credential = AzureCliCredential()
            asyncio.run(credential.get_token("https://eventhubs.azure.net/.default"))
            logger.info("Using Azure CLI credential")
            return credential
        except Exception as e:
            logger.warning(f"Azure CLI failed: {e}")
            credential = DefaultAzureCredential()
            logger.info("Using DefaultAzureCredential fallback")
            return credential

    async def produce_event(self, payload: Dict) -> Dict:
        try:
            await self.producer.send_event(EventData(json.dumps(payload)))
            return {"success": True, "message": "Event sent!"}
        except Exception as e:
            return {"success": False, "message": str(e)}
