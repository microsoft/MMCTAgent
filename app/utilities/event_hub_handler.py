from dotenv import load_dotenv, find_dotenv
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential, AzureCliCredential as AsyncAzureCliCredential
from azure.eventhub.aio import EventHubProducerClient, EventHubConsumerClient
from azure.eventhub import EventData
from loguru import logger
from azure.eventhub.exceptions import ConnectionLostError, AuthenticationError
from typing import Annotated, Dict
import os
import json

load_dotenv(find_dotenv(), override=True)

class EventHubHandler:
    def __init__(self, hub_name: Annotated[str, "Event Hub Name"]):
        try:
            if not hub_name:
                raise Exception(
                    "Invalid event hub name. Event hub name cannot be empty!"
                )

            # Use Azure CLI credential first, then fallback to DefaultAzureCredential
            self.credential = self._get_credential()
                
            self.host_name = os.getenv("EVENT_HUB_HOSTNAME")
            self.hub_name = hub_name
            if self.host_name is None or self.hub_name is None:
                raise Exception("Event Hub Host Name or Hub Name is missing!")

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
            )
        except Exception as e:
            logger.exception(
                f"Exception occured while creating producer and consumer client: {e}"
            )
            raise

    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            # Try Azure CLI credential first
            credential = AsyncAzureCliCredential()
            logger.info("Using Azure CLI credential for Event Hub")
            return credential
        except Exception as e:
            logger.warning(f"Azure CLI credential failed: {e}")
            # Fall back to DefaultAzureCredential
            credential = AsyncDefaultAzureCredential()
            logger.info("Using DefaultAzureCredential for Event Hub")
            return credential

    async def produce_event(
        self,
        payload: Annotated[
            Dict, "JSON payload that needs to be published to the event hub"
        ],
    ) -> Annotated[
        Dict,
        "JSON containing the message if the payload was successfully produced to event hub or not",
    ]:
        try:
            await self.producer.send_event(EventData(json.dumps(payload)))
            return {"success": True, "message": "Event sent!"}
        except (ConnectionLostError, AuthenticationError) as e:
            return {"success": False, "message": str(e)}
        except Exception as e:
            return {"success": False, "message": str(e)}
