"""
This tool allows you to issue a search query over the video and return the timestamps of the 
top 3 relevant scenes.
"""

# Importing Libraries
import os
import requests
from typing_extensions import Annotated
from azure.identity import DefaultAzureCredential
from datetime import datetime


async def query_frames_azure_computer_vision(frames_query:Annotated[str,"search query over the frames"], video_id:Annotated[str,'video id'])->str:
        # Getting requred environment variables
        AZURECV_ENDPOINT = os.environ.get("COMPUTER_VISION_ENDPOINT")
        BLOB_MANAGED_IDENTITY = os.environ.get("BLOB_MANAGED_IDENTITY") == "True"
        MANAGED_IDENTITY = os.environ.get("MANAGED_IDENTITY") == "True"
        AZURE_OPENAI_MANAGED_IDENTITY = os.environ.get("MANAGED_IDENTITY") == "True"
        AZURE_MODERATION_MANAGED_IDENTITY = os.environ.get("AZURE_MODERATION_MANAGED_IDENTITY") == "True"
        AZURECV_API_VERSION = os.environ.get("AZURECV_API_VERSION")


        if BLOB_MANAGED_IDENTITY or MANAGED_IDENTITY or AZURE_OPENAI_MANAGED_IDENTITY or AZURE_MODERATION_MANAGED_IDENTITY:
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
        if MANAGED_IDENTITY:
            azurecv_headers = {
                "Authorization": "Bearer "+token.token,
                "Content-Type": "application/json"
            }
        else:
            COMPUTER_VISION_KEY = os.environ.get("COMPUTER_VISION_KEY")
            azurecv_headers = {
                "Ocp-Apim-Subscription-Key": COMPUTER_VISION_KEY,
                "Content-Type": "application/json"
            }
        search_url = f"{AZURECV_ENDPOINT}/computervision/retrieval/indexes/{video_id.replace('-','').replace('_','')}:queryByText?api-version={AZURECV_API_VERSION}"
        search_payload = {
            "queryText": frames_query,
            "moderation": False,
            "top": 3
        }
        search_response = requests.post(search_url, headers=azurecv_headers, json=search_payload)
        response_data = search_response.json()

        results = []
        relevant_timestamps = []
        for item in response_data.get("value", [])[:3]:
            best_time = item["best"]
            try:
                # Splitting the time and microsecond parts
                if '.' in best_time:
                    time_part, microsecond_part = best_time.split('.')
                    # Truncating or padding the microsecond part to six digits
                    microsecond_part = microsecond_part[:6].ljust(6, '0')
                    # Reconstructing the best_time string with the adjusted microsecond part
                    adjusted_best_time = f"{time_part}.{microsecond_part}"
                else:
                    adjusted_best_time = best_time
                # First, try parsing with microseconds
                formatted_relevant_timestamp = datetime.strptime(adjusted_best_time, "%H:%M:%S.%f").strftime("%H:%M:%S")
            except ValueError:
                # If there's a ValueError, it means microseconds weren't present, so parse without them
                formatted_relevant_timestamp = datetime.strptime(best_time, "%H:%M:%S").strftime("%H:%M:%S")
            relevant_timestamps.append(formatted_relevant_timestamp)
            results.append({
                "best": item["best"],
                "start": item["start"],
                "end": item["end"],
                "relevance": item["relevance"]
            })

        return ", ".join(relevant_timestamps)