import os
import aiohttp
import asyncio
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

class AzureComputerVision:
    def __init__(self, video_id):
        self.AZURECV_ENDPOINT = os.getenv("AZURECV_ENDPOINT", None)
        self.video_id  = video_id
        if self.AZURECV_ENDPOINT is None:
            raise Exception("AzureCV endpoint is missing!")

        self.AZURECV_API_VERSION = os.getenv("AZURECV_API_VERSION", None)
        if self.AZURECV_API_VERSION is None:
            raise Exception("AzureCV API version is missing!")

        if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
            raise Exception(
                "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
            )
        self.azure_managed_identity = os.environ.get(
            "AZURE_OPENAI_MANAGED_IDENTITY", ""
        ).upper()
        if self.azure_managed_identity == "TRUE":
            token_provider = DefaultAzureCredential()
            token = token_provider.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
        else:
            token = os.getenv("AZURE_CV_KEY")

        self.headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }

        self.url =  (
            f"{self.AZURECV_ENDPOINT}/computervision/retrieval/indexes/{self.video_id[:16].replace('_','').replace('-','')}"
        )
        
    async def create_index(self):
        request_body = {
        "features": [
            {
                "name": "vision",
                "domain": "generic"
            }
        ],
        }
        async with aiohttp.ClientSession() as session:
            sub_url = f"?api-version={self.AZURECV_API_VERSION}"
            index_create_url = self.url+sub_url
            async with session.put(index_create_url, headers=self.headers, json=request_body) as response:
                return await response.json()
            
    async def add_video_to_index(self, blob_url):
        sub_url = f"/ingestions/my-ingestion?api-version={self.AZURECV_API_VERSION}"
        ingestion_url = self.url+sub_url
        request_body = {
        "moderation": False,
        "videos": [
            {
                "mode": "add",
                "documentUrl": blob_url
            }
        ],
        "documentAuthenticationKind": "managedIdentity" if self.azure_managed_identity=="TRUE" else "key"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.put(ingestion_url, headers=self.headers, json=request_body) as response:
                response_data = await response.json()

            check_url = self.url+f"/ingestions?api-version={self.AZURECV_API_VERSION}&$top=1"

            while True:
                async with session.get(check_url, headers=self.headers) as status_response:
                    status_data = await status_response.json()
                    
                    
                    if status_data['value'][0]['state'] == "Completed":
                        print("Video Addition Completed")
                        return self.video_id
                    elif status_data['value'][0]['state'] == "Running":
                        await asyncio.sleep(10)
                    else:
                        print("Error adding video to index")
                        return None

if __name__=="__main__":
    blob_url = "https://geckostorageaccount.blob.core.windows.net/gecko-videocontainer/009d738d0b4bb8374830a7894c7f3cd4134c6c89a440bfd656cea819c9bf4565.mp4"
    azurecv = AzureComputerVision(video_id="test")
    resp = asyncio.run(azurecv.add_video_to_index(blob_url=blob_url))
    print(resp)