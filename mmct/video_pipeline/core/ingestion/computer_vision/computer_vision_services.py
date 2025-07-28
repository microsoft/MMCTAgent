import os
import aiohttp
import asyncio
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

class ComputerVisionService:
    def __init__(self, video_id):
        self.AZURECV_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT", None)
        self.video_id  = video_id
        if self.AZURECV_ENDPOINT is None:
            raise Exception("AzureCV endpoint is missing!")

        self.AZURECV_API_VERSION = os.getenv("AZURECV_API_VERSION", None)
        if self.AZURECV_API_VERSION is None:
            raise Exception("AzureCV API version is missing!")

        if os.environ.get("MANAGED_IDENTITY", None) is None:
            raise Exception(
                "MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
            )
        self.azure_managed_identity = os.environ.get(
            "MANAGED_IDENTITY", ""
        ).upper()
        if self.azure_managed_identity == "TRUE":
            # Use Azure CLI credential if available, fallback to DefaultAzureCredential
            token_provider = self._get_credential()
                
            token = token_provider.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
        else:
            token = os.getenv("COMPUTER_VISION_KEY")

        self.headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }

        self.url =  (
            f"{self.AZURECV_ENDPOINT}/computervision/retrieval/indexes/{self.video_id[:16].replace('_','').replace('-','')}"
        )
    
    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            from azure.identity import AzureCliCredential
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://cognitiveservices.azure.com/.default")
            return cli_credential
        except Exception:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        
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
    # Example usage - replace with your actual values
    blob_url = "https://example.blob.core.windows.net/container/video.mp4"
    azurecv = ComputerVisionService(video_id="example_video_id")
    resp = asyncio.run(azurecv.add_video_to_index(blob_url=blob_url))
    print(resp)