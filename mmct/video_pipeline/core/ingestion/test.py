import asyncio
import os
from dotenv import load_dotenv
from azure.search.documents.aio import SearchClient
from azure.identity import AzureCliCredential, DefaultAzureCredential

load_dotenv()

def _get_credential():
    try:
        cli_credential = AzureCliCredential()
        cli_credential.get_token("https://search.azure.com/.default")
        return cli_credential
    except Exception:
        return DefaultAzureCredential()

async def main():
    token_provider = _get_credential()

    index_client = SearchClient(
        endpoint=os.getenv("SEARCH_ENDPOINT"),
        index_name="keyframes-nptel",
        credential=token_provider
    )
    video_id = "951befc2333d28d512f1226e91f224fefd1e0a895c96aaef982571ec336fdfe0"
    time_filter = f"timestamp_seconds ge {1} and timestamp_seconds le {20}"
    video_filter = f"video_id eq '{video_id}'"
    combined_filter = f"{time_filter} and {video_filter}"
    results = await index_client.search(
        search_text="*",
        filter=combined_filter,
        order_by=["created_at asc"]
    )

    async for result in results:
        print(result['keyframe_filename'])
    await index_client.close()

if __name__ == "__main__":
    asyncio.run(main())

