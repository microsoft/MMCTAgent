import asyncio
import sys

# Mock dependent modules since we can't easily run full ingestion here without data
# We'll just instantiate the IngestionPipeline to check header/init logic and run it with a mocked pipeline config if possible,
# or just check that we can call run() and get the progress bar.
# Actually, running the full run() might be hard without valid video files.
# Let's rely on checking the __init__ logic and maybe mocking PipelineRunner logic if we can.
# Better yet, let's just inspect the logging configuration results.

from mmct.utils.logging_config import log_manager
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import IngestionPipeline


# Mock IngestionProviderConfig
class MockProvider:
    pass


async def main():
    print("--- Test Verbosity 0 (Default) ---")
    try:
        ip0 = IngestionPipeline(
            video_path="dummy.mp4",
            video_id="dummy_id",
            provider=MockProvider(),
            language="en",
            verbosity=0,
        )
        print("IngestionPipeline instantiated for v=0. Logger level should be WARNING.")
        # We can't easily check internal logger level property exposed by loguru, but we can verify no INFO logs appeared above.
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Test Verbosity 1 (Info) ---")
    try:
        ip1 = IngestionPipeline(
            video_path="dummy.mp4",
            video_id="dummy_id",
            provider=MockProvider(),
            language="en",
            verbosity=1,
        )
        print(
            "IngestionPipeline instantiated for v=1. INFO log 'Successfully retrieved...' should appear above."
        )
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Test Verbosity 2 (Debug) ---")
    try:
        ip2 = IngestionPipeline(
            video_path="dummy.mp4",
            video_id="dummy_id",
            provider=MockProvider(),
            language="en",
            verbosity=2,
        )
        print(
            "IngestionPipeline instantiated for v=2. INFO log should appear above (as DEBUG includes INFO)."
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
