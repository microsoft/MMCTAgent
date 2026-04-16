import os
import pytest
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import IngestionPipeline
from mmct.video_pipeline.core.ingestion.languages import Languages

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_full_ingestion_smoke():
    """
    Perform a full end-to-end ingestion of a sample video.
    
    This test is intended to be run manually or in a controlled environment
    as it makes real provider calls and consumes resources.
    
    Configure the video path via SMOKE_TEST_VIDEO_PATH env var.
    Defaults to examples/bcFvbtZafKM.mp4.
    """
    video_path = os.getenv("SMOKE_TEST_VIDEO_PATH", "examples/bcFvbtZafKM.mp4")
    
    if not os.path.exists(video_path):
        pytest.skip(f"Smoke test video not found at {video_path}. Skipping.")

    import time
    # We use a unique video_id with timestamp for the test to avoid collisions and bypass skipping
    video_id = f"smoke_test_{os.path.basename(video_path).split('.')[0]}_{int(time.time())}"
    
    # Initialize the high-level IngestionPipeline
    # By default, it uses the temporal_graph_ingestion config
    from config.provider_config import get_ingestion_providers
    pipeline = IngestionPipeline(
        video_path=video_path,
        video_id=video_id,
        provider=get_ingestion_providers(),
        language=Languages.ENGLISH_UNITED_STATES,
        save_local_report=True
    )
    
    try:
        print(f"\nStarting E2E Smoke Test with video: {video_path}")
        report = await pipeline.run()
        
        assert report.status == "completed"
        assert len(report.steps) > 5 # Ensure multiple steps were executed
        
        print("\nSmoke Test Summary:")
        print(report.summary())
        
        # Verify that the report file was created
        report_path = os.path.join(pipeline.context.output_dir, f"{video_id}_ip_report.json")
        assert os.path.exists(report_path)
        
    except Exception as e:
        pytest.fail(f"E2E Smoke Test failed: {e}")
