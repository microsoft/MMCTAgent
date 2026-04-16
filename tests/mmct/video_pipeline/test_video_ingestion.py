import pytest
from unittest.mock import MagicMock, patch
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import IngestionPipeline
from mmct.video_pipeline.core.ingestion.languages import Languages

@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.llm_provider = MagicMock()
    provider.embedding_provider = MagicMock()
    provider.transcription_provider = MagicMock()
    provider.storage_provider = MagicMock()
    provider.graph_store_provider = MagicMock()
    return provider

@pytest.mark.unit
def test_ingestion_pipeline_initialization(mock_provider):
    """Test that IngestionPipeline initializes correctly with providers."""
    video_path = "test.mp4"
    video_id = "test_hash"
    
    pipeline = IngestionPipeline(
        video_path=video_path,
        video_id=video_id,
        provider=mock_provider,
        language=Languages.ENGLISH_UNITED_STATES
    )
    
    assert pipeline.video_path == video_path
    assert pipeline.video_id == video_id
    assert pipeline.provider == mock_provider
    assert pipeline.language == Languages.ENGLISH_UNITED_STATES

@pytest.mark.unit
def test_ingestion_pipeline_validation(mock_provider):
    """Verify that language is required if transcript_path is missing."""
    with pytest.raises(ValueError, match="language parameter is required"):
        IngestionPipeline(
            video_path="test.mp4",
            video_id="test_hash",
            provider=mock_provider,
            language=None,
            transcript_path=None
        )

@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingestion_pipeline_run_setup(mock_provider):
    """Test that run() correctly sets up the runner and context."""
    pipeline = IngestionPipeline(
        video_path="test.mp4",
        video_id="test_hash",
        provider=mock_provider,
        language=Languages.ENGLISH_UNITED_STATES
    )
    
    from unittest.mock import AsyncMock
    with patch("mmct.video_pipeline.core.ingestion.ingestion_pipeline.get_video_duration", return_value=10.0), \
         patch("mmct.video_pipeline.core.ingestion.ingestion_pipeline.get_media_folder", return_value="/tmp/media"), \
         patch("mmct.video_pipeline.core.ingestion.ingestion_pipeline.PipelineRunner") as MockRunner:
        
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run = AsyncMock(return_value=MagicMock(status="success"))
        
        await pipeline.run()
        
        MockRunner.assert_called_once()
        mock_runner_instance.run.assert_called_once()
