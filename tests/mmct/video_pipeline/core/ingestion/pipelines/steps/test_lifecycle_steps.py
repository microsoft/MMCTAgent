import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import StepContext, StepResult
from mmct.video_pipeline.core.ingestion.pipelines.steps.data_store import StepDataStore

# Import the actual classes to test
from mmct.video_pipeline.core.ingestion.pipelines.steps.export.step import ExportStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.cleanup.step import CleanupStep

@pytest.fixture
def lifecycle_context():
    ctx = StepContext(
        video_path="test.mp4",
        provider=MagicMock(),
        data_store=StepDataStore(),
        logger=MagicMock(),
        verbosity=1,
        video_id="test_vid",
        output_dir="/tmp/test_output"
    )
    return ctx

@pytest.mark.asyncio
@pytest.mark.unit
async def test_export_step(lifecycle_context):
    """Verify generation of export artifacts."""
    step = ExportStep(step_id="exp", params={"include_graph_json": True})
    
    # Mock some data in the store
    lifecycle_context.data_store.set("transcribe", "transcript", {"text": "hi"})
    
    mock_provider = MagicMock()
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 1
    mock_provider._graph = mock_graph
    lifecycle_context.data_store.set("graph_construction", "graph_provider", mock_provider)
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.export.step.os.makedirs"), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.export.step.open", create=True):
        
        result = await step.run(lifecycle_context)
        # Export steps usually return paths to artifacts
        assert result.outputs["run_id"] is not None

@pytest.mark.asyncio
@pytest.mark.unit
async def test_cleanup_step(lifecycle_context):
    """Verify removal of temporary files."""
    step = CleanupStep(step_id="clean", params={"keep_keyframes": False})
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.cleanup.step.CleanupManager") as MockManager:
        mock_instance = MockManager.return_value
        mock_instance.cleanup = AsyncMock(return_value=5)
        
        result = await step.run(lifecycle_context)
        assert result.metrics["items_deleted"] == 5
        assert result.outputs["cleanup_completed"] is True
