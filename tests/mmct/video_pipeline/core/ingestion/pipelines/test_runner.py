import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from mmct.video_pipeline.core.ingestion.pipelines.runner import PipelineRunner
from mmct.video_pipeline.core.ingestion.pipelines.config.schemas import PipelineConfig, StepConfig
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import StepContext, StepResult

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=StepContext)
    ctx.data_store = MagicMock()
    ctx.logger = MagicMock()
    ctx.verbosity = 1
    ctx.video_id = "test_video"
    ctx.output_dir = "/tmp"
    ctx.save_local_report = False
    return ctx

@pytest.mark.asyncio
@pytest.mark.unit
async def test_runner_sequential_execution(mock_context):
    """Verify that the runner executes steps in sequence and passes data."""
    config = PipelineConfig(
        name="test_pipeline",
        steps=[
            StepConfig(id="step1", type="test.step"),
            StepConfig(id="step2", type="test.step")
        ]
    )
    
    runner = PipelineRunner(pipeline_config=config, context=mock_context)
    
    # Mock the step class and its run method
    mock_result1 = StepResult(step_id="step1", outputs={"data": "val1"})
    mock_result2 = StepResult(step_id="step2", outputs={"data": "val2"})
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.runner.get_step_class") as mock_get_class:
        MockStepClass = MagicMock()
        mock_get_class.return_value = MockStepClass
        
        # Configure the mock instances to return specified results
        step_instance1 = MockStepClass.return_value
        step_instance2 = MockStepClass.return_value # Registry returns same class for simplicity in this mock
        
        step_instance1.run = AsyncMock(side_effect=[mock_result1, mock_result2])
        
        report = await runner.run()
        
        assert report.status == "completed"
        assert len(report.steps) == 2
        # Verify data was set in the store
        assert mock_context.data_store.set.call_count == 2
        mock_context.data_store.set.assert_any_call("step1", "data", "val1")

@pytest.mark.asyncio
@pytest.mark.unit
async def test_runner_error_handling(mock_context):
    """Verify that a step failure marks the pipeline as failed."""
    config = PipelineConfig(
        name="fail_pipeline",
        steps=[StepConfig(id="step1", type="test.fail")]
    )
    
    runner = PipelineRunner(pipeline_config=config, context=mock_context)
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.runner.get_step_class") as mock_get_class:
        MockStepClass = MagicMock()
        mock_get_class.return_value = MockStepClass
        MockStepClass.return_value.run = AsyncMock(side_effect=Exception("Step Failed"))
        
        with pytest.raises(Exception, match="Step Failed"):
            await runner.run()
        
        # Even with exception, runner should have an execution record
        assert len(runner.execution_records) == 1
        assert runner.execution_records[0].status == "failed"
