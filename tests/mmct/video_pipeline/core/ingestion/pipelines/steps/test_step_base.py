import pytest
from unittest.mock import MagicMock
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import PipelineStep, StepContext, StepResult

class MockStep(PipelineStep):
    """Concrete implementation of PipelineStep for testing."""
    step_type = "test.mock"
    async def run(self, context: StepContext) -> StepResult:
        return StepResult(step_id=self.step_id)

@pytest.fixture
def mock_context():
    return StepContext(
        video_path="test.mp4",
        provider=MagicMock(),
        user_params={"runtime_key": "runtime_val"}
    )

@pytest.mark.unit
def test_step_param_priority(mock_context):
    """Verify that get_param follows the correct priority hierarchy."""
    step = MockStep(step_id="step1", params={"static_key": "static_val", "shared_key": "static_val"})
    mock_context.user_params["shared_key"] = "runtime_val"
    
    # Priority 1: User runtime params
    assert step.get_param("shared_key", mock_context) == "runtime_val"
    
    # Priority 2: Static YAML params
    assert step.get_param("static_key", mock_context) == "static_val"
    
    # Priority 3: Default value
    assert step.get_param("missing_key", mock_context, default="default") == "default"

@pytest.mark.unit
def test_step_result_initialization():
    """Verify that StepResult ensures metrics and artifacts are not None."""
    result = StepResult(step_id="test", outputs=None, metrics=None, artifacts=None)
    assert isinstance(result.outputs, dict)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.artifacts, list)
