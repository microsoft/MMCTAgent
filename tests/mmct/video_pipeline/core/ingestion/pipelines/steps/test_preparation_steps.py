import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import StepContext, StepResult
from mmct.video_pipeline.core.ingestion.pipelines.steps.data_store import StepDataStore

# Import the actual classes to test
from mmct.video_pipeline.core.ingestion.pipelines.steps.graph_validation.step import GraphValidationStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.compress.step import CompressionStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.transcription_step import TranscriptionStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.video_chunking.step import VideoChunkingStep

@pytest.fixture
def base_context():
    ctx = StepContext(
        video_path="test.mp4",
        provider=MagicMock(),
        data_store=StepDataStore(),
        logger=MagicMock(),
        verbosity=1,
        video_id="test_vid"
    )
    return ctx

@pytest.mark.asyncio
@pytest.mark.unit
async def test_graph_validation_step(base_context):
    """Verify graph validation logic."""
    step = GraphValidationStep(step_id="val", params={"require_chapter_groups": True})
    
    # Mock graph_store_provider.get_video_stats
    mock_stats = {"node_counts": {"Chapter": 5, "ChapterGroup": 1}}
    base_context.provider.graph_store_provider = MagicMock()
    base_context.provider.graph_store_provider.get_video_stats = AsyncMock(return_value=mock_stats)
    
    result = await step.run(base_context)
    assert result.outputs["should_continue"] is False # Should stop if already exists
    assert result.outputs["graph_exists"] is True

@pytest.mark.asyncio
@pytest.mark.unit
async def test_video_compression_step(base_context):
    """Verify compression step logic (mocked VideoCompressor)."""
    step = CompressionStep(step_id="comp", params={"target_resolution": 720})
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.compress.step.os.path.exists", return_value=True), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.compress.step.os.path.getsize", return_value=100*1024*1024), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.compress.step.get_media_folder", return_value="/tmp"), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.compress.step.VideoCompressor") as MockCompressor:
        
        mock_compressor_instance = MockCompressor.return_value
        mock_compressor_instance.needs_transcode.return_value = True
        mock_compressor_instance.output_path = "compressed.mp4"
        
        result = await step.run(base_context)
        assert result.outputs["video_path"] == "compressed.mp4"

@pytest.mark.asyncio
@pytest.mark.unit
async def test_transcription_step(base_context):
    """Verify transcription integration with provider."""
    step = TranscriptionStep(step_id="trans", params={})
    
    mock_transcript = "Hello world"
    # TranscriptionStep uses context.provider.transcription_provider.transcribe_video
    base_context.provider.transcription_provider.transcribe_video = AsyncMock(
        return_value=(mock_transcript, ["out.srt"])
    )
    
    base_context.data_store.set("compress", "video_path", "compressed.mp4")
    
    result = await step.run(base_context)
    assert result.outputs["transcript"] == mock_transcript

@pytest.mark.asyncio
@pytest.mark.unit
async def test_video_chunking_step(base_context):
    """Verify chunking logic (mocked SemanticChunker)."""
    step = VideoChunkingStep(step_id="chunk", params={"chunking_strategy": "transcript"})
    
    base_context.data_store.set("transcribe", "transcript", "test")
    base_context.data_store.set("compress", "video_path", "test.mp4")
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.video_chunking.step.Path.exists", return_value=True), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.video_chunking.step.SemanticChunker") as MockSemantic, \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.video_chunking.step.TranscriptChunker") as MockTranscript:
        
        MockSemantic.return_value.run = AsyncMock(return_value=[{"start": 0, "end": 10}])
        MockTranscript.return_value.run = AsyncMock(return_value=[{"id": 1, "start": 0, "end": 10}])
        
        result = await step.run(base_context)
        assert len(result.outputs["video_chunks"]) == 1

from mmct.video_pipeline.core.ingestion.pipelines.steps.extraction_planning.step import ExtractionPlanningStep

import subprocess, sys, os
_repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
sys.path.insert(0, os.path.join(_repo_root, "scripts"))
from custom_steps.uniform_frames import UniformFrameExtractionStep

@pytest.mark.asyncio
@pytest.mark.unit
async def test_extraction_planning_step(base_context):
    """Verify extraction planning logic."""
    step = ExtractionPlanningStep(step_id="plan", params={})
    base_context.data_store.set("chunk", "video_chunks", [{"id": 1, "start": 0, "end": 10}])
    
    result = await step.run(base_context)
    assert "extraction_plan" in result.outputs

@pytest.mark.asyncio
@pytest.mark.unit
async def test_uniform_frame_extraction_step(base_context):
    """Verify uniform frame extraction and upload."""
    step = UniformFrameExtractionStep(step_id="frames", params={})
    base_context.data_store.set("compress", "video_path", "test.mp4")
    base_context.output_dir = "/tmp"
    
    with patch("custom_steps.uniform_frames._extract_frames_at_1fps", return_value=[{"timestamp_second": 1, "filepath": "f1.jpg"}]):
        
        base_context.provider.storage_provider.upload_file = AsyncMock(return_value="http://blob/f1.jpg")
        
        result = await step.run(base_context)
        assert len(result.outputs["frames"]) == 1
