import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import StepContext, StepResult
from mmct.video_pipeline.core.ingestion.pipelines.steps.data_store import StepDataStore

# Import the actual classes to test
from mmct.video_pipeline.core.ingestion.pipelines.steps.keyframes.step import KeyframeExtractionStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.chapters.step import ChapterGenerationStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.temporal_graph.step import TemporalGraphStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.chapter_grouping.step import ChapterGroupingStep

@pytest.fixture
def analysis_context():
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
async def test_keyframe_extraction_step(analysis_context):
    """Verify keyframe selection logic."""
    step = KeyframeExtractionStep(step_id="kf", params={"sample_fps": 1})
    
    analysis_context.data_store.set("video_chunking", "video_chunks", [{"chunk_id": "c1", "start": 0, "end": 10}])
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.keyframes.step._extract_keyframes_from_video", return_value=[{"timestamp": 1, "filepath": "f1.jpg"}]), \
         patch("mmct.video_pipeline.core.ingestion.pipelines.steps.keyframes.step.detect_action_boundaries", return_value=[]):
        
        result = await step.run(analysis_context)
        assert len(result.outputs["keyframes_per_chunk"]) == 1
        assert result.metrics["total_keyframes"] == 1

@pytest.mark.asyncio
@pytest.mark.unit
async def test_chapter_generation_step(analysis_context):
    """Verify multimodal chapter summarization."""
    step = ChapterGenerationStep(step_id="chap", params={})
    
    analysis_context.data_store.set("video_chunking", "video_chunks", [{"start": 0, "end": 10}])
    analysis_context.data_store.set("keyframes", "keyframes_per_chunk", [{"chunk_id": "c1", "keyframes": [{"filepath": "f1.jpg"}]}])
    
    # Mock LLM and extraction helper
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.chapters.step.extract_chapters_parallel", return_value=([{"summary": "Chapter Summary", "start": 0, "end": 10}], [])):
        
        result = await step.run(analysis_context)
        assert len(result.outputs["chapters"]) == 1
    assert result.outputs["chapters"][0]["summary"] == "Chapter Summary"

@pytest.mark.asyncio
@pytest.mark.unit
async def test_temporal_graph_extraction_step(analysis_context):
    """Verify extraction of events and objects from chapters."""
    step = TemporalGraphStep(step_id="tg", params={})
    
    analysis_context.data_store.set("chap", "chapters", [{"summary": "Test", "start": 0, "end": 10}])
    analysis_context.data_store.set("kf", "keyframes", [["f1.jpg"]])
    
    # Mock LLM for event/object extraction
    analysis_context.provider.llm_provider.chat_completion = AsyncMock(return_value={"content": '{"events": [{"description": "Action"}], "objects": []}'})
    
    result = await step.run(analysis_context)
    assert "events" in result.outputs
    assert "objects" in result.outputs

@pytest.mark.asyncio
@pytest.mark.unit
async def test_chapter_grouping_step(analysis_context):
    """Verify semantic similarity grouping."""
    step = ChapterGroupingStep(step_id="group", params={"similarity_threshold": 0.5})
    
    analysis_context.data_store.set("chapters", "chapters", [
        {"summary": "A", "start": 0, "end": 5, "chapter_id": "1"},
        {"summary": "B", "start": 5, "end": 10, "chapter_id": "2"}
    ])
    
    # Mock Embedding provider
    analysis_context.provider.embedding_provider.embed_query = AsyncMock(return_value=[0.1, 0.2])
    
    # Mock LLM for group summary
    analysis_context.provider.llm_provider.chat_completion = AsyncMock(return_value={"content": "Group Summary"})
    
    # Mock ChapterGrouper
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.chapter_grouping.step.ChapterGrouper") as MockGrouper:
        mock_grouper = MockGrouper.return_value
        from mmct.video_pipeline.core.ingestion.models import ChapterGroup
        mock_grouper.group_chapters.return_value = ([ChapterGroup(id="g1", name="Group 1", start_time=0, end_time=10, chapter_indices=[0, 1])], [])
        
        result = await step.run(analysis_context)
        assert "chapter_groups" in result.outputs
        assert result.outputs["group_count"] == 1
