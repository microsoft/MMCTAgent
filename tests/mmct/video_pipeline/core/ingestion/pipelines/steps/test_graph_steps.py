import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import StepContext, StepResult
from mmct.video_pipeline.core.ingestion.pipelines.steps.data_store import StepDataStore

# Import the actual classes to test
from mmct.video_pipeline.core.ingestion.pipelines.steps.keyframe_upload.step import KeyframeUploadStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.graph_construction.step import GraphConstructionStep
from mmct.video_pipeline.core.ingestion.pipelines.steps.graph_upload.step import GraphUploadStep

@pytest.fixture
def graph_context():
    mock_provider = MagicMock()
    mock_provider.graph_provider = MagicMock()
    mock_provider.graph_provider.clear_database = AsyncMock()
    
    ctx = StepContext(
        video_path="test.mp4",
        provider=mock_provider,
        data_store=StepDataStore(),
        logger=MagicMock(),
        verbosity=1,
        video_id="test_vid"
    )
    return ctx

@pytest.mark.asyncio
@pytest.mark.unit
async def test_keyframe_upload_step(graph_context):
    """Verify keyframe upload to storage."""
    step = KeyframeUploadStep(step_id="kf_up", params={"container_name": "test-kf"})
    
    graph_context.data_store.set("keyframes", "keyframes_per_chunk", [{"chunk_id": "0", "keyframes": [{"filepath": "f1.jpg"}]}])
    
    # Mock Storage provider
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.keyframe_upload.step.os.path.exists", return_value=True):
        graph_context.provider.storage_provider.upload_file = AsyncMock(return_value="https://blob/f1.jpg")
        
        result = await step.run(graph_context)
        assert result.outputs["upload_complete"] is True
        assert result.metrics["keyframes_uploaded"] == 1

@pytest.mark.asyncio
@pytest.mark.unit
async def test_graph_construction_step(graph_context):
    """Verify hierarchical graph construction logic."""
    step = GraphConstructionStep(step_id="graph_build", params={"graph_provider": "networkx"})
    
    graph_context.data_store.set("temporal_graph", "events", [{"id": "e1", "start": 0, "end": 5}])
    graph_context.data_store.set("temporal_graph", "objects", [{"id": "o1", "name": "object"}])
    graph_context.data_store.set("chapters", "chapters", [{"summary": "C1", "start": 0, "end": 10}])
    
    with patch("mmct.video_pipeline.core.ingestion.pipelines.steps.graph_construction.step.GraphBuilder") as MockBuilder:
        mock_builder_instance = MockBuilder.return_value
        mock_builder_instance.build_graph = AsyncMock()
        mock_builder_instance.build_graph.return_value = MagicMock(
            chapter_nodes_created=1, event_nodes_created=1, 
            group_temporal_edges_created=0, chapter_temporal_edges_created=0,
            transcript_temporal_edges_created=0, event_temporal_edges_created=0,
            hierarchy_edges_created=0, contains_edges_created=0,
            keyframe_edges_created=0, has_transcript_edges_created=0,
            errors=[]
        )
        
        result = await step.run(graph_context)
        assert result.outputs["graph_built"] is True

@pytest.mark.asyncio
@pytest.mark.unit
async def test_graph_upload_step(graph_context):
    """Verify graph upload to Neo4j."""
    step = GraphUploadStep(step_id="graph_up", params={"provider": "neo4j"})
    
    # Mocking the GraphConstructionStep output
    mock_provider = MagicMock()
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 2
    mock_graph.number_of_edges.return_value = 1
    nodes_data = [
        ("c1", {"label": "Chapter", "_type": "Chapter", "summary": "C1"}),
        ("e1", {"label": "Event", "_type": "Event", "description": "E1"})
    ]
    mock_graph.nodes = MagicMock()
    mock_graph.nodes.return_value = nodes_data
    mock_graph.nodes.__iter__.return_value = iter(nodes_data)
    mock_provider._graph = mock_graph
    
    graph_context.data_store.set("graph_construction", "graph_provider", mock_provider)
    
    # Mock Neo4j provider and embedding providers
    graph_context.provider.graph_store_provider = MagicMock()
    graph_context.provider.graph_store_provider.upload_graph = AsyncMock(return_value={"nodes_uploaded": 2, "edges_uploaded": 1})
    
    with patch("mmct.providers.custom_providers.FastEmbedBGEsmallEmbeddingProvider") as MockText:
        MockText.return_value.batch_embedding = AsyncMock(return_value=[[0.1]*384, [0.1]*384])
        
        result = await step.run(graph_context)
        assert result.outputs["upload_complete"] is True
