import pytest
from unittest.mock import MagicMock, patch
from mmct.video_pipeline.graph_state.orchestrator import StateOrchestrator
from mmct.video_pipeline.graph_agent.orchestrator import GraphOrchestrator
from mmct.providers.base import BaseLLMProvider

@pytest.fixture
def mock_orch_deps():
    return {
        "model_client": MagicMock(),
        "neo4j_provider": MagicMock(),
        "storage_provider": MagicMock(),
        "image_llm_provider": MagicMock(spec=BaseLLMProvider)
    }

@pytest.mark.unit
def test_state_orchestrator_initialization(mock_orch_deps):
    """Test that StateOrchestrator initializes correctly."""
    orchestrator = StateOrchestrator(**mock_orch_deps)
    assert orchestrator.model_client == mock_orch_deps["model_client"]
    assert orchestrator.use_critic is True

@pytest.mark.unit
def test_graph_orchestrator_initialization(mock_orch_deps):
    """Test that GraphOrchestrator initializes correctly."""
    orchestrator = GraphOrchestrator(**mock_orch_deps)
    assert orchestrator.model_client == mock_orch_deps["model_client"]
    assert orchestrator.use_critic is True

@pytest.mark.asyncio
@pytest.mark.unit
async def test_state_orchestrator_query_flow(mock_orch_deps):
    """Verify the query flow of StateOrchestrator with mocked state machine."""
    orchestrator = StateOrchestrator(**mock_orch_deps)
    
    # Mock the internal state machine runner
    mock_response = {"answer": "Test answer", "sources": []}
    with patch.object(StateOrchestrator, "_run_state_machine", return_value=mock_response):
        result = await orchestrator.query(user_query="Who is in the video?")
        assert result["answer"] == "Test answer"
        assert "token_usage" in result

@pytest.mark.asyncio
@pytest.mark.unit
async def test_graph_orchestrator_query_flow(mock_orch_deps):
    """Verify the query flow of GraphOrchestrator using mocked swarm."""
    orchestrator = GraphOrchestrator(**mock_orch_deps)
    
    # Mock the swarm run_stream
    with patch("mmct.video_pipeline.graph_agent.orchestrator.Swarm") as MockSwarm:
        mock_swarm_instance = MockSwarm.return_value
        
        async def mock_run_stream(**kwargs):
            from autogen_agentchat.base import TaskResult
            yield TaskResult(messages=[], stop_reason="completed")
            
        mock_swarm_instance.run_stream = mock_run_stream
        
        with patch.object(GraphOrchestrator, "_process_result", return_value={"answer": "Graph answer", "sources": []}):
            result = await orchestrator.query(user_query="What happens at 1:00?")
            assert result["answer"] == "Graph answer"
