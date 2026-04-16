import pytest
from mmct.video_pipeline.core.ingestion.pipelines.steps.data_store import StepDataStore

@pytest.fixture
def store():
    return StepDataStore()

@pytest.mark.unit
def test_data_store_set_get(store):
    """Verify basic set and get operations."""
    store.set("step1", "key1", "value1")
    assert store.get("step1", "key1") == "value1"
    assert store.get("step1", "missing") is None
    assert store.get("step1", "missing", default="default") == "default"

@pytest.mark.unit
def test_data_store_has(store):
    """Verify has() logic for steps and keys."""
    store.set("step1", "key1", "value1")
    assert store.has("step1") is True
    assert store.has("step1", "key1") is True
    assert store.has("step1", "key2") is False
    assert store.has("step2") is False

@pytest.mark.unit
def test_data_store_get_all(store):
    """Verify retrieving all values for a step."""
    store.set("step1", "k1", "v1")
    store.set("step1", "k2", "v2")
    all_data = store.get_all("step1")
    assert all_data == {"k1": "v1", "k2": "v2"}
    assert store.get_all("missing") == {}

@pytest.mark.unit
def test_data_store_clear(store):
    """Verify clearing specific steps and the whole store."""
    store.set("step1", "k1", "v1")
    store.set("step2", "k2", "v2")
    
    store.clear("step1")
    assert store.has("step1") is False
    assert store.has("step2") is True
    
    store.clear()
    assert store.has("step2") is False
