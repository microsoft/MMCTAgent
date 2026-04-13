# MMCTAgent Testing Guide

This directory contains a comprehensive testing suite for the Multi-Modal Critical Thinking Agent (MMCT). The tests follow a tiered strategy designed for reliability and speed.

## 🎯 Testing Scope & Coverage

| Level | Component | Test Script | Description |
| :--- | :--- | :--- | :--- |
| **Config** | .env Loader | `tests/config/test_provider_config.py` | Validates environment variable loading and placeholder detection. |
| **Providers** | Azure Services | `tests/mmct/providers/test_azure_providers.py` | Unit tests for LLM, Storage, and Speech clients (Mocked). |
| **Pipeline Core** | State & Runner | `tests/mmct/video_pipeline/core/ingestion/pipelines/` | Tests for `StepDataStore`, `PipelineRunner`, and `BaseStep`. |
| **Ingestion Steps** | Preparation | `.../steps/test_preparation_steps.py` | Validation, Compression, Transcription, Chunking logic. |
| **Ingestion Steps** | Analysis | `.../steps/test_analysis_steps.py` | Keyframe extraction, Chapter generation, Temporal reasoning. |
| **Ingestion Steps** | Graph Utils | `.../steps/test_graph_steps.py` | Graph construction and upload logic verification. |
| **Orchestration** | Ingestion Mgr | `tests/mmct/video_pipeline/test_video_ingestion.py` | High-level `VideoIngestionPipeline` orchestration. |
| **API** | FastAPI Layer | `tests/api/test_api.py` | Verifies health check and documentation endpoints. |
| **MCP** | Model Context | `tests/mcp_server/test_server.py` | Verifies MCP server initialization and tool availability. |

## 🛠️ Execution Commands

### 1. Run All Unit Tests (Fast, No Cloud Costs)
```bash
conda run -n mmct-agent pytest tests/ -m unit -vv
```

### 2. Run Integration Tests (Requires Real .env)
```bash
# Verify connectivity to Azure LLM
conda run -n mmct-agent pytest tests/config/test_provider_config.py -m smoke -vv
```

### 3. Run End-to-End Smoke Test (Real Video)
```bash
# Performs full ingestion from scratch
conda run -n mmct-agent pytest tests/test_smoke_ingestion.py -vv
```

### 4. Run Specific Component Tests
```bash
# Test only the Ingestion Steps
conda run -n mmct-agent pytest tests/mmct/video_pipeline/core/ingestion/pipelines/steps/ -vv
```

## 📋 Inter-Step Communication
Inter-step data flow is verified in:
- `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_data_store.py` (Centralized state)
- `tests/mmct/video_pipeline/core/ingestion/pipelines/test_runner.py` (Execution orchestration)

## 📊 View Reports
After running ingestion tests, execution reports and artifacts can be found in the `media/` directory under the respective `video_id` folders.
