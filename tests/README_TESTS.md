# MMCTAgent Testing Guide

This directory contains a comprehensive testing suite for the Multi-Modal Critical Thinking
Agent (MMCT). Tests are split into two clear tiers: **unit tests** (fast, no cloud costs,
mocked dependencies) and **connectivity tests** (real service calls, require a valid `.env`).

---

## Test Tiers

### Tier 1 — Unit Tests (fast, no cloud)

Mock-based tests that verify logic, data flow, and component behaviour in isolation.
Run these first — they complete in seconds and require no Azure credentials.

| Component | Test file |
| --- | --- |
| Config / .env loading | `tests/config/test_provider_config.py` |
| Pipeline state & runner | `tests/mmct/video_pipeline/core/ingestion/pipelines/test_runner.py` |
| Pipeline base step | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_step_base.py` |
| Ingestion steps (preparation) | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_preparation_steps.py` |
| Ingestion steps (analysis) | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_analysis_steps.py` |
| Ingestion steps (graph) | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_graph_steps.py` |
| Ingestion steps (lifecycle) | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_lifecycle_steps.py` |
| Ingestion steps (data store) | `tests/mmct/video_pipeline/core/ingestion/pipelines/steps/test_data_store.py` |
| Video ingestion orchestration | `tests/mmct/video_pipeline/test_video_ingestion.py` |
| Video pipeline orchestrators | `tests/mmct/video_pipeline/test_orchestrators.py` |
| Image pipeline | `tests/mmct/image_pipeline/test_image_pipeline.py` |
| API (health + docs) | `tests/api/test_api.py` |
| MCP server init + tools | `tests/mcp_server/test_mcp_server.py` |

```bash
pytest tests/ -m unit -vv
```

---

### Tier 2 — Connectivity Tests (real .env required)

Real accessibility checks — each test makes an actual call to the live service.
Tests **skip automatically** if the relevant credential is missing or is still a
placeholder (`<your-...>`) in `.env`. They never fail purely due to missing config.

| Service | Test | Skips if missing |
| --- | --- | --- |
| Azure OpenAI (LLM) | `test_llm_connectivity` | `LLM_ENDPOINT` |
| Azure OpenAI (Embedding) | `test_embedding_connectivity` | `EMBEDDING_SERVICE_ENDPOINT` |
| Azure Blob Storage | `test_storage_connectivity` | `STORAGE_ACCOUNT_NAME` |
| Azure Speech Service | `test_speech_connectivity` | `SPEECH_SERVICE_REGION` |
| Neo4j | `test_neo4j_connectivity` | `NEO4J_PASSWORD` |
| Whisper (Azure OpenAI) | `test_whisper_connectivity` | `WHISPER_DEPLOYMENT_NAME` |

API keys are optional — if placeholder values (`<your-...>`) are present, providers
fall back to **Azure CLI credential** (`az login`) automatically.

```bash
# Run all connectivity checks
pytest tests/connectivity/ -m connectivity -vv

# Run a single service check
pytest tests/connectivity/test_service_connectivity.py::test_neo4j_connectivity -vv

# Run provider-layer connectivity checks only
pytest tests/mmct/providers/ -m connectivity -vv
```

---

## End-to-end Smoke Test (real video)

Performs full ingestion from scratch. Requires all services to be reachable.

```bash
pytest tests/test_smoke_ingestion.py -vv
```

---

## Quick Reference

| Goal | Command |
| --- | --- |
| Fast unit tests only | `pytest tests/ -m unit -vv` |
| Connectivity / env health check | `pytest tests/connectivity/ -m connectivity -vv` |
| Single service check (e.g. Neo4j) | `pytest tests/connectivity/test_service_connectivity.py::test_neo4j_connectivity -vv` |
| Full smoke test (real video) | `pytest tests/test_smoke_ingestion.py -vv` |
| Everything | `pytest tests/ -vv` |

