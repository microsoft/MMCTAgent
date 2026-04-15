# Graph State (Deterministic Pipeline)

A deterministic state-machine workflow for video question answering over a Neo4j knowledge graph. Unlike the agentic Graph Agent, this pipeline follows an explicit state machine for high-precision, reproducible, and efficient retrieval.

<p align="center">
  <img src="../../../docs/multimedia/media_src/Graph%20State.jpg" alt="Graph State pipeline" width="85%" />
</p>

## States

The pipeline transitions through well-defined states (see [state_machine.py](state_machine.py)):

`PARSE_INPUT` → `PLAN` → `VALIDATE_PLAN` → `DISCOVER_VIDEOS` → `RETRIEVE` → `CHECK_EVIDENCE` → `EXPAND_CONTEXT` / `REPHRASE` → `ANALYZE_IMAGES` → `SYNTHESIZE` → `CRITIQUE` → `REVISE` → `SUBMIT`

Hard limits on retries and sub-queries are enforced in code to keep runs bounded and reproducible.

## Layout

```
graph_state/
├── orchestrator.py     # Pipeline entry point
├── state_machine.py    # States, context, and limits
├── agents/             # Planner and Critic agents
├── tools/              # Retrieval, video discovery, image analysis
├── query/              # Neo4j query provider
├── llm/                # LLM client helpers
├── prompts/            # State-specific prompts
└── schemas.py          # Response and state schemas
```

## Usage

Typically invoked via the unified pipeline — see the root [README.md](../../../README.md) Video Q&A example using (examples/video_pipeline.ipynb)
