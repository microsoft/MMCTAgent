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

## Custom State Hooks

You can intercept any state transition with **before** and **after** hooks — no extra LLM calls, no core code changes.

### Quick start

```python
from mmct.video_pipeline.graph_state import (
    StateOrchestrator,
    StateHook,
    get_query_context,
)
from mmct.video_pipeline.graph_state.state_machine import QueryContext, QueryState

class AuthHook(StateHook):
    """Filter discovered videos to only those the user can access."""

    def applies_to(self, state):
        return state in {QueryState.DISCOVER_VIDEOS, QueryState.RETRIEVE}

    async def after_state(self, state, ctx, next_state):
        allowed = get_query_context().get("allowed_video_ids")
        if not allowed:
            return None
        if state == QueryState.DISCOVER_VIDEOS:
            ctx.effective_video_ids = [
                v for v in (ctx.effective_video_ids or []) if v in allowed
            ]
        elif state == QueryState.RETRIEVE:
            ctx.evidence = [
                e for e in ctx.evidence if e.get("video_id") in allowed
            ]
        return None

orchestrator = StateOrchestrator(
    model_client=client,
    neo4j_provider=provider,
    state_hooks=[AuthHook()],
)

result = await orchestrator.query(
    "What topics are covered?",
    query_context={"allowed_video_ids": ["vid_1", "vid_2"]},
)
```

### How it works

| Concept | Detail |
|---|---|
| **StateHook** ABC | Subclass and override `before_state`, `after_state`, and optionally `applies_to`. |
| **`before_state(state, ctx)`** | Mutate `QueryContext` in-place before the state handler runs. |
| **`after_state(state, ctx, next_state)`** | Inspect/mutate context after the handler. Return a `QueryState` to override the transition, or `None` to keep the original. |
| **Ordering** | Before hooks run first → last. After hooks run last → first (standard middleware unwinding). |
| **`query_context`** | Pass per-query data (user_id, roles, etc.) via the `query_context` dict on `orchestrator.query()`. Hooks read it with `get_query_context()`. |
| **Error handling** | Hook errors are logged but don't block state execution. |
| **Transition override** | `after_state` returning a `QueryState` lets hooks redirect the pipeline (e.g., skip CRITIQUE, route to ERROR). |

A full working example is at `scripts/custom_steps/auth_state_hook_example.py`.

## Usage

Typically invoked via the unified pipeline — see the root [README.md](../../../README.md) Video Q&A example using (examples/video_pipeline.ipynb)
