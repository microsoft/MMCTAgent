# Graph Agent (Swarm)

Agentic, swarm-based orchestration for video question answering over a Neo4j knowledge graph. A team of specialized agents (Planner, Video, Image, Critic) collaborate via AutoGen Swarm handoffs to plan retrieval, traverse the graph, inspect keyframes, and synthesize a cited answer.

<p align="center">
  <img src="../../../docs/multimedia/media_src/Graph%20Agent.jpg" alt="Graph Agent architecture" width="85%" />
</p>

## Workflow

1. **Planner** — analyzes the query and produces a retrieval plan.
2. **VideoAgent** — searches and traverses the Neo4j knowledge graph.
3. **ImageAgent** — performs perceptual analysis on selected keyframes.
4. **CriticAgent** — validates the drafted answer (optional).
5. **Planner** — synthesizes the final answer with evidence and citations.

## Layout

```
graph_agent/
├── orchestrator.py   # Swarm coordinator and entry point
├── agents/           # Planner, Video, Image, Critic agents
├── tools/            # Graph search/traversal, keyframe, overview, discovery
├── query/            # Neo4j query provider
├── prompts/          # Agent system prompts
└── schemas.py        # Response and state schemas
```

## Custom Tool Middleware

You can intercept any agent tool call with **before** and **after** hooks — no extra LLM calls, no core code changes.

### Quick start

```python
from mmct.video_pipeline.graph_agent import (
    GraphOrchestrator,
    ToolMiddleware,
    get_query_context,
)

class AuthMiddleware(ToolMiddleware):
    """Filter results to only videos the user can access."""

    def applies_to(self, tool_name: str) -> bool:
        return tool_name in {"search_graph", "find_relevant_videos"}

    async def before_tool_call(self, tool_name, kwargs):
        ctx = get_query_context()
        user_id = ctx.get("user_id")
        allowed = await my_auth_service.get_allowed_videos(user_id)
        if "video_ids" in kwargs and kwargs["video_ids"]:
            kwargs["video_ids"] = [v for v in kwargs["video_ids"] if v in allowed]
        return kwargs

    async def after_tool_call(self, tool_name, result, kwargs):
        # optionally filter the result string
        return result

orchestrator = GraphOrchestrator(
    model_client=client,
    neo4j_provider=provider,
    tool_middleware=[AuthMiddleware()],
)

result = await orchestrator.query(
    "What topics are covered?",
    query_context={"user_id": "user_123"},
)
```

### How it works

| Concept | Detail |
|---|---|
| **ToolMiddleware** ABC | Subclass and override `before_tool_call`, `after_tool_call`, and optionally `applies_to`. |
| **`apply_middleware()`** | Utility that wraps a tool callable with all applicable middlewares. Uses `functools.wraps` to preserve the original signature so AutoGen schema generation is unaffected. |
| **Ordering** | Before hooks run first → last. After hooks run last → first (standard middleware unwinding). |
| **`query_context`** | Pass per-query data (user_id, roles, etc.) via the `query_context` dict on `orchestrator.query()`. Middleware reads it with `get_query_context()` — no extra tool parameters needed. |
| **Error handling** | Middleware errors are logged but don't block the tool call. |

A full working example is at `scripts/custom_steps/auth_middleware_example.py`.

## Usage

The Graph Agent is typically invoked via the unified pipeline — see the root [README.md](../../../README.md) Video Q&A example using (examples/video_pipeline.ipynb)