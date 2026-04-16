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

## Usage

The Graph Agent is typically invoked via the unified pipeline — see the root [README.md](../../../README.md) Video Q&A example using (examples/video_pipeline.ipynb)