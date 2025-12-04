# GitHub Copilot Instructions – Agentic Framework

You are assisting in building a **Python agentic framework** (like AutoGen) that allows users to create:
- LLM-based agents and non-LLM agents,
- function-based tools callable by LLMs,
- multi-agent collaboration using a **swarm mentality** (handoff and message-passing between agents).

The framework must be:
- **Provider-agnostic**, with initial support for **Azure OpenAI** but easy extensibility to other LLM providers.
- **Modular, well-abstracted, and maintainable** rather than a monolithic script.

---

## Core guidelines when generating or modifying code

### Architecture & Abstractions
- Keep the system loosely coupled:
  - LLM provider logic belongs only in LLM client classes (e.g., `AzureOpenAIClient`).
  - Tool management belongs in tool abstractions.
  - Agent reasoning + messaging belongs in agent classes.
  - Swarm / orchestration logic belongs in an orchestrator component.
- Do **not** hard-code Azure/OpenAI behavior into agent or swarm logic — always depend on a base interface like `BaseLLMClient`.

### Coding Style
- Use **Python 3.11+**, **type hints everywhere**, and clear **docstrings**.
- Prefer **composition over inheritance**; keep classes focused on a single responsibility.
- Small, readable functions are preferred over deeply nested logic.
- Use **logging**, not `print()`.
- Handle failures explicitly with meaningful exceptions rather than swallowing them silently.

### Tools / Function Calling
- Tools should wrap real Python functions and expose:
  - Name, description, argument schema, callable reference.
- Argument validation and error messaging must be clear and safe.
- Agents must decide deterministically whether to use a tool or respond directly.

### Agents & Swarm Collaboration
- Agent messaging and state should be explicit and inspectable — avoid hidden coupling.
- Handoff between agents should be managed through structured messages (not implicit action).
- The orchestrator must control turn-taking, routing, stopping conditions, and safety — not the agents themselves.

### Extensibility
- Design everything to support **future LLM providers** without rewriting core logic.
- Keep configuration isolated (env vars, credentials, Azure deployment name, etc.) and never leak it into business logic.

---

## What NOT to do
- ❌ Do **NOT** create or modify **tests** unless explicitly asked.
- ❌ Do **NOT** introduce tight coupling between agents, tools, and LLM providers.
- ❌ Do **NOT** mix orchestration logic into individual agents.
- ❌ Do **NOT** rely on global state or side effects in message passing.
- ❌ Do **NOT** expose secrets or access environment variables inside random modules — all config must flow from a central settings object.

---

## What Copilot should optimize for
- Readability and maintainability over cleverness.
- Predictable abstractions that are easy for new users to extend.
- Clear separation of concerns so that:
  - Adding a new LLM provider is easy,
  - Adding a new agent is easy,
  - Adding a new tool is easy,
  - Evolving swarm policies does NOT break existing components.

When unsure, prioritize **clarity, modularity, and future extensibility** over short-term convenience.
