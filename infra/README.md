# MMCTAgent — Infrastructure

This folder contains everything needed to run MMCTAgent end-to-end:

| Subfolder | Purpose |
| --- | --- |
| [`neo4j/`](neo4j/) | Deploy the Neo4j graph database (local Docker **or** Azure VM) |
| [`container_app/`](container_app/) | Build and deploy the FastAPI and MCP containers to Azure Container Apps |

---

## Why Neo4j is required

MMCTAgent's video pipeline converts raw video into a **temporal knowledge graph**. Every
scene segment, extracted object, transcription chunk, and inter-frame relationship is
written as a node or edge in Neo4j during ingestion. At query time the Graph Agent and
Graph State pipeline both read directly from that graph.

Without a running Neo4j instance:

- `python -m mmct ingest …` fails at the graph-write step.
- All `/video-query` API endpoints return empty results or raise a connection error.
- The MCP server graph tools have no data to serve.

Neo4j must be running and reachable **before** you start the API or MCP server.

---

## Quick start — Neo4j

| Goal | `version` | `vm_deployment` | Command |
| --- | --- | --- | --- |
| Local dev (free) | `community` | `false` | `python infra/neo4j/deploy.py` |
| Azure VM (free) | `community` | `true` | `python infra/neo4j/deploy.py` |
| Azure VM (enterprise) | `enterprise` | `true` | `python infra/neo4j/deploy.py` |

Edit [`neo4j/neo4j_config.yaml`](neo4j/neo4j_config.yaml) to match your scenario, then run.

## Quick start — Container Apps

| Goal | `service` | Command |
| --- | --- | --- |
| Deploy FastAPI only | `api` | `python infra/container_app/deploy.py` |
| Deploy MCP server only | `mcp` | `python infra/container_app/deploy.py` |
| Deploy both | `both` | `python infra/container_app/deploy.py` |

Edit [`container_app/container_app_config.yaml`](container_app/container_app_config.yaml) to set `service` and `acr_name`, then run.

---

## Pre-requisites

### Neo4j — local Docker (`vm_deployment: false`)
- Docker Engine / Docker Desktop running (`docker info` must succeed).
- No Azure account needed. Community edition only.

### Neo4j — Azure VM (`vm_deployment: true`)
- Azure CLI installed and logged in (`az login`).
- Three environment variables exported:

  ```bash
  export AZURE_RESOURCE_GROUP=<your-resource-group>
  export AZURE_LOCATION=<e.g. eastus>
  export AZURE_ADMIN_PASSWORD=<strong-vm-password>
  ```

### Container Apps
- Docker Engine / Docker Desktop running.
- Azure CLI installed and logged in (`az login`).
- ACR already exists (`az acr show --name <acr_name>`).
- Two environment variables exported:

  ```bash
  export AZURE_RESOURCE_GROUP=<your-resource-group>
  export AZURE_LOCATION=<e.g. eastus>
  ```

- All other settings (`acr_name`, `service`, replicas, etc.) live in
  [`container_app/container_app_config.yaml`](container_app/container_app_config.yaml).

---

## End-to-end walkthrough

```bash
# ── Step 1: Configure and start Neo4j ─────────────────────────────────────────

# Edit neo4j_config.yaml (set version, vm_deployment, password)

# Local Docker:
python infra/neo4j/deploy.py

# OR Azure VM:
export AZURE_RESOURCE_GROUP=mmct-rg
export AZURE_LOCATION=eastus
export AZURE_ADMIN_PASSWORD=<strong-password>
python infra/neo4j/deploy.py
# Wait ~10 min. Script prints bolt URI when done.

# ── Step 2: Copy Neo4j connection strings into .env ───────────────────────────
# NEO4J_URI=bolt://...
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=<your-password>
# NEO4J_DATABASE=neo4j

# ── Step 3: Configure container_app_config.yaml ───────────────────────────────
# Set acr_name and service (api | mcp | both)

# ── Step 4: Deploy to Azure Container Apps ────────────────────────────────────
export AZURE_RESOURCE_GROUP=mmct-rg
export AZURE_LOCATION=eastus
python infra/container_app/deploy.py
# Prints the live HTTPS endpoint(s) when done.
```

---

## After deployment — update your `.env`

Both deploy scripts print the exact lines to add. Key entries:

```text
NEO4J_URI=bolt://localhost:7687     ← replace with printed value
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=                     ← must match neo4j_config.yaml → password
NEO4J_DATABASE=neo4j
```

---

## Folder structure

```text
infra/
├── README.md                              ← you are here
├── neo4j/
│   ├── README.md                          ← Neo4j deployment guide + troubleshooting
│   ├── neo4j_config.yaml                  ← edit this before running deploy.py
│   ├── deploy.py                          ← orchestrator: local Docker or Azure VM
│   └── bicep/
│       ├── main.bicep                     ← Community: Azure VM + VNet + LB
│       ├── main.parameters.json
│       ├── main-enterprise.bicep          ← Enterprise: Azure VM + metrics + APOC Extended
│       ├── main-enterprise.parameters.json
│       └── azure.yaml                     ← Azure Developer CLI (azd) config
└── container_app/
    ├── README.md                          ← Container Apps deployment guide
    ├── container_app_config.yaml          ← edit this before running deploy.py
    ├── deploy.py                          ← build → push to ACR → deploy Container App(s)
    ├── Dockerfile.api                     ← FastAPI image (extends Dockerfile.base)
    ├── Dockerfile.mcp                     ← MCP server image (extends Dockerfile.base)
    └── bicep/
        ├── container_env.bicep            ← Log Analytics + Container Apps Environment
        └── container_env.parameters.json
```
