# Container App Deployment

MMCTAgent runs two containerised services — a **FastAPI** REST layer and a
**FastMCP** server — each deployable as an Azure Container App. A single YAML
config file drives everything: set which service(s) to deploy, fill in your ACR
name, and run one command.

---

## Quick start

| Goal | `service` | Command |
| --- | --- | --- |
| Deploy FastAPI only | `api` | `python infra/container_app/deploy.py` |
| Deploy MCP server only | `mcp` | `python infra/container_app/deploy.py` |
| Deploy both | `both` | `python infra/container_app/deploy.py` |

Edit [container_app_config.yaml](container_app_config.yaml) to set `service` and
`acr_name`, then run the command. The script prints the live HTTPS endpoint(s)
on completion.

---

## Files

```text
container_app/
├── container_app_config.yaml     ← edit this first (all user decisions live here)
├── deploy.py                     ← run this after editing the config
├── Dockerfile.api                ← FastAPI image (extends Dockerfile.base)
├── Dockerfile.mcp                ← MCP server image (extends Dockerfile.base)
└── bicep/
    ├── container_env.bicep       ← Log Analytics Workspace + Container Apps Environment
    └── container_env.parameters.json
```

Both Dockerfiles extend `Dockerfile.base` at the repo root. `deploy.py` builds
the base image first, then the service image(s) — no manual build order needed.

---

## Services at a glance

| | FastAPI (`api`) | MCP Server (`mcp`) |
| --- | --- | --- |
| Dockerfile | `Dockerfile.api` | `Dockerfile.mcp` |
| Container App name | `mmctagent-api` | `mmctagent-mcp` |
| Port | 8000 | 8000 |
| Key endpoints | `/health`, `/video-query`, `/ingestion`, `/image-query`, `/docs` | `/` (health), `/mcp` |
| Default CPU / RAM | 1.0 vCPU / 2 Gi | 0.5 vCPU / 1 Gi |
| Default replicas | 1–3 | 1–2 |

---

## Step 1 — edit `container_app_config.yaml`

Every field is annotated with its purpose and valid values. Key decisions:

| Field | Notes |
| --- | --- |
| `service` | `api` \| `mcp` \| `both` |
| `acr_name` | ACR name **without** `.azurecr.io` — must already exist |
| `environment_name` | Container Apps Environment name (created by the Bicep template) |
| `api.image_tag` / `mcp.image_tag` | Use `latest` for dev; use `git rev-parse --short HEAD` for production |
| `api.min_replicas` | Set to `1` to avoid cold starts; `0` to save cost when idle |
| `api.cpu` / `api.memory` | Azure requires matching pairs — see table below |

### Valid CPU / memory pairs

| cpu | memory |
| --- | --- |
| `0.25` | `0.5Gi` |
| `0.5` | `1Gi` |
| `0.75` | `1.5Gi` |
| `1.0` | `2Gi` |
| `2.0` | `4Gi` |
| `4.0` | `8Gi` |

---

## Step 2 — fill out `.env`

`deploy.py` reads every variable in your repo-root `.env` and injects them as
environment variables into the Container App at deploy time. No secrets are
stored in Bicep files.

```bash
cp .env.example .env
# Fill in all required values. At minimum:
#   LLM_ENDPOINT, LLM_DEPLOYMENT_NAME, LLM_MODEL_NAME
#   EMBEDDING_SERVICE_ENDPOINT, EMBEDDING_SERVICE_DEPLOYMENT_NAME
#   STORAGE_ACCOUNT_NAME, KEYFRAME_CONTAINER_NAME
#   NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
```

> **Deploy Neo4j first.** Without a running Neo4j instance all video-query
> endpoints will fail. Use `infra/neo4j/deploy.py` and copy the printed
> `NEO4J_URI` into your `.env`.

---

## Step 3 — export shell variables

```bash
export AZURE_RESOURCE_GROUP=<existing-resource-group>
export AZURE_LOCATION=<e.g. eastus, southeastasia, westeurope>
```

---

## Step 4 — run `deploy.py`

```bash
# From the repo root:
python infra/container_app/deploy.py
```

### Pre-requisites

```bash
docker info                              # Docker must be running
az login                                 # Azure CLI must be authenticated
az acr show --name <acr_name>            # ACR must already exist
az group show --name $AZURE_RESOURCE_GROUP
```

The script runs five numbered steps and prints the live HTTPS endpoint(s) on
completion.

---

## What gets deployed

### Container Apps Environment (`bicep/container_env.bicep`)

Created once and reused by both containers (idempotent — safe to re-run).

| Resource | Purpose |
| --- | --- |
| Log Analytics Workspace | Receives all container stdout/stderr |
| Container Apps Environment | Shared serverless runtime for all Container Apps |

### Container App(s)

| Setting | API | MCP |
| --- | --- | --- |
| Image | `<ACR>.azurecr.io/mmctagent-api:<tag>` | `<ACR>.azurecr.io/mmctagent-mcp:<tag>` |
| Ingress | External (public HTTPS) | External (public HTTPS) |
| Env vars | All variables from `.env` | All variables from `.env` |

---

## Connection strings after deployment

`deploy.py` prints the HTTPS endpoint(s) at the end of a successful run.

### FastAPI (`api`)

```text
Base URL  →  https://<fqdn>
Health    →  https://<fqdn>/health
Docs      →  https://<fqdn>/docs
ReDoc     →  https://<fqdn>/redoc
```

API endpoints:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness probe |
| POST | `/ingestion` | Ingest an MP4 video |
| POST | `/video-query` | One-shot knowledge-graph query |
| GET | `/video-query/stream` | SSE streaming query |
| POST | `/image-query` | Analyse an image |

### MCP Server (`mcp`)

```text
Health  →  https://<fqdn>/
MCP     →  https://<fqdn>/mcp
```

Connect any MCP-compatible client (e.g. Claude Desktop) to `https://<fqdn>/mcp`.

---

## Tunable parameters

All fields below live in `container_app_config.yaml`. No other files need
editing for routine tuning.

| Field | Default | Notes |
| --- | --- | --- |
| `api.image_tag` | `latest` | Pin to a Git SHA for production |
| `api.min_replicas` | `1` | `0` saves cost but causes cold starts |
| `api.max_replicas` | `3` | Upper limit for HTTP autoscaling |
| `api.cpu` | `1.0` | vCPU per replica |
| `api.memory` | `2Gi` | RAM per replica — must match cpu tier |
| `mcp.min_replicas` | `1` | Same as above |
| `mcp.max_replicas` | `2` | MCP is lighter — 2 replicas usually sufficient |
| `mcp.cpu` | `0.5` | |
| `mcp.memory` | `1Gi` | |
| `environment_name` | `mmct-env` | Rename to isolate environments (dev/prod) |
| `log_analytics_workspace` | `mmct-logs` | Rename if you share a workspace |

---

## Updating a running deployment

```bash
# Rebuild and redeploy with a traceable image tag
export IMAGE_TAG=$(git rev-parse --short HEAD)
# Update image_tag in container_app_config.yaml, then:
python infra/container_app/deploy.py
```

`deploy.py` detects whether each Container App already exists and runs
`az containerapp update` instead of `create` — safe to re-run at any time.

---

## Viewing logs

```bash
# Stream live logs (replace app name as needed)
az containerapp logs show \
  --name mmctagent-api \
  --resource-group $AZURE_RESOURCE_GROUP \
  --follow

az containerapp logs show \
  --name mmctagent-mcp \
  --resource-group $AZURE_RESOURCE_GROUP \
  --follow
```

Or query Log Analytics in the Azure portal:

```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "mmctagent-api"
| order by TimeGenerated desc
```

---

## Troubleshooting

### Build fails — base image not found

```bash
# Build the base image manually from repo root
docker build -f Dockerfile.base -t mmctagent-base:latest .
```

### ACR push fails with authentication error

```bash
az acr login --name <acr_name>
docker push <acr_name>.azurecr.io/mmctagent-api:latest
```

### Container App not reachable after deployment

```bash
# Confirm ingress is external
az containerapp ingress show \
  --name mmctagent-api \
  --resource-group $AZURE_RESOURCE_GROUP

# Check the app is Running
az containerapp show \
  --name mmctagent-api \
  --resource-group $AZURE_RESOURCE_GROUP \
  --query "properties.runningStatus"
```

### Container starts but `/health` returns 500

- Verify all required `.env` variables are filled in.
- Confirm Neo4j is reachable at the `NEO4J_URI` in your `.env`.
- Stream logs with the command above and look for startup errors.

### `.env` not found

```bash
cp .env.example .env
# Fill in the required values before redeploying.
```

### `acr_name` validation error on startup

`deploy.py` rejects the placeholder `<your-acr-name>`. Set a real ACR name in
`container_app_config.yaml` before running the script.

### Deployment exits at step 1 with missing env var

```bash
export AZURE_RESOURCE_GROUP=<your-rg>
export AZURE_LOCATION=<your-region>
```

### MCP client cannot connect

- Confirm the MCP Container App is in Running state.
- Use the HTTPS FQDN (not HTTP): `https://<fqdn>/mcp`.
- The MCP server uses streamable-HTTP transport — ensure your client supports it.
