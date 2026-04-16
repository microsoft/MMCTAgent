# Neo4j Deployment

Neo4j is the graph database that stores MMCTAgent's video knowledge graphs. Every
ingested video is stored as a temporal knowledge graph — scenes, objects,
transcriptions, and their temporal relationships are all nodes and edges here.

Two editions and two deployment modes are supported. A single config file drives
everything — `deploy.py` reads it and provisions the right infrastructure automatically.

---

## Quick start

| Goal | `version` | `vm_deployment` | Command |
| --- | --- | --- | --- |
| Local dev, no cloud | `community` | `false` | `python infra/neo4j/deploy.py` |
| Azure VM, production | `community` | `true` | `python infra/neo4j/deploy.py` |
| Azure VM, Enterprise features | `enterprise` | `true` | `python infra/neo4j/deploy.py` |

Edit [neo4j_config.yaml](neo4j_config.yaml) to set those two fields, then run the
command. The script prints the exact `.env` lines to copy on completion.

---

## Files

```text
neo4j/
├── neo4j_config.yaml              ← edit this first (all user decisions live here)
├── deploy.py                      ← run this after editing the config
└── bicep/
    ├── main.bicep                 ← Azure VM template for Community edition
    ├── main.parameters.json       ← Default parameters for Community deployment
    ├── main-enterprise.bicep      ← Azure VM template for Enterprise edition
    ├── main-enterprise.parameters.json
    └── azure.yaml                 ← Azure Developer CLI config (azd up alternative)
```

`deploy.py` selects the correct Bicep template and Docker image based on `version`
in `neo4j_config.yaml` — **no manual template switching is needed**.

---

## Community vs Enterprise

| Capability | Community | Enterprise |
| --- | --- | --- |
| Cost | Free, open-source | Requires Neo4j license |
| MMCTAgent graph pipeline | Full support | Full support |
| Single database (`neo4j`) | Yes | Yes |
| Multiple named databases | No | **Yes** |
| Prometheus metrics endpoint | No | **Yes** (port 2004) |
| APOC Core procedures | Yes | Yes |
| APOC Extended procedures | No | **Yes** (optional) |
| Advanced role-based security | No | **Yes** |
| Clustering / HA | No | **Yes** (separate template needed) |
| Local Docker | `neo4j:5.26.0-community` | Not supported (Azure VM only) |
| Azure VM Bicep | `main.bicep` | `main-enterprise.bicep` |

For most MMCTAgent workloads **community is sufficient**. Choose enterprise when you
need multiple isolated databases, Prometheus metrics, or APOC Extended procedures.

---

## Step 1 — edit `neo4j_config.yaml`

Every field is annotated with its purpose and valid values. Key decisions:

| Field | Local dev | Azure Community | Azure Enterprise |
| --- | --- | --- | --- |
| `version` | `community` | `community` | `enterprise` |
| `vm_deployment` | `false` | `true` | `true` |
| `password` | any strong string | strong password | strong password |
| `license_key` | *(ignored)* | *(ignored)* | blank = eval, or paste `.license` text |
| `data_path` | `./neo4j/data` | *(ignored — uses Azure SSD)* | *(ignored — uses Azure SSD)* |

---

## Step 2 — run `deploy.py`

```bash
# From the repo root:
python infra/neo4j/deploy.py
```

### Pre-requisites — local (`vm_deployment: false`)

```bash
docker info   # Docker must be running; Community edition only locally
```

### Pre-requisites — Azure VM (`vm_deployment: true`)

```bash
az login   # Azure CLI must be installed and authenticated

export AZURE_RESOURCE_GROUP=<existing-resource-group>
export AZURE_LOCATION=<e.g. southeastasia, eastus, westeurope>
export AZURE_ADMIN_PASSWORD=<strong-password-for-the-Ubuntu-VM>
```

---

## Connection strings after deployment

### Local Docker (both editions)

```text
Browser  →  http://localhost:7474/browser/
Bolt     →  bolt://localhost:7687
User     →  neo4j
Password →  <value of 'password' in neo4j_config.yaml>
```

`.env` entries:

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your password>
NEO4J_DATABASE=neo4j
```

### Azure VM (both editions)

```text
Private Bolt    →  bolt://10.0.1.10:7687    (within VNet only)
Private HTTP    →  http://10.0.1.10:7474    (within VNet only)
Public Bolt     →  bolt://<public-ip>:7687  (requires enablePublicIP=true)
Public HTTP     →  http://<public-ip>:7474  (requires enablePublicIP=true)
Metrics (ent.)  →  http://10.0.1.10:2004/metrics  (Enterprise, VNet-scoped)
```

> `deploy.py` prints the actual public IP at the end of a successful VM deployment.
> Use the private IP when connecting from within the same Azure VNet.

`.env` entries (use public IP if connecting from outside Azure):

```text
NEO4J_URI=bolt://10.0.1.10:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your password>
NEO4J_DATABASE=neo4j
```

---

## Bicep templates — what is deployed and what you can tune

Both templates create the same base Azure infrastructure:

| Resource | Default value | Tunable? |
| --- | --- | --- |
| Ubuntu 22.04 VM | `Standard_D16s_v5` | Yes — `vmSize` in parameters file |
| OS disk | 128 GB Premium SSD | No |
| Data disk | 1 TB Premium SSD at `/var/lib/neo4j` | Yes — `dataDiskSizeGB` (256/512/1024/2048) |
| VNet | `10.0.0.0/16`, subnet `10.0.1.0/24` | Yes — `vnetAddressPrefix`, `subnetAddressPrefix` |
| Internal LB private IP | `10.0.1.10` | Yes — `loadBalancerPrivateIP` |
| Public LB + public IP | Disabled | Yes — `enablePublicIP: true` in parameters file |
| Neo4j heap | `31G` | Yes — `heapSize` in parameters file |
| Neo4j page cache | `24G` | Yes — `pagecacheSize` in parameters file |
| APOC Core 5.26.0 | Enabled | Yes — `enableApocPlugin` |
| GDS 2.13.2 | Enabled | Yes — `enableGdsPlugin` |

### Community-only parameters (`main.parameters.json`)

| Parameter | Default | Notes |
| --- | --- | --- |
| `neo4jEdition` | `community` | Change to `enterprise` if deploying enterprise via this template |

### Enterprise-only parameters (`main-enterprise.parameters.json`)

| Parameter | Default | Notes |
| --- | --- | --- |
| `neo4jLicenseKeyBase64` | `""` | Leave blank for eval; paste base64-encoded license for production |
| `enableApocExtended` | `false` | Set `true` to install APOC Extended (`apoc-5.26.0-extended.jar`) |
| `enableMetrics` | `true` | Prometheus metrics on port 2004; set `false` to disable |

> To change a parameter, edit the relevant `.parameters.json` file **or** pass it as
> a `--parameters key=value` override when calling `az deployment group create` directly.

---

## Enterprise features guide

### Multi-database (Enterprise only)

After deploying enterprise, connect Neo4j Browser and run against the `system` database:

```text
:use system
```

```cypher
-- Create a named database
CREATE DATABASE mydb IF NOT EXISTS

-- List all databases and their status
SHOW DATABASES

-- Switch to it in Browser
:use mydb

-- Stop / drop a database
STOP DATABASE mydb
DROP DATABASE mydb IF EXISTS
```

From Python (target `system` for DDL, target the new database for queries):

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://<ip>:7687", auth=("neo4j", "<password>"))

# DDL always runs against the system database
with driver.session(database="system") as session:
    session.run("CREATE DATABASE mmct IF NOT EXISTS")

# Application queries run against the named database
with driver.session(database="mmct") as session:
    session.run("CREATE (n:Video {title: 'example'})")

driver.close()
```

Set `NEO4J_DATABASE=mmct` in your `.env` to point MMCTAgent at the named database.

### Prometheus metrics (Enterprise only)

Metrics are exposed at `http://<private-ip>:2004/metrics` (VNet-scoped only).

To scrape from Azure Monitor or a Prometheus instance in the same VNet:

```yaml
# prometheus.yml scrape config
scrape_configs:
  - job_name: neo4j
    static_configs:
      - targets: ['10.0.1.10:2004']
```

Key metrics available: `neo4j_database_count`, `neo4j_bolt_connections_opened_total`,
`neo4j_page_cache_hits_total`, `neo4j_transaction_committed_total`.

To disable metrics, set `enableMetrics: false` in `main-enterprise.parameters.json`
and redeploy.

### APOC Extended (Enterprise only)

APOC Extended adds procedures not in APOC Core: ML integrations, advanced graph
algorithms, and data export utilities. Disabled by default to keep startup lean.

To enable, set `enableApocExtended: true` in `main-enterprise.parameters.json` before
deploying. Adds `apoc-5.26.0-extended.jar` to `/opt/neo4j/plugins/` on the VM.

---

## Enterprise license

Two modes are supported — both work without any manual steps beyond config:

### Evaluation / developer (default — `license_key: ''`)

`deploy.py` injects `NEO4J_ACCEPT_LICENSE_AGREEMENT=yes` into the VM's systemd unit
and Docker run flags. Neo4j starts without a license file. Suitable for development,
staging, and evaluation use.

### Paid production license (`license_key: '<text>'`)

Paste the raw text of your `.license` file into `license_key` in `neo4j_config.yaml`:

```yaml
neo4j:
  version: enterprise
  license_key: |
    <paste raw contents of your neo4j.license file here>
```

`deploy.py` base64-encodes it and passes it as the `neo4jLicenseKeyBase64` Bicep
parameter. The VM setup script writes it to `/opt/neo4j/licenses/neo4j.license` before
Neo4j starts — no SSH or manual file transfer needed.

Obtain a license at [neo4j.com/licensing](https://neo4j.com/licensing/).

---

## Switching editions locally

```bash
docker rm -f neo4j             # stop and remove current container

# Delete data dir — Community and Enterprise formats are incompatible
rm -rf ./neo4j/data

# Edit neo4j_config.yaml: change version community ↔ enterprise
python infra/neo4j/deploy.py
```

---

## Alternative: `azd up` (Azure Developer CLI)

```bash
cd infra/neo4j
azd up
```

Uses `bicep/azure.yaml`, prompts for passwords interactively and stores them in Azure
Key Vault. `azure.yaml` currently targets `main.bicep` (Community). For Enterprise,
use `deploy.py` directly or update `infra:path` in `azure.yaml` to point to the
enterprise template.

---

## Troubleshooting

### Local Docker

#### Container exits immediately

```bash
docker logs neo4j
# Common cause: data directory permission mismatch
sudo chown -R 7474:7474 ./neo4j/data
```

#### Port already in use

```bash
lsof -i :7687        # find what holds the port
docker rm -f neo4j   # remove any stale container
```

#### Cannot connect from application

- Confirm `NEO4J_URI` in your `.env` matches what `deploy.py` printed.
- Run `docker exec neo4j neo4j status` to confirm the service is up inside the container.

#### Switching editions — container refuses to start

Community and Enterprise store data in incompatible formats. Delete `./neo4j/data`
before switching, or point `data_path` to a fresh directory.

### Azure VM

#### Deployment times out

The Custom Script Extension (Neo4j setup) takes 8–12 minutes. Check the extension
logs in the Azure portal: VM → Extensions → `neo4j-setup` (or `neo4j-enterprise-setup`).

Or via CLI:

```bash
az vm extension show \
  --resource-group <rg> \
  --vm-name neo4j-ent-1 \
  --name neo4j-enterprise-setup \
  --query "provisioningState" -o tsv
```

#### Cannot reach Neo4j after deployment

- The private IP `10.0.1.10` is the internal LB frontend — **only reachable from
  within the same VNet**. It will not open in a local browser.
- For browser access from your machine, set `enablePublicIP: true` in the parameters
  file and redeploy. `deploy.py` prints the public IP on completion.
- Verify NSG rules allow ports 7687 and 7474 from Internet.

#### Deploy output shows wrong public IP

`deploy.py` names deployments `neo4j-community` or `neo4j-enterprise` so outputs
always come from the current run. If you see a stale IP, check the actual LB IP:

```bash
az network public-ip show \
  --resource-group <rg> \
  --name neo4j-ent-lb-pip \
  --query ipAddress -o tsv
```

#### Enterprise: Neo4j crash-loops on startup

Check the Neo4j log on the VM:

```bash
az vm run-command invoke \
  --resource-group <rg> --vm-name neo4j-ent-1 \
  --command-id RunShellScript \
  --scripts "sudo cat /var/lib/neo4j/logs/neo4j.log | tail -40"
```

Common cause: missing directory permissions under `/opt/neo4j/`. Fix:

```bash
az vm run-command invoke \
  --resource-group <rg> --vm-name neo4j-ent-1 \
  --command-id RunShellScript \
  --scripts "sudo mkdir -p /opt/neo4j/metrics /opt/neo4j/run && sudo chown neo4j:neo4j /opt/neo4j/metrics /opt/neo4j/run && sudo systemctl restart neo4j"
```

#### Enterprise: License error on start

- Blank `license_key` (eval mode): confirm the env var is set in the unit:
  `sudo systemctl cat neo4j | grep LICENSE`
- Paid license: confirm the file was written:
  `sudo ls -la /opt/neo4j/licenses/`

#### Enterprise: `CREATE DATABASE` fails

Ensure you are connected to the `system` database, not `neo4j`:

```text
:use system
CREATE DATABASE mydb IF NOT EXISTS
```

#### Password authentication failed

The `neo4jPassword` Bicep parameter must match `password` in `neo4j_config.yaml`.
To reset without redeploying, SSH into the VM (requires adding an SSH NSG rule) or
use Azure Serial Console:

```bash
sudo /opt/neo4j/bin/neo4j-admin dbms set-initial-password <new-password>
sudo systemctl restart neo4j
```
